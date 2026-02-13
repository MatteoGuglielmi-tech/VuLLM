from rich.table import Table
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from .datatypes import PromptPhase, AssumptionMode, PromptVersion
from .dataset_handler import DatasetHandler
from .model_handler import ModelHandler
from .custom import WeightedCoTTrainer
from ..utilities import is_main_process, build_table, rich_panel, RichColors

import logging
import os
import gc
import torch
import wandb

from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from enum import StrEnum
from typing import Any, Optional, cast
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from datasets import DatasetDict


logger = logging.getLogger(name=__name__)


class TrainingStrategy(StrEnum):
    FAST = "fast"  # Quick iteration with early stopping
    EXPLORE = "explore"  # Full warm restarts exploration
    BALANCED = "balanced"  # Moderate approach


@dataclass
class StrategyConfig:
    """Configuration overrides for training strategies."""

    epochs: int
    lr_scheduler_type: str
    use_early_stopping: bool
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None
    warmup_ratio: float = 0.05


class FineTuningHandler:
    def __init__(
        self,
        dataset_handler_class: type[DatasetHandler],
        model_loader_class: type[ModelHandler],
        # -- dataset --
        dataset_path: str,
        formatted_dataset_dir: Path,
        num_cpus: int,
        target_vulnerable_ratio: Optional[float],
        prompt_mode: PromptPhase,
        assumption_mode: AssumptionMode,
        add_hierarchy: bool,
        prompt_version: PromptVersion,
        # -- model loading --
        base_model_name: str,
        chat_template: str,
        max_seq_length: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        use_loftq: bool,
        # -- fine-tuning HP --
        learning_rate: float,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        gradient_accumulation_steps: int,
        weight_decay: float,
        max_grad_norm: float,
        strategy: TrainingStrategy,
        epochs: Optional[int] = None,
        use_early_stopping: Optional[bool] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: Optional[float] = None,
        resume_from_checkpoint: Optional[Path] = None,
        warmup_ratio: Optional[float] = None,
        logging_steps: int = 100,
        use_weighted_cot_trainer: bool = False,
        use_deepspeed: bool = False,
        # -- general --
        debug: bool = False,
    ):
        """
        Initialize the fine-tuning handler.

        Parameters
        ----------
        dataset_handler_class: type[DatasetHandler]
            DatasetHandler class object/reference

        model_loader_class: type[ModelHandler]
            ModelHandler class object/reference

        dataset_path: str
            Path string leading to source dataset

        formatted_dataset_dir: Path
            Output directory where to save formatted dataset (after `apply_chat_template`) to

        num_cpus: int
            Number of cpu processes available for this task

        target_vulnerable_ratio: float, optional, default=None
            Specifies whether to duplicate vulnearble samples if unbalance in SFT

        prompt_mode: str
            Ablation mode to use

        assumption_mode: str
            whether to add positive or negative assumptions (or none)

        add_hierarchy: bool
            whether to add cwe hierarchy guidelines to system prompt

        prompt_version: PromptVersion
            Prompt version to use

        base_model_name: str
            Name of the model to load (from huggingface or unsloth)

        chat_template: str
            Chat template to apply to the tokenizer for correct tokenization and formatting

        max_seq_length: int
            Maximum sequence length the model needs to handle (input+output)

        lora_rank: int
            LoRA rank to use when patching model (reflects the capacity to capture complex patterns)

        lora_alpha: int
            LoRA parameter which scales magnitude of low-rank updates to the original weights.
            Low alpha (α=8, r=16 → scale=0.5):
                - LoRA changes are subtle
                - Original model behavior mostly preserved
                - Conservative adaptation

            High alpha (α=64, r=16 → scale=4.0):
                - LoRA changes are strong
                - Can override original model behavior more
                - Aggressive adaptation

        lora_dropout: float
            LoRA droupout regularization parameter, it specifies the amount of values (in %) to set to zero
            during an update.

        use_rslora: bool
            Enable smarter scaling for the LoRA weights (no lora_alpha needed)

        use_loftq: bool
            Enable smarter initialization for the LoRA weights (requires loading the entire module in memory)

        learning_rate: float
            Learning rate value to use (the higher the steeper the learning curve)

        per_device_train_batch_size: int
            Real batch size capacity

        per_device_eval_batch_size: int
            Real batch size capacity for evaluation

        gradient_accumulation_steps: int
            Represents how many steps to take before performing an update.
            effective_batch_size = per_device_batch_size × gradient_accumulation_steps × num_gpus

        weight_decay: float
            L2 regularization penatly term

        max_grad_norm: float
            Value for gradient clipping

        strategy: TrainingStrategy, options={"fast", "explore", "balanced"}
            Training strategy preset:
            - "fast": Quick iteration with early stopping (4 epochs, cosine scheduler)
            - "explore": Full warm restarts exploration (7 epochs, no early stopping)
            - "balanced": Moderate approach (5 epochs, gentle early stopping)

        epochs: int, optional, default=None
            Override strategy default epochs

        use_early_stopping: bool, optional, default=None
            Override strategy default early stopping behavior

        early_stopping_patience: int, optional, default=None
            Evaluation steps to wait without improvement before stopping

        early_stopping_threshold: Optional[float]= None
            Threshold defining whether an imporvement took place or not

        resume_from_checkpoint: Optional[Path] = None
            If provided, it resumes a previously interrupted fine-tuning run from specified checkpoint dir path.

        warmup_ratio: Optional[float] = None
            Scheduler parameter. Ratio of total training steps used for a linear warmup from 0 to learning_rate

        logging_steps: int = 100
            How often logging happens

        use_weighted_cot_trainer: bool, default=False
            Whether to use custom Trainer which weights parts of the prompt differently

        use_deepspeed: bool, default=False
            Whether deepspeed is used for the run or not

        debug: bool, default=False
            Enable debug logging
        """

        self.dataset_handler_class = dataset_handler_class
        self.dataset_path = dataset_path
        self.formatted_dataset_dir = formatted_dataset_dir
        self.num_cpus = num_cpus
        self.target_vulnerable_ratio = target_vulnerable_ratio
        self.prompt_mode = prompt_mode
        self.assumption_mode = assumption_mode
        self.add_hierarchy = add_hierarchy
        self.prompt_version = prompt_version

        self.model_loader_class = model_loader_class
        self.base_model_name = base_model_name
        self.chat_template = chat_template
        self.lora_r = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_seq_length = max_seq_length
        self.use_rslora = use_rslora
        self.use_loftq = use_loftq

        self.lr = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.strategy_name = strategy
        self.strategy_config = self._get_strategy_defaults(strategy)
        self.epochs = epochs if epochs is not None else self.strategy_config.epochs
        self.lr_scheduler_type = self.strategy_config.lr_scheduler_type
        self.use_early_stopping = (
            use_early_stopping
            if use_early_stopping is not None
            else self.strategy_config.use_early_stopping
        )
        self.early_stopping_patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else self.strategy_config.early_stopping_patience
        )
        self.early_stopping_threshold = (
            early_stopping_threshold
            if early_stopping_threshold is not None
            else self.strategy_config.early_stopping_threshold
        )
        self.resume_from_checkpoint = resume_from_checkpoint
        self.warmup_ratio = (
            warmup_ratio
            if warmup_ratio is not None
            else self.strategy_config.warmup_ratio
        )

        self.logging_steps = logging_steps
        self.wd = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_weighted_trainer = use_weighted_cot_trainer
        self.use_deepspeed = use_deepspeed

        self.debug = debug
        self._dataset_dict = None
        self._base_tokenizer = None

    def _strat_settings(self, strategy: TrainingStrategy) -> Table:
        tb_dict = {
            "TrainingStrategy": strategy,
            "Epochs": self.epochs,
            "LR scheduler": self.lr_scheduler_type,
            "Early stopping?": self.use_early_stopping,
            "Early stopping patience": (
                self.early_stopping_patience if self.use_early_stopping else "None"
            ),
            "Early stopping threshold": (
                self.early_stopping_threshold if self.use_early_stopping else "None"
            ),
            "Warm-up ratio (%)": self.warmup_ratio,
        }
        tb = build_table(
            data=tb_dict, columns=["Parameter", "Value"], title="Fine-tuning strategy"
        )
        return tb

    @staticmethod
    def _get_strategy_defaults(strategy: TrainingStrategy) -> StrategyConfig:
        """Get configuration for a training strategy."""

        configs = {
            TrainingStrategy.FAST: StrategyConfig(
                epochs=4,
                lr_scheduler_type="cosine",
                use_early_stopping=True,
                early_stopping_patience=3,
                early_stopping_threshold=0.001,
                warmup_ratio=0.05,
            ),
            TrainingStrategy.EXPLORE: StrategyConfig(
                epochs=7,
                lr_scheduler_type="cosine_with_restarts",
                use_early_stopping=False,
                warmup_ratio=0.03,
            ),
            TrainingStrategy.BALANCED: StrategyConfig(
                epochs=5,
                lr_scheduler_type="cosine_with_restarts",
                use_early_stopping=True,
                early_stopping_patience=8,
                early_stopping_threshold=0.0005,
                warmup_ratio=0.05,
            ),
        }

        return configs[strategy]

    def setup_paths(self) -> None:
        provider, model_id = self.base_model_name.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")

        common_suffix: str = os.path.join(provider, model_id, date, time)
        history_dir: str = "./checkpoints"

        self.lora_best_model_dir: str = os.path.join(
            history_dir,
            "best_model",
            self.base_model_name,
            self.prompt_mode,
            self.assumption_mode,
            date,
            time
        )
        self.checkpoint_dir: str = os.path.join(history_dir, common_suffix)
        self.output_dir: str = os.path.join("./results", f"{model_id}_{date}_{time}")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.lora_best_model_dir, exist_ok=True)

    def _prepare_dataset_with_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Prepare dataset using the provided tokenizer."""

        if self._dataset_dict is not None:
            logger.info("♻️ Reusing cached dataset")
            return self._dataset_dict

        dataset_handler = self.dataset_handler_class(
            dataset_path=self.dataset_path,
            formatted_dataset_dir=self.formatted_dataset_dir,
            tokenizer=tokenizer,
            num_cpus=self.num_cpus,
            debug_mode=self.debug,
            prompt_phase=self.prompt_mode,
            assumption_mode=self.assumption_mode,
            add_cwe_guidelines=self.add_hierarchy,
            prompt_version=self.prompt_version
        )
        self._dataset_dict = dataset_handler.run_pipeline(
            target_vulnerable_ratio=self.target_vulnerable_ratio
        )

        return self._dataset_dict

    def get_instruction_response_parts(self) -> tuple[str, str]:
        """Get instruction and response delimiters for train_on_responses_only."""

        model_name_lower = self.base_model_name.lower()

        # DeepSeek
        if "deepseek" in model_name_lower:
            logger.info("🎯 Using DeepSeek delimiters")
            return ("### Instruction:", "### Response:")

        # CodeLlama
        elif "codellama" in model_name_lower:
            logger.info("🎯 Using Llama 2 delimiters")
            return ("[INST]", "[/INST]")

        # Auto-detect
        else:
            chat_template = getattr(self.tokenizer, "chat_template", "")

            if not chat_template:
                raise ValueError("Tokenizer has no chat template!")

            template_lower = chat_template.lower()

            if "### instruction" in template_lower and "### response" in template_lower:
                logger.info("🎯 Detected ### format from template")
                return ("### Instruction:", "### Response:")

            elif "[inst]" in template_lower:
                logger.info("🎯 Detected Llama 2/Mistral format from template")
                return ("[INST]", "[/INST]")

            elif "start_header_id" in template_lower:
                logger.info("🎯 Detected Llama 3+ format from template")
                return (
                    "<|start_header_id|>user<|end_header_id|>",
                    "<|start_header_id|>assistant<|end_header_id|>",
                )

            elif "im_start" in template_lower:
                logger.info("🎯 Detected Qwen/ChatML format from template")
                return ("<|im_start|>user", "<|im_start|>assistant")

            elif "<|user|>" in template_lower:
                logger.info("🎯 Detected Zephyr/Phi format from template")
                return ("<|user|>", "<|assistant|>")

            elif "start_of_turn" in template_lower:
                logger.info("🎯 Detected Gemma format from template")
                return ("<start_of_turn>user", "<start_of_turn>model")

            else:
                raise ValueError(
                    f"Could not detect chat template format.\n"
                    f"Model: {self.base_model_name}\n"
                    f"Model type: {getattr(self.base_model.config, 'model_type', 'unknown')}\n"  # type: ignore[reportAttributeAccessIssue]
                    f"Template preview: {chat_template[:300]}"
                )

    def apply_response_only_training(self, trainer: SFTTrainer | WeightedCoTTrainer):
        """
        Apply train_on_responses_only to the trainer.

        Automatically detects the correct delimiters based on the model.

        Parameters
        ----------
        trainer : Trainer
            The trainer instance

        Returns
        -------
        Trainer
            Trainer with response-only training applied
        """
        instruction_part, response_part = self.get_instruction_response_parts()

        logger.info(f"🎯 Applying train_on_responses_only")
        logger.info(f"   Instruction delimiter: {repr(instruction_part)}")
        logger.info(f"   Response delimiter: {repr(response_part)}")

        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            tokenizer=self.tokenizer,
            num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        )

        logger.info("✅ Response-only training applied")

        return trainer

    def wandb_init(self) -> None:
        start_time = datetime.now().strftime("%b %d '%y %H:%M")
        job_name = os.environ.get("SLURM_JOB_NAME", "unknown")
        job_id = os.environ.get("SLURM_JOB_ID", "0")

        wandb.init(
            project="huggingface",
            name=f"{self.base_model_name.split('/')[-1]}-{job_id}",
            config={
                "job_name": job_name,
                "job_id": job_id,
                "start_time": start_time,
                "strategy": self.strategy_name,
                "epochs": self.epochs,
                "train_on_responses_only": True,
                "use_early_stopping": self.use_early_stopping,
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
                "warmup_ratio": self.warmup_ratio,
                "best_model_directory": self.lora_best_model_dir,
                "prompt_mode": self.prompt_mode,
                "assumption_mode": self.assumption_mode,
                "RsLora?": self.use_rslora,
                "LoftQ?": self.use_loftq,
                "guidelines?": self.add_hierarchy
            },
        )

    def wandb_close(self) -> None:
        wandb.finish()

    def create_trainer_with_params(self) -> SFTTrainer | WeightedCoTTrainer:
        """Create a new model and trainer with specific hyperparameters using your existing class."""

        model_loader = self.model_loader_class(
            base_model_name=self.base_model_name,
            chat_template=self.chat_template,
            max_seq_length=self.max_seq_length,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_rslora=self.use_rslora,
            use_loftq=self.use_loftq,
            use_deepspeed=self.use_deepspeed,
        )

        model_loader._load_base_model()
        model_loader.patch_model()  # LoRA patch
        model, self.tokenizer = model_loader.obtain_components()

        dataset_dict: DatasetDict = self._prepare_dataset_with_tokenizer(
            tokenizer=self.tokenizer
        )

        if is_main_process():
            print(f"📊 Model trainable parameters:")
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()  # type: ignore

        sft_args: SFTConfig = self._ft_args(dataset_dict=dataset_dict)
        callbacks = self.get_callbacks()

        if not self.use_weighted_trainer:
            trainer = SFTTrainer(
                model=model,  # type: ignore
                processing_class=self.tokenizer,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                args=sft_args,
                callbacks=callbacks,
            )
        else:
            trainer = WeightedCoTTrainer(
                model=model,
                tokenizer=self.tokenizer,
                args=sft_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                reasoning_weight=1.0,
                answer_weight=1.5,
                answer_marker="Final Answer:",  # prompt-specific!
                callbacks=callbacks,
            )

        trainer = self.apply_response_only_training(trainer=trainer)

        return cast(SFTTrainer, trainer)

    def calculate_training_steps(self, train_dataset_size: int) -> dict[str, int]:
        """
        Calculate training and warmup steps based on dataset size.

        Parameters
        ----------
        train_dataset_size : int
            Length of the training dataset

        Returns
        -------
        dict[str, int]
            Dictionary with 'total_steps', 'warmup_steps', 'steps_per_epoch'
        """
        steps_per_epoch = train_dataset_size // (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )

        total_steps = steps_per_epoch * self.epochs
        warmup_steps = int(self.warmup_ratio * total_steps)

        if self.debug:
            tb_dict = {
                "Dataset size": train_dataset_size,
                "Batch size": self.per_device_train_batch_size,
                "Gradient accumulation": self.gradient_accumulation_steps,
                "Steps per epoch": steps_per_epoch,
                "Total epochs": self.epochs,
                "Total steps": total_steps,
                "Warmup ratio": f"{self.warmup_ratio} ({warmup_steps} steps)",
            }
            global steps_tb
            steps_tb = build_table(data=tb_dict, columns=["Info", "Value"], title="Fine-tuning parameters")
            del tb_dict

        return {
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "steps_per_epoch": steps_per_epoch,
        }

    def calculate_eval_steps(
        self, train_dataset_size: int, evals_per_epoch: int = 4
    ) -> int:
        """
        Calculate optimal eval_steps for step-based evaluation.

        Parameters
        ----------
        train_dataset_size : int
            Length of the TRAINING dataset only
        evals_per_epoch : int
            Desired number of evaluations per epoch (default: 4)

        Returns
        -------
        int
            Number of training steps between evaluations
        """
        steps_per_epoch = train_dataset_size // (
            self.per_device_train_batch_size * self.gradient_accumulation_steps
        )

        eval_steps = max(steps_per_epoch // evals_per_epoch, 50)
        actual_evals_per_epoch = steps_per_epoch / eval_steps

        if self.debug:
            tb_dict = {
                "Steps per epoch": steps_per_epoch,
                "Target evals/epoch": evals_per_epoch,
                "Eval steps": eval_steps,
                "Actual evals/epoch": f"{actual_evals_per_epoch:.1f}",
                "Total evaluations": int(actual_evals_per_epoch * self.epochs),
            }

            eval_steps_tb = build_table(
                data=tb_dict, columns=["Info", "Value"], title="Evaluation parameters"
            )
            rich_panel(
                tables=[steps_tb, eval_steps_tb],
                panel_title="Learning parameters",
                border_style=RichColors.BRIGHT_GREEN
            )
            del tb_dict, eval_steps_tb

        return eval_steps

    def _ft_args(self, *, dataset_dict: DatasetDict) -> SFTConfig:
        """Configures the training parameters for SFTTrainer."""

        if self.epochs <= 0:
            logger.warning(
                "No training duration specified. Defaulting to StrategyConfig.epochs."
            )
            self.epochs = self.strategy_config.epochs

        train_size = len(dataset_dict["train"])
        self.calculate_training_steps(train_size)
        eval_steps = self.calculate_eval_steps(train_size)
        bfloat16_supported: bool = is_bfloat16_supported()

        sft_config_params: dict[str, Any] = {
            # optimized kernel
            "use_liger_kernel": True if not "deepseek" in self.base_model_name.lower() else False,
            # paths
            "output_dir": self.checkpoint_dir,
            # model related args
            "fp16": not bfloat16_supported,
            "bf16": bfloat16_supported,
            "max_length": self.max_seq_length,
            # fine-tuning hps
            "num_train_epochs": self.epochs,
            # "train"
            "gradient_checkpointing": True,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.lr,
            "weight_decay": self.wd,
            "max_grad_norm": self.max_grad_norm,
            # warmup
            "warmup_ratio": self.warmup_ratio,
            # data handling
            "packing": False,
            "dataset_text_field": "text",
            # evaluation step
            "eval_strategy": "steps",
            "eval_steps": eval_steps,
            # checkpointing
            "save_strategy": "steps",
            "save_steps": eval_steps,
            "save_total_limit": 3,
            # best model selection
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            # logging
            "logging_steps": self.logging_steps,
            "report_to": "wandb",
            # reproducibility
            "seed": 3407,
        }

        # Non-DeepSpeed specific optimizations
        if not self.use_deepspeed:
            sft_config_params["optim"] = "paged_adamw_8bit"
            sft_config_params["lr_scheduler_type"] = self.lr_scheduler_type

            if self.lr_scheduler_type == "cosine_with_restarts" and self.epochs < 6:
                logger.warning(
                    f"Using cosine_with_restarts with only {self.epochs} epochs. "
                    "Consider 6+ epochs for multiple restart cycles to fully benefit."
                )

        return SFTConfig(**sft_config_params)

    def get_callbacks(self) -> Optional[list]:
        """Build callbacks list based on configuration."""
        callbacks = []

        if self.use_early_stopping:
            from transformers import EarlyStoppingCallback

            assert self.early_stopping_patience is not None

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold,
                )
            )
        else:
            pass

        return callbacks if callbacks else None

    def debug_trainer(self, trainer):

        if is_main_process():
            print("\n" + "=" * 50)
            print("TESTING TRAINER DATA LOADING")
            print("=" * 50)

            # Get one batch from the trainer's dataloader
            try:
                # This will trigger the actual data processing pipeline
                train_dataloader = trainer.get_train_dataloader()
                first_batch = next(iter(train_dataloader))

                print(f"✓ Successfully loaded first batch!")
                print(f"Batch keys: {first_batch.keys()}")
                print(f"Input IDs shape: {first_batch['input_ids'].shape}")
                print(f"Labels shape: {first_batch['labels'].shape}")

                # Check if labels are properly masked for assistant-only loss
                # -100 tokens should be present for system/user parts
                labels = first_batch["labels"][0]
                num_masked = (labels == -100).sum().item()
                num_unmasked = (labels != -100).sum().item()
                print(f"✓ Masked tokens (system/user): {num_masked}")
                print(f"✓ Unmasked tokens (assistant): {num_unmasked}")

                print(self.tokenizer.decode(first_batch["input_ids"][0]))
                space = self.tokenizer(" ", add_special_tokens=False).input_ids[0]
                print(
                    self.tokenizer.decode(
                        [space if x == -100 else x for x in first_batch["labels"]]
                    )
                )

                # Sequence length check
                print(f"\nMax sequence length setting: {trainer.args.max_length}")
                print(f"Actual batch max length: {first_batch['input_ids'].shape[1]}")

                # Check for padding
                padding_mask = first_batch["attention_mask"][0]
                num_padding_tokens = (padding_mask == 0).sum().item()
                num_real_tokens = (padding_mask == 1).sum().item()

                print(f"Real tokens: {num_real_tokens}")
                print(f"Padding tokens: {num_padding_tokens}")

                # Check if any sequences are truncated
                if first_batch["input_ids"].shape[1] >= trainer.args.max_length:
                    print("⚠️  Warning: Sequences may be truncated at max_length")
                else:
                    print(
                        f"✓ Sequences fit within max_seq_length ({trainer.args.max_length})"
                    )

                # Final pre-training checklist
                print("\n" + "=" * 60)
                print("FINAL PRE-TRAINING CHECKLIST")
                print("=" * 60)

                print(
                    f"✓ Batch loading successful: shape {first_batch['input_ids'].shape}"
                )
                print(
                    f"✓ Assistant-only loss active: {num_masked} masked, {num_unmasked} unmasked tokens"
                )
                print(
                    f"✓ Masking ratio: {num_masked/(num_masked+num_unmasked)*100:.1f}% masked"
                )
                print(f"✓ Training for {self.epochs} epochs")
                print(f"✓ Batch size: {self.per_device_train_batch_size}")
                print(f"✓ Gradient accumulation: {self.gradient_accumulation_steps}")
                print(
                    f"✓ Effective batch size: {self.per_device_train_batch_size * self.gradient_accumulation_steps}"
                )
                print(f"✓ Learning rate: {self.lr}")

                print("\n🚀 Ready to start training!")
                print("=" * 60)

            except Exception as e:
                print(f"✗ Error loading batch: {e}")
                import traceback

                traceback.print_exc()

            print("=" * 50)

    def fine_tune(self):
        self.setup_paths()
        trainer = self.create_trainer_with_params()

        self.wandb_init()
        logger.info("🚀 Starting training...")
        trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)  # type: ignore[reportArgumentType]
        logger.info("✅ Training completed.")
        self.wandb_close()

        if self.debug:
            # Check trainer state
            tb_dict = {
                "Best metric": trainer.state.best_metric,
                "Best model checkpoint": trainer.state.best_model_checkpoint,
                "Total steps": trainer.state.global_step / trainer.state.max_steps,
                "Early stopped": trainer.state.global_step < trainer.state.max_steps,
            }

            rich_panel(
                tables=[build_table(data=tb_dict, columns=["Parameter", "Value"])],
                panel_title="Training State",
                border_style=RichColors.SKY_BLUE2,
            )
            del tb_dict

        # save the LoRA adapters of best model
        logger.info(f"💾 Saving best model to {self.base_model_name}...")
        trainer.save_model(output_dir=self.lora_best_model_dir)

        if self.tokenizer:
            self.tokenizer.save_pretrained(self.lora_best_model_dir)
            logger.info(f"✅ Tokenizer saved to {self.lora_best_model_dir}")

        logger.info(f"✅ Best LoRA adapters saved to {self.lora_best_model_dir}")

        # Cleanup
        logger.debug("🧹 Cleaning up training resources...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("✅ Cleanup complete. VRAM released.")

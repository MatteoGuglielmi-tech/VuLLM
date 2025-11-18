from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from .dataset_handler import DatasetHandler
from .model_handler import ModelHandler
from .custom import WeightedCoTTrainer
from ..loader_config import Loader
from ..utilities import get_instruction_response_parts, is_main_process, build_table, rich_panel

import logging
import os
import gc
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from enum import Enum
from typing import Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from datasets import DatasetDict


logger = logging.getLogger(name=__name__)


class TrainingStrategy(Enum):
    """Training presets"""
    FAST = "fast"          # Quick iteration with early stopping
    EXPLORE = "explore"    # Full warm restarts exploration
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
        gradient_accumulation_steps: int,
        weight_decay: float,

        strategy: TrainingStrategy = TrainingStrategy.EXPLORE,
        epochs: Optional[int] = None,
        use_early_stopping: Optional[bool] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: Optional[float]= None,
        warmup_ratio: Optional[float] = None,

        logging_steps: int = 100,
        use_weighted_cot_trainer: bool = False,
        use_deepspeed: bool = False,
        # -- general --
        debug: bool = False,
    ):

        self.dataset_handler_class = dataset_handler_class
        self.dataset_path = dataset_path
        self.formatted_dataset_dir = formatted_dataset_dir
        self.num_cpus = num_cpus

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
        self.gradient_accumulation_steps = gradient_accumulation_steps


        self.strategy_config = self._get_strategy_defaults(strategy)
        self.epochs = epochs if epochs is not None else self.strategy_config.epochs
        self.lr_scheduler_type = self.strategy_config.lr_scheduler_type
        self.use_early_stopping = (
            use_early_stopping
            if use_early_stopping is not None
            else self.strategy_config.epochs
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
        self.warmup_ratio = (
            warmup_ratio if warmup_ratio is not None else self.strategy_config.warmup_ratio
        )

        self.logging_steps = logging_steps
        self.wd = weight_decay
        self.use_weighted_trainer = use_weighted_cot_trainer
        self.use_deepspeed = use_deepspeed

        self.debug = debug
        self._dataset_dict = None
        self._base_tokenizer = None

        self._log_settings(strategy=strategy)

    def _log_settings(self, strategy: TrainingStrategy) -> None:
        tb_dict = {
            "TrainingStrategy": strategy.value,
            "Epochs": self.epochs,
            "LR scheduler": self.lr_scheduler_type,
            "Early stopping?": self.use_early_stopping,
            "Early stopping patience": self.early_stopping_patience if self.use_early_stopping else "None",
            "Early stopping threshold": self.early_stopping_threshold if self.use_early_stopping else "None",
            "Warm-up ratio (%)": self.warmup_ratio,
        }
        tb = build_table(data=tb_dict, columns=["Parameter", "Value"])
        rich_panel(
            tables=tb,
            panel_title="🗺️ Fine-tuning strategy",
            border_style="cornflower_blue",
        )

        del tb_dict, tb

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

        self.lora_best_model_dir: str = os.path.join(history_dir, "best_model")
        self.checkpoint_dir: str = os.path.join(history_dir, common_suffix)
        self.output_dir: str = os.path.join("./results", f"{model_id}_{date}_{time}")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.lora_best_model_dir, exist_ok=True)

    def _prepare_dataset_with_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Prepare dataset using the provided tokenizer."""

        if self._dataset_dict is not None:
            logger.info("♻️ Reusing cached dataset")
            return self._dataset_dict

        with Loader(
            "📚 Preparing dataset with tokenizer...",
            "✅ Dataset prepared and cached for reuse",
            logger=logger,
        ):
            dataset_handler = self.dataset_handler_class(
                dataset_path=self.dataset_path,
                formatted_dataset_dir=self.formatted_dataset_dir,
                tokenizer=tokenizer,
                num_cpus=self.num_cpus,
                debug_mode=self.debug,
            )
            self._dataset_dict = dataset_handler.run_pipeline()
        return self._dataset_dict

    def formatting_prompts_func(self, examples):
        """Format messages into text using the tokenizer's chat template.
        Handles both batched and single-conversation inputs robustly.
        """

        if "messages" not in examples:
            raise ValueError("Dataset must contain 'messages' field")

        conversations = examples["messages"]

        if not isinstance(conversations, list):
            raise TypeError(f"Expected 'messages' to be a list, got {type(conversations)}")

        if len(conversations) == 0: # check format
            return []

        # if first element is a dict with 'role' key, it's a single conversation
        if isinstance(conversations[0], dict) and "role" in conversations[0]:
            # Single conversation format: [{"role": "system", ...}, {"role": "user", ...}]
            conversations = [conversations] # enforce batch of 1

        texts = []
        for idx, conversation in enumerate(conversations):
            try:
                text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            except Exception as e:
                print(f"Error processing conversation {idx}: {e}")
                print(f"Conversation type: {type(conversation)}")
                print(f"Conversation: {conversation}")
                raise

        return texts

    def debug_fmt_dataset_structure(self, dataset_dict: DatasetDict):
        if is_main_process():
            print("=" * 50)
            print("DEBUGGING DATASET STRUCTURE")
            print("=" * 50)

            # Check a single sample
            sample = dataset_dict["train"][0]
            print(f"\n1. Single sample keys: {sample.keys()}")
            print(f"2. Type of 'messages': {type(sample['messages'])}")
            print(f"3. Length of messages: {len(sample['messages'])}")
            print(f"4. First message: {sample['messages'][0]}")
            print(f"5. All messages:\n{sample['messages']}")

            # Check batched data (how SFTTrainer will see it)
            batch = dataset_dict["train"][:2]
            print(f"\n6. Batch keys: {batch.keys()}")
            print(f"7. Type of batch['messages']: {type(batch['messages'])}")
            print(f"8. Length of batch['messages']: {len(batch['messages'])}")
            print(f"9. Type of first conversation: {type(batch['messages'][0])}")
            print(f"10. First conversation:\n{batch['messages'][0]}")

            # Test the formatting function
            print("\n" + "=" * 50)
            print("TESTING FORMATTING FUNCTION")
            print("=" * 50)
            try:
                formatted_texts = self.formatting_prompts_func(batch)
                print(f"✓ Formatting successful!")
                print(f"✓ Generated {len(formatted_texts)} texts")
                print(f"\nFirst formatted text (truncated):\n{formatted_texts[0][:500]}...")
            except Exception as e:
                print(f"✗ Formatting failed: {e}")
                import traceback
                traceback.print_exc()

            print("=" * 50)

    def create_trainer_with_params(self) -> SFTTrainer|WeightedCoTTrainer:
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
            use_deepspeed=self.use_deepspeed
        )

        model_loader._load_base_model()
        model_loader.patch_model()  # LoRA patch
        model, self.tokenizer = model_loader.obtain_components()

        dataset_dict: DatasetDict = self._prepare_dataset_with_tokenizer(tokenizer=self.tokenizer)

        if self.debug:
            self.debug_fmt_dataset_structure(dataset_dict)

        if is_main_process():
            print(f"📊 Model trainable parameters:")
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters() # type: ignore

        sft_args: SFTConfig = self._ft_args(dataset_dict=dataset_dict)
        callbacks = self.get_callbacks()

        if not self.use_weighted_trainer:
            trainer = SFTTrainer(
                model=model, # type: ignore
                processing_class=self.tokenizer,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                args=sft_args,
                callbacks=callbacks
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
                callbacks=callbacks
            )

        instruction_part, response_part = get_instruction_response_parts(tokenizer=self.tokenizer)
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            tokenizer=self.tokenizer,
            num_proc=int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        )

        logger.info(f"✅ Trainer configured with response-only training")
        logger.info(f"   Instruction marker: {instruction_part}")
        logger.info(f"   Response marker: {response_part}")

        return trainer

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
            tb = build_table(data=tb_dict, columns=["Info", "Value"])
            rich_panel(tb, panel_title="Training steps calculation", border_style="medium_spring_green")
            del tb_dict, tb

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
                "Total evaluations": int(actual_evals_per_epoch * self.epochs)
            }
            tb = build_table(data=tb_dict, columns=["Info", "Value"])
            rich_panel(tb, panel_title="Evaluation steps calculation", border_style="medium_spring_green")
            del tb_dict, tb

        return eval_steps

    def _ft_args(self, *, dataset_dict: DatasetDict) -> SFTConfig:
        """Configures the training parameters for SFTTrainer."""

        if self.epochs <= 0:
            logger.warning("No training duration specified. Defaulting to StrategyConfig.epochs.")
            self.epochs = self.strategy_config.epochs

        train_size = len(dataset_dict["train"])
        self.calculate_training_steps(train_size)
        eval_steps = self.calculate_eval_steps(train_size)
        bfloat16_supported: bool = is_bfloat16_supported()

        sft_config_params: dict[str, Any] = {
            # optimized kernel
            "use_liger_kernel": True,

            # paths
            "output_dir": self.checkpoint_dir,
            "run_name": f"{self.base_model_name.split('/')[-1]}-epochs-{self.epochs}",

            # model related args
            "fp16": not bfloat16_supported,
            "bf16": bfloat16_supported,
            "max_length": self.max_seq_length,

            # fine-tuning hps
            "num_train_epochs": self.epochs,

            # "train"
            "gradient_checkpointing": True,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,

            "learning_rate": self.lr,
            "weight_decay": self.wd,
            "max_grad_norm": 0.3,

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

            #logging
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
                space = self.tokenizer(" ", add_special_tokens = False).input_ids[0]
                print(self.tokenizer.decode([space if x == -100 else x for x in first_batch["labels"]]))

                # Sequence length check
                print(f"\nMax sequence length setting: {trainer.args.max_length}")
                print(f"Actual batch max length: {first_batch['input_ids'].shape[1]}")

                # Check for padding
                padding_mask = first_batch['attention_mask'][0]
                num_padding_tokens = (padding_mask == 0).sum().item()
                num_real_tokens = (padding_mask == 1).sum().item()

                print(f"Real tokens: {num_real_tokens}")
                print(f"Padding tokens: {num_padding_tokens}")

                # Check if any sequences are truncated
                if first_batch['input_ids'].shape[1] >= trainer.args.max_length:
                    print("⚠️  Warning: Sequences may be truncated at max_length")
                else:
                    print(f"✓ Sequences fit within max_seq_length ({trainer.args.max_length})")

                # Final pre-training checklist
                print("\n" + "=" * 60)
                print("FINAL PRE-TRAINING CHECKLIST")
                print("=" * 60)

                print(f"✓ Batch loading successful: shape {first_batch['input_ids'].shape}")
                print(f"✓ Assistant-only loss active: {num_masked} masked, {num_unmasked} unmasked tokens")
                print(f"✓ Masking ratio: {num_masked/(num_masked+num_unmasked)*100:.1f}% masked")
                print(f"✓ Training for {self.epochs} epochs")
                print(f"✓ Batch size: {self.per_device_train_batch_size}")
                print(f"✓ Gradient accumulation: {self.gradient_accumulation_steps}")
                print(f"✓ Effective batch size: {self.per_device_train_batch_size * self.gradient_accumulation_steps}")
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

        if self.debug:
            self.debug_trainer(trainer)

        logger.info("🚀 Starting training...")
        trainer.train()
        logger.info("✅ Training completed.")

        if self.debug:
            # Check trainer state
            print("=== Training State ===")
            print(f"Best metric: {trainer.state.best_metric}")
            print(f"Best model checkpoint: {trainer.state.best_model_checkpoint}")
            print(f"Total steps: {trainer.state.global_step} / {trainer.state.max_steps}")
            print(f"Early stopped: {trainer.state.global_step < trainer.state.max_steps}")

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

from unsloth import is_bfloat16_supported
from src.core.cot_training.processing_lib.dataset_handler import DatasetHandler
from src.core.cot_training.processing_lib.model_handler import ModelHandler
from src.core.cot_training.loader_config import Loader
from src.core.cot_training.processing_lib.custom.weighted_cot_trainer import WeightedCoTTrainer

import logging
import os
import gc
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from pathlib import Path
from datetime import datetime
from transformers import PreTrainedTokenizer
from datasets import DatasetDict


logger = logging.getLogger(name=__name__)


class FineTuningHandler:
    def __init__(
        self,
        dataset_handler_class: type[DatasetHandler],
        model_loader_class: type[ModelHandler],
        # -- dataset --
        dataset_path: str,
        formatted_dataset_path: Path,
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
        training_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        weight_decay: float,
        eval_steps: float,
        warmup_steps: int = 10,
        logging_steps: int = 100,
        use_weighted_cot_trainer: bool = False,
    ):

        self.dataset_handler_class = dataset_handler_class
        self.dataset_path = dataset_path
        self.formatted_dataset_path = formatted_dataset_path
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
        self.num_train_epochs = training_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.wd = weight_decay
        self.use_weighted_trainer = use_weighted_cot_trainer

        self._dataset_dict = None
        self._base_tokenizer = None

    def setup_paths(self) -> None:
        provider, model_id = self.base_model_name.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")

        common_suffix: str = os.path.join(provider, model_id, date, time)
        trainer_dir: str = os.path.join("./trainer", common_suffix)

        self.lora_model_dir: str = os.path.join(trainer_dir, "lora_model")
        self.checkpoint_dir: str = os.path.join("./checkpoints", common_suffix)
        self.output_dir: str = os.path.join("./results", f"{model_id}_{date}_{time}")

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(trainer_dir, exist_ok=True)

    def _prepare_dataset_with_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Prepare dataset using the provided tokenizer."""

        if self._dataset_dict is not None:
            logger.info("♻️ Reusing cached dataset")
            return self._dataset_dict

        with Loader("📚 Preparing dataset with tokenizer...", "✅ Dataset prepared and cached for reuse", logger=logger):
            dataset_handler = self.dataset_handler_class(
                dataset_path=self.dataset_path,
                formatted_dataset_path=self.formatted_dataset_path,
                tokenizer=tokenizer,
                num_cpus=self.num_cpus,
            )
            self._dataset_dict = dataset_handler.run_pipeline()
        return self._dataset_dict

    def create_trainer_with_params(self) -> tuple[SFTTrainer, tuple]:
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
        )

        model_loader._load_base_model()
        model_loader.patch_model()  # LoRA patch
        model, tokenizer = model_loader.obtain_components()

        dataset_dict: DatasetDict = self._prepare_dataset_with_tokenizer(tokenizer=tokenizer)

        logger.info(f"📊 Model trainable parameters:")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()  # type: ignore

        sft_args = self._ft_args()


        if not self.use_weighted_trainer:
            trainer = SFTTrainer(
                model=self.model,  # type: ignore
                processing_class=tokenizer,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                args=sft_args,
            )
        else:
            trainer = WeightedCoTTrainer(
                model=model,
                tokenizer=tokenizer,
                args=sft_args,
                train_dataset=dataset_dict["train"],
                eval_dataset=dataset_dict["validation"],
                reasoning_weight=1.0,
                answer_weight=1.5,
                answer_marker="Final Answer:",  # Only this is task-specific!
            )


        return trainer, (model, tokenizer)

    def _ft_args(self) -> SFTConfig:
        """Configures the training parameters for SFTTrainer."""

        if self.training_epochs <= 0:
            logger.warning("No training duration has been specified. Fallback to 1")
            self.training_epochs = 1

        bfloat16_supported: bool = is_bfloat16_supported()

        return SFTConfig(
            assistant_only_loss=True,  # only on assistant generated tokens
            use_liger_kernel=True,
            output_dir=self.checkpoint_dir,
            run_name=f"{self.base_model_name.split('/')[-1]}-epochs-{self.training_epochs}",
            num_train_epochs=self.training_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.lr,
            weight_decay=self.wd,
            max_grad_norm=0.3,
            logging_steps=self.logging_steps,
            eval_strategy="steps",
            eval_steps=self.eval_steps,
            save_strategy="steps",  # "best"
            save_steps=self.eval_steps,
            fp16=not bfloat16_supported,
            bf16=bfloat16_supported,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            max_length=self.max_seq_length,
            dataset_text_field="text",
            packing=False,
            report_to="wandb",
            gradient_checkpointing=True,
            seed=3407,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

    def fine_tune(self):
        self.setup_paths()
        trainer, (model, tokenizer) = self.create_trainer_with_params()
        trainer.train()

        logger.info("✅ Training completed. ✅")

        # Save the LoRA adapters
        if model and tokenizer:
            model.save_pretrained(save_directory=self.lora_model_dir)  # type: ignore
            tokenizer.save_pretrained(save_directory=self.lora_model_dir)

        logger.info(f"✅ LoRA adapters saved successfully. ✅")

        # <----- CLEANUP BLOCK ---->
        logger.debug("Cleaning up training resources to free VRAM...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("✅ Cleanup complete. VRAM should now be released. ✅")

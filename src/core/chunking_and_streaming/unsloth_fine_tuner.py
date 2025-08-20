import os
import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from pathlib import Path

# official documentation : https://docs.unsloth.ai/
from unsloth import FastLanguageModel, is_bfloat16_supported

import torch
import wandb
import bitsandbytes as bnb
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ...common.loading_config import Loader


logger = logging.getLogger(name=__name__)

# supported 4bit pre-quantized models for 4x faster downloading + no OOMs.
# More models at https://huggingface.co/unsloth
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",
    "unsloth/Mistral-Small-Instruct-2409",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
]


@dataclass
class UnslothFineTunePipeline:
    """
    A class to handle the fine-tuning of a base model from Hugging Face.

    This class manages loading a base model, applying PEFT for LoRA fine-tuning,
    configuring the trainer, and running the training process.

    Attributes:
        hf_train_data (Dataset): The training dataset in Hugging Face's `Dataset` format.
        hf_eval_data (Dataset): The evaluation dataset for use during training.
        base_model_str (str): The model identifier from the Hugging Face Hub (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit").
        max_seq_length (int): The maximum sequence length for the model's tokenizer.
        training_epochs (int): The number of epochs to train for. Set to -1 to use steps instead.
        training_steps (int): The number of training steps. Overrides `training_epochs` if not -1.
        PROMPT_TEMPLATE_TRAINING (str): The f-string template used to format examples for training.
        output_dir (str): The automatically generated path where the computed metrics on test set will be saved.
        lora_model_dir (str): The automatically generated path where the trained LoRA adapters will be saved.
    """

    hf_train_data: Dataset
    hf_eval_data: Dataset
    base_model_str: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" # base model to fine-tune
    max_seq_length: int = 2048  # Max sequence length for the model's tokenizer.
    # how long to train for precedence to epochs
    training_epochs: int = -1
    training_steps: int = -1
    use_double_quantization: bool = False
 
    lora_model_dir: str = field(init=False)
    base_model: FastLanguageModel|None = field(init=False, default=None)
    base_tokenizer: PreTrainedTokenizer|None= field(init=False, default=None)

    def __post_init__(self) -> None:
        provider, model_id = self.base_model_str.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")

        common_suffix: Path = Path(provider) / model_id / date / time

        trainer_dir: Path = Path("./trainer") / common_suffix
        self.lora_model_dir: str = str(trainer_dir / "lora_model")
        self.checkpoint_dir: str = str(Path("./checkpoints") / common_suffix)
        self.output_dir: str = str(Path("./results") / f"{model_id}_{date}_{time}")

        # Ensure output directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(trainer_dir, exist_ok=True)

        self.bnb_config_arguments: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": (torch.bfloat16 if is_bfloat16_supported() else None),
            "bnb_4bit_use_double_quant": self.use_double_quantization,
        }

    def wb_init(self) -> None:
        """Initializes Weights & Biases for experiment tracking."""

        try:
            wandb.init(
                project=f"Unsloth fine-tune {os.path.split(self.base_model_str)[-1]} for vulnerability detection.",
                job_type="training",
                anonymous="allow"
            )
            logger.info("Weights & Biases initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")

    def unsloth_load_base_model(self):
        """Dynamically loads the base model by first attempting to load it all onto
        the GPU, and then falling back to automatic CPU offloading if VRAM is insufficient.
        """

        logger.info(f"Initializing base model and tokenizer for '{self.base_model_str}'...")

        load_kwargs: dict[str,Any] = {
            "model_name": self.base_model_str,
            "max_seq_length": self.max_seq_length,
            "device_map": "auto",  # let Unsloth/Accelerate handle device placement
            "attn_implementation": "flash_attention_2",
        }

        if self.base_model_str in fourbit_models:
            load_kwargs["load_in_4bit"] = True
        else:
            logger.info("Model is not pre-quantized. Applying custom BitsAndBytesConfig...")
            load_kwargs["quantization_config"] = BitsAndBytesConfig(**self.bnb_config_arguments)

        try:
            self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
            logger.info("✅ Model and tokenizer loaded successfully.")
        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("⚠️ OOM Error: Model does not fit entirely on GPU.")
            logger.critical(f"Please try a smaller model or use a GPU with more VRAM.")
            torch.cuda.empty_cache()  # Clear cache after OOM error
            raise oom
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise e

        # --- final Verification ---
        if self.base_model:
            is_quantized = (
                hasattr(self.base_model, "quantization_config")
                and self.base_model.quantization_config.load_in_4bit # type: ignore
                or any(isinstance(m, bnb.nn.Linear4bit) for m in self.base_model.modules()) # type: ignore
            )

            if is_quantized: logger.info("✅ Verification successful: Model is loaded in 4-bit.")
            else: logger.warning("⚠️ Verification warning: Model does not appear to be loaded in 4-bit.")

    def unsloth_patch_model(self) -> None:
        """Applies PEFT to the base model for LoRA fine-tuning."""

        if not self.base_model:
            self.unsloth_load_base_model() # load base model

        def _find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit  # Assuming 4-bit quantization
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split(".")
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if "lm_head" in lora_module_names: # lm_head is usually not LoRA-adapted
                lora_module_names.remove("lm_head")

            return list(lora_module_names)

        with Loader("Patching model"):
            self.base_model = FastLanguageModel.get_peft_model(
                model=self.base_model,
                max_seq_length=self.max_seq_length,
                r=16,  # r>0, suggested vals: {8, 16, 32, 64, 128}
                lora_alpha=16,  # Alpha parameter for LoRA scaling, suggestion r=alpha or alpha=2*r
                target_modules=_find_all_linear_names(model=self.base_model),
                modules_to_save=None,
                lora_dropout=0,  # Dropout probability for LoRA layers. Supports any, but = 0 is optimized
                bias="none",  # Bias type for LoRA layers. Supports any, but = "none" is optimized
                # "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing="unsloth",  # type: ignore
                random_state=3407,
            )

    def _unsloth_set_fine_tune(self) -> SFTConfig:
        """Configures the training parameters for SFTTrainer."""

        if self.training_epochs != -1 and self.training_steps != -1:
            logger.warning("Both training_epochs and training_steps have been specified. Doing so, steps are taken as reference")

        if self.training_epochs == self.training_steps == -1:
            logger.warning("No training duration has been specified. Fallback to 1 training epoch")
            self.training_epochs = 1

        bfloat16_supported: bool = is_bfloat16_supported()

        return SFTConfig(
            use_liger_kernel=True,
            max_length=self.max_seq_length,
            # with full_finetuning=True
            num_train_epochs=self.training_epochs,
            max_steps=self.training_steps,
            per_device_train_batch_size=2,  # do not increase
            gradient_accumulation_steps=4,  # has the same effect of increasing batch size for smoother curves
            warmup_steps=10,
            learning_rate=2e-4,  # suggested values: 2e-4 (QLoRA paper), 1e-4, 5e-5, 2e-5
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            eval_strategy="steps",  # Evaluate every eval_steps
            eval_steps=0.2,  # Evaluate every 20% of the epoch
            fp16=not bfloat16_supported,
            bf16=bfloat16_supported,
            logging_steps=20,
            output_dir=self.checkpoint_dir,
            run_name=f"{os.path.basename(self.base_model_str)}-sft",  # Unique run name for WandB
            # optim="adamw_8bit",
            optim="paged_adamw_8bit",  # paged optimizer for memory efficiency (8bit for better compatibility with 4bit quant)
            lr_scheduler_type="cosine",  # cosine learning rate scheduler
            seed=3407,  # for reproducibility
            packing=False,  # don't merge small samples into one
            dataset_text_field="text",  # header where input prompt is
            report_to="wandb",
        )

    def unsloth_fine_tune(self) -> None:
        """Starts the fine-tuning process."""

        if not self.base_model: self.unsloth_patch_model()

        trainer = SFTTrainer(
            model=self.base_model,  # type: ignore # `FastLanguageModel` acts like a `PreTrainedModel` at runtime
            processing_class=self.base_tokenizer,
            train_dataset=self.hf_train_data,
            eval_dataset=self.hf_eval_data,
            args=self._unsloth_set_fine_tune(),
        )
        trainer.train()
        logger.info("✅ Training completed. ✅")

        # Save the LoRA adapters
        if self.base_model and self.base_tokenizer:
            self.base_model.save_pretrained(save_directory=self.lora_model_dir)  # type: ignore
            self.base_tokenizer.save_pretrained(save_directory=self.lora_model_dir)

        logger.info(f"✅ LoRA adapters saved successfully. ✅")

        # <----- CLEANUP BLOCK ---->
        logger.debug("Cleaning up training resources to free VRAM...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("✅ Cleanup complete. VRAM should now be released. ✅")

    def execute(self):
        self.unsloth_load_base_model()
        self.unsloth_patch_model()
        self.unsloth_fine_tune()

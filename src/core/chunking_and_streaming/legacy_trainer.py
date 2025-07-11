# src/core/chunking_and_streaming/legacy_trainer.py

import os
import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

import bitsandbytes as bnb
import torch
import wandb
from datasets import Dataset

import bitsandbytes as bnb
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from .stdout import logger


@dataclass
class TrainingPipeline:
    """Handles the fine-tuning of a standard Hugging Face model with LoRA."""

    hf_train_data: Dataset
    hf_eval_data: Dataset
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    max_seq_length: int = 2048
    # how long to train for precedence to epochs
    training_epochs: int = -1
    training_steps: int = -1
    use_double_quantization: bool = False

    # <---- Attributes for managing paths and models ---->
    lora_model_dir: str = field(init=False)
    base_model: Optional[AutoModelForCausalLM] = field(init=False, default=None)
    base_tokenizer: Optional[PreTrainedTokenizer] = field(init=False, default=None)

    def __post_init__(self):
        """Initializes paths for saving the trained model."""
        provider, model_id = self.base_model_name.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")
        common_suffix: str = os.path.join(provider, model_id, date, time)

        trainer_dir: str = os.path.join("./trainer", common_suffix)
        self.lora_model_dir: str = os.path.join(trainer_dir, "lora_model")
        self.checkpoint_dir: str = os.path.join("./checkpoints/", common_suffix)
        self.output_dir: str = os.path.join("./results/", f"{model_id}_{date}_{time}")

        # Ensure output directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(trainer_dir, exist_ok=True)

        self.bnb_config_arguments: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else None
            ),
            "bnb_4bit_use_double_quant": self.use_double_quantization,
        }

    def wb_init(self) -> None:
        """Initializes Weights & Biases for experiment tracking."""

        try:
            # wandb.login()
            wandb.init(
                project=f"Fine-tune {os.path.split(self.base_model_name)[-1]} for vulnerability detection.",
                job_type="training",
                anonymous="allow",
            )
            print("Weights & Biases initialized.")
        except Exception as e:
            print(f"Failed to initialize Weights & Biases: {e}")

    def load_model_and_tokenizer(self) -> None:
        """Loads the base model and tokenizer with 4-bit quantization."""

        logger.info(f"Loading base model: {self.base_model_name}")

        # <---- Step 1: The "Pre-check" - Attempt to load everything to GPU ---->
        try:
            print("Attempting to load model directly onto GPU...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                max_seq_lenth=self.max_seq_length,
                quantization_config=BitsAndBytesConfig(**self.bnb_config_arguments),
                device_map="cuda:0",
                attn_implementation="flash_attention_2",
            )

        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM Error: Model does not fit entirely on GPU.")
            print("Retrying with automatic CPU offloading enabled...")
            torch.cuda.empty_cache()  # Clear cache after OOM error

            # --- Step 2: The "Fallback" - Load with dynamic offloading ---
            try:
                # offloading
                self.bnb_config_arguments["llm_int8_enable_fp32_cpu_offload"] = True
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    max_seq_lenth=self.max_seq_length,
                    quantization_config=BitsAndBytesConfig(**self.bnb_config_arguments),
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )

                print("✅ Model loaded successfully with dynamic CPU offloading.")

            except Exception as e:
                print(f"❌ Failed to load model even with offloading enabled: {e}")
                raise e

        finally:
            # if no error raised, load tokenizer as well
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.base_tokenizer:
                if self.base_tokenizer.pad_token is None:
                    self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
                self.base_tokenizer.padding_side = "right"

            print("✅ Model fits entirely on GPU.")

    def run_training(self) -> None:
        """Configures and runs the SFTTrainer fine-tuning process."""
        if not self.base_model or not self.base_tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before training.")

        if self.training_epochs != -1 and self.training_steps != -1:
            logger.warning(
                msg="Both training_epochs and training_steps have been specified. Doing so, steps are taken as reference"
            )

        if self.training_epochs == self.training_steps == -1:
            logger.warning(
                msg="No training duration has been specified. Fallback to 1 training epoch"
            )
            self.training_epochs = 1

        def find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit
            lora_module_names = {
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, cls)
            }
            lora_module_names.discard("lm_head")

            return list(lora_module_names)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # r>0, suggested vals: {8, 16, 32, 64, 128}
            lora_alpha=16,  # Alpha parameter for LoRA scaling, suggestion r=alpha or alpha=2*r
            inference_mode=False,  # Set to True for inference after training
            target_modules=find_all_linear_names(self.base_model),
            modules_to_save=None,
            lora_dropout=0,
            bias="none",
        )

        bfloat16_supported: bool = torch.cuda.is_bf16_supported()
        training_args = SFTConfig(
            use_liger_kernel=True,
            max_length=self.max_seq_length,
            num_train_epochs=self.training_epochs,
            max_steps=self.training_steps,
            per_device_train_batch_size=2,  # do not increase
            gradient_accumulation_steps=4,  # has the same effect of increasing batch size for smoother curves
            gradient_checkpointing=True,  # Use gradient checkpointing to save memory
            warmup_steps=10,
            learning_rate=2e-4,  # suggested values: 2e-4 (QLoRA paper), 1e-4, 5e-5, 2e-5
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            eval_strategy="steps",  # Evaluate every eval_steps
            eval_steps=0.2,  # Evaluate every 20% of the epoch
            fp16=not bfloat16_supported,
            bf16=bfloat16_supported,
            logging_steps=20,
            output_dir=self.checkpoint_dir,
            run_name=f"{os.path.basename(self.base_model_name)}-sft",  # Unique run name for WandB
            optim="paged_adamw_8bit",  # paged optimizer for memory efficiency (8bit for better compatibility with 4bit quant)
            lr_scheduler_type="cosine",  # cosine learning rate scheduler
            seed=3407,  # for reproducibility
            packing=False,  # don't merge small samples into one
            dataset_text_field="text",
            report_to="wandb",
        )

        trainer = SFTTrainer(
            model=self.base_model,  # type: ignore
            processing_class=self.base_tokenizer,
            train_dataset=self.hf_train_data,
            eval_dataset=self.hf_eval_data,
            peft_config=peft_config,
            args=training_args,
        )

        try:
            logger.info("Starting fine-tuning...")
            trainer.train()
            logger.info("Fine-tuning complete.")
            logger.info("Closing WandB portal...")
            wandb.finish()
            self.base_model.config.use_cache = True  # type: ignore
            logger.info(f"Saving LoRA adapters to: {self.lora_model_dir}")
            trainer.save_model(output_dir=self.lora_model_dir)
            self.base_tokenizer.save_pretrained(save_directory=self.lora_model_dir)
            # trainer.save_model(os.path.join(self.trainer_dir, "model"))
            logger.info(f"Model and tokenizer saved to {self.lora_model_dir}")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

        # <----- CLEANUP BLOCK ---->
        print("Cleaning up training resources to free VRAM...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete. VRAM should now be released.")

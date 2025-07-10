# official documentation : https://docs.unsloth.ai/
# Standard library imports
import os
import gc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# Third-party library imports
import pandas as pd

# FIX: Unsloth must be imported before transformers and peft
from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: ignore
import torch
from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import bitsandbytes as bnb
from transformers.utils.quantization_config import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import wandb
from .stdout import logger

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
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


# =================
# TRAINING PIPELINE
# =================
@dataclass
class UnslothModel:
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
    # base model to fine-tune
    base_model_str: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    # Max sequence length for the model's tokenizer.
    max_seq_length: int = 2048
    max_tokens_per_answer: int = 5
    # how long to train for precedence to epochs
    training_epochs: int = -1
    training_steps: int = -1

    # The prompt structure for training (includes the answer)
    PROMPT_TEMPLATE_TRAINING: str = (
        "You are an AI system that analyzes C code for vulnerabilities.\n\n"
        "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
        "KEY:Code is chunked; reassemble by function signature to obtain full original source code.\n"
        "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
        "Function signature:\n{signature}\n\n"
        "Code Fragment:\n{subchunk}\n\n"
        "Answer 'YES' if vulnerable, 'NO' otherwise.\n"
        "Correct answer:\n{ground_truth}"
    )

    # <---- Attributes for managing paths and models ---->
    lora_model_dir: str = field(init=False)
    base_model: Optional[FastLanguageModel] = field(init=False, default=None)
    base_tokenizer: Optional[PreTrainedTokenizer] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Determine paths for checkpoints and trainer output
        provider, model_id = self.base_model_str.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")
        common_suffix: str = os.path.join(provider, model_id, date, time)

        # saving model checkpoint
        self.checkpoint_dir: str = os.path.join("./checkpoints/", common_suffix)
        # target directory for metrics
        self.output_dir: str = os.path.join("./results/", f"{model_id}_{date}_{time}")
        trainer_dir: str = os.path.join("./trainer", common_suffix)
        self.lora_model_dir: str = os.path.join(trainer_dir, "lora_model")

        # Ensure output directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(trainer_dir, exist_ok=True)

        self.bnb_config_arguments: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": (
                torch.bfloat16 if is_bfloat16_supported() else None
            ),
            # "bnb_4bit_use_double_quant": True,
        }

    def wb_init(self) -> None:
        """Initializes Weights & Biases for experiment tracking."""

        try:
            # wandb.login()
            wandb.init(
                project=f"Fine-tune {os.path.split(self.base_model_str)[-1]} for vulnerability detection.",
                job_type="training",
                anonymous="allow",
            )
            print("Weights & Biases initialized.")
        except Exception as e:
            print(f"Failed to initialize Weights & Biases: {e}")

    # <---- MODEL LOADING, PATCHING, AND TRAINING METHODS ---->
    def unsloth_load_base_model(self) -> None:
        """
        Dynamically loads the base model by first attempting to load it all onto
        the GPU, and then falling back to automatic CPU offloading if VRAM is insufficient.
        """

        print(f"Initializing base model and tokenizer for '{self.base_model_str}'...")

        # <---- Step 1: The "Pre-check" - Attempt to load everything to GPU ---->
        try:
            print("Attempting to load model directly onto GPU...")
            load_kwargs = {
                "model_name": self.base_model_str,
                "max_seq_length": self.max_seq_length,
                "device_map": "cuda:0",  # Force to GPU
                "attn_implementation": "flash_attention_2",
            }
            if self.base_model_str in fourbit_models:
                load_kwargs["load_in_4bit"] = True
            else:
                # For any other model, apply the custom quantization config on the fly
                print(
                    "Model is not pre-quantized. Applying custom BitsAndBytesConfig..."
                )
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    **self.bnb_config_arguments
                )

            self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(
                **load_kwargs
            )
            print("✅ Model fits entirely on GPU.")
            return

        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM Error: Model does not fit entirely on GPU.")
            print("Retrying with automatic CPU offloading enabled...")
            torch.cuda.empty_cache()  # Clear cache after OOM error

            # --- Step 2: The "Fallback" - Load with dynamic offloading ---
            try:
                # For offloading to work, we MUST provide the quantization config with the
                self.bnb_config_arguments["llm_int8_enable_fp32_cpu_offload"] = True

                self.base_model, self.base_tokenizer = (
                    FastLanguageModel.from_pretrained(
                        model_name=self.base_model_str,
                        max_seq_length=self.max_seq_length,
                        quantization_config=BitsAndBytesConfig(
                            **self.bnb_config_arguments
                        ),
                        device_map="auto",  # The key is 'auto' + the quantization_config
                        attn_implementation="flash_attention_2",
                    )
                )
                print("✅ Model loaded successfully with dynamic CPU offloading.")

            except Exception as e:
                print(f"❌ Failed to load model even with offloading enabled: {e}")
                raise e

        # --- Step 3: Final Verification ---
        # This check runs after a successful load from either the try or except block.
        if self.base_model:
            is_quantized = hasattr(self.base_model, "quantization_config") and self.base_model.quantization_config.load_in_4bit or any(isinstance(m, bnb.nn.Linear4bit) for m in self.base_model.modules())  # type: ignore
            print(
                "✅ Verification successful: Model is loaded in 4-bit."
                if is_quantized
                else "⚠️ Verification warning: Model does not appear to be loaded in 4-bit."
            )

    def unsloth_patch_model(self) -> None:
        """Applies PEFT to the base model for LoRA fine-tuning."""

        if not self.base_model:
            raise ValueError(
                "Base model is not loaded. Please run `unsloth_load_base_model` first."
            )

        def find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit  # Assuming 4-bit quantization
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split(".")
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if "lm_head" in lora_module_names:
                # lm_head is usually not LoRA-adapted
                lora_module_names.remove("lm_head")

            return list(lora_module_names)

        print("Applying PEFT to the model...")
        self.base_model = FastLanguageModel.get_peft_model(
            model=self.base_model,
            max_seq_length=self.max_seq_length,
            r=16,  # r>0, suggested vals: {8, 16, 32, 64, 128}
            lora_alpha=16,  # Alpha parameter for LoRA scaling, suggestion r=alpha or alpha=2*r
            target_modules=find_all_linear_names(model=self.base_model),
            modules_to_save=None,
            lora_dropout=0,  # Dropout probability for LoRA layers. Supports any, but = 0 is optimized
            bias="none",  # Bias type for LoRA layers. Supports any, but = "none" is optimized
            # "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # type: ignore
            random_state=3407,
        )
        print("PEFT applied successfully.")

    def _unsloth_config_training(self) -> SFTConfig:
        """Configures the training parameters for SFTTrainer."""

        if self.training_epochs != -1 and self.training_steps != -1:
            logger.warning(
                msg="Both training_epochs and training_steps have been specified. Doing so, steps are taken as reference"
            )

        if self.training_epochs == self.training_steps == -1:
            logger.warning(
                msg="No training duration has been specified. Fallback to 1 training epoch"
            )
            self.training_epochs = 1

        bfloat16_supported: bool = is_bfloat16_supported()

        return SFTConfig(
            use_liger_kernel=True,
            max_seq_length=self.max_seq_length,
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

    def unsloth_start_training(self) -> None:
        """Starts the fine-tuning process."""

        if not self.base_model or not self.base_tokenizer:
            raise ValueError(
                "Base model or tokenizer not loaded. Cannot start training."
            )

        print("Starting model training...")
        trainer = SFTTrainer(
            # `FastLanguageModel` acts like a `PreTrainedModel` at runtime,
            # but the type checker can't see this. We ignore the error.
            model=self.base_model,  # type: ignore
            processing_class=self.base_tokenizer,
            train_dataset=self.hf_train_data,
            eval_dataset=self.hf_eval_data,
            args=self._unsloth_config_training(),
        )
        trainer.train()
        print("Training completed.")

        # Save the LoRA adapters
        print(f"Saving LoRA adapters to {self.lora_model_dir}")
        if self.base_model and self.base_tokenizer:
            # `save_pretrained` is dynamically available on the model object.
            self.base_model.save_pretrained(save_directory=self.lora_model_dir)  # type: ignore
            self.base_tokenizer.save_pretrained(save_directory=self.lora_model_dir)

        # <----- CLEANUP BLOCK ---->
        print("Cleaning up training resources to free VRAM...")
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        print("Cleanup complete. VRAM should now be released.")


if __name__ == "__main__":
    # 1. Create Dummy Data reflecting the new prompt structure
    training_data = {
        "signature": [
            "strcpy_vulnerable(char *dst, const char *src)",
            "safe_copy(char *dst, const char *src, size_t size)",
            "process_request(http_request *req)",
            "validate_input(const char* input)",
        ],
        "subchunk": [
            "{\n  strcpy(dst, src);\n}",
            "{\n  strncpy(dst, src, size - 1);\n  dst[size - 1] = '\\0';\n}",
            '{\n  char buffer[128];\n  sprintf(buffer, "User agent: %s", req->user_agent);\n}',
            "{\n  if (strlen(input) < 10) return true;\n  return false;\n}",
        ],
        "ground_truth": ["YES", "NO", "YES", "NO"],
    }
    dummy_train_df = pd.DataFrame(training_data)

    # Format the training data into the 'text' column required by SFTTrainer
    dummy_train_df["text"] = dummy_train_df.apply(
        lambda row: UnslothModel.PROMPT_TEMPLATE_TRAINING.format(
            signature=row["signature"],
            subchunk=row["subchunk"],
            ground_truth=row["ground_truth"],
        ),
        axis=1,
    )

    # Test data uses the same structure but 'ground_truth' is named 'label' for evaluation
    test_data = {
        "signature": [
            "get_user_data(const char *user)",
            "init_secure_buffer(size_t n)",
        ],
        "subchunk": [
            "{\n  char query[256];\n  sprintf(query, \"SELECT * FROM users WHERE name = '%s'\", user);\n}",
            "{\n  char *buf = malloc(n);\n  if (!buf) return NULL;\n  memset(buf, 0, n);\n  return buf;\n}",
        ],
        "label": ["YES", "NO"],
    }
    dummy_test_df = pd.DataFrame(test_data)

    hf_train_data = Dataset.from_pandas(pd.DataFrame(dummy_train_df["text"]))
    hf_eval_data = hf_train_data  # Use same for eval in this example

    # 2. Instantiate and Run the UnslothModel pipeline
    model_pipeline = UnslothModel(
        hf_train_data=hf_train_data,
        hf_eval_data=hf_eval_data,
        # df_test_data=dummy_test_df,
    )

    # Execute the training workflow
    model_pipeline.unsloth_load_base_model()
    model_pipeline.unsloth_patch_model()
    model_pipeline.unsloth_start_training()

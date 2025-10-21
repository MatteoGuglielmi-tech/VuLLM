from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import logging
import torch
import bitsandbytes as bnb

from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Any, cast

from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from peft import LoftQConfig

from src.core.cot_training.loader_config import Loader

logger = logging.getLogger(name=__name__)


@dataclass
class ModelHandler:

    base_model_name: str = "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit"
    chat_template: str = field(init=True, default="qwen-2.5")

    lora_model_dir: str = field(init=False)
    base_model: FastLanguageModel | None = field(init=False, default=None)
    patched_model: FastLanguageModel | None = field(init=False, default=None)
    tokenizer: PreTrainedTokenizer | None = field(init=False, default=None)

    # --- LoRA Hyperparameters ---
    # LoRA update is scaled by the formula alpha / r
    lora_r: int = 16 # controls the number of trainable parameters
    lora_alpha: int = 16 # controls the magnitude of that new knowledge
    lora_dropout: float = 0

    # --- PEFT Configuration ---
    max_seq_length: int = 2048
    use_rslora: bool = True # smarter scaling for the LoRA weights (no lora_alpha needed)
    use_loftq: bool = False # smarter initialization for the LoRA weights (this requires loading the entire module in memory at first)

    def __post_init__(self) -> None:
        # immediately check on `lora_r` value
        if self.lora_r <= 0:
            logger.warning("⚠️ You've specified an invalid value for LoRA rank. Falling back to default.")
            self.lora_r = 16
        if (self.lora_r != self.lora_alpha) and (self.lora_alpha / self.lora_r != 2):
            logger.warning("⚠️ It's suggested to set the value for `lora_alpha` to: `lora_r` or `2*lora_r`")

    def _load_base_model(self):
        logger.info(f"Initializing base model and tokenizer for '{self.base_model_name}'...")

        try:
            self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                load_in_4bit=not self.use_loftq,
                device_map="auto",  # let accelerate handle device placement
                attn_implementation="flash_attention_2",
            )

            try:
                self.tokenizer = get_chat_template(
                    self.tokenizer,
                    chat_template=self.chat_template
                )
            except ValueError:
                raise ValueError(f"available chat templates are: {print(list(CHAT_TEMPLATES.keys()))}")

            if "llama" in self.chat_template:
                if self.tokenizer is not None: # redundant but this is to silence the linter
                    self.tokenizer.padding_side = "left"
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Model and tokenizer loaded successfully.")
        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("⚠️ OOM Error: Model does not fit entirely on GPU.")
            torch.cuda.empty_cache()
            raise oom
        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise e

        if getattr(self.base_model, "is_loaded_in_4bit", False):
            logger.info("✅ Verification successful: Model is loaded in 4-bit.")
        else:
            logger.warning(
                "⚠️ Verification warning: Model does not appear to be loaded in 4-bit."
            )

    def patch_model(self) -> None:
        """Applies PEFT to the base model for LoRA fine-tuning."""

        if self.base_model is None:
            self._load_base_model()

        def _find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split(".")
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if "lm_head" in lora_module_names:  # lm_head is usually not LoRA-adapted
                lora_module_names.remove("lm_head")

            return list(lora_module_names)

        dynamic_args: dict[str, Any] = (
            {"loftq_config": LoftQConfig(), "init_lora_weights": "loftq"}
            if self.use_loftq
            else {  # default
                "loftq_config": {},
                "init_lora_weights": True,
                "target_modules": _find_all_linear_names(model=self.base_model),
            }
        )

        with Loader(" 🩹 Patching model ..."):
            self.patched_model = FastLanguageModel.get_peft_model(
                model=self.base_model,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                max_seq_length=self.max_seq_length,
                use_rslora=self.use_rslora,
                **dynamic_args,
            )

    def obtain_components(self) -> tuple[FastLanguageModel, PreTrainedTokenizer]:
        return cast(FastLanguageModel, self.patched_model), cast(PreTrainedTokenizer, self.tokenizer)

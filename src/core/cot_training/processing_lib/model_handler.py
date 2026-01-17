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

from ..utilities import rich_table, rich_status

logger = logging.getLogger(name=__name__)


@dataclass
class ModelHandler:

    base_model_name: str
    chat_template: str|None = None

    lora_model_dir: str = field(init=False)
    base_model: FastLanguageModel|None = field(init=False, default=None, repr=False)
    patched_model: FastLanguageModel|None = field(init=False, default=None, repr=False)
    tokenizer: PreTrainedTokenizer|None = field(init=False, default=None, repr=False)

    # --- LoRA Hyperparameters ---
    # LoRA update is scaled by the formula alpha / r
    lora_r: int = 16 # controls the number of trainable parameters
    lora_alpha: int = 16 # controls the magnitude of that new knowledge
    lora_dropout: float = 0

    # --- PEFT Configuration ---
    max_seq_length: int = 2048
    use_rslora: bool = True # smarter scaling for the LoRA weights (no lora_alpha needed)
    use_loftq: bool = False # smarter initialization for the LoRA weights (this requires loading the entire module in memory at first)
    use_deepspeed: bool = False

    def __post_init__(self) -> None:
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate constructor inputs."""

        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if self.lora_r <= 0:
            logger.warning("⚠️ You've specified an invalid value for LoRA rank. Falling back to default (16).")
            self.lora_r = 16
        if self.lora_alpha <= 0:
            logger.warning(f"⚠️ You've specified an invalid value for LoRA alpha. Falling back to default ({2*self.lora_r}).")
            self.lora_alpha = 2*self.lora_r
        if self.lora_alpha < 0:
            logger.warning(f"⚠️ You've specified an invalid value for LoRA dropout. Falling back to default (0.0).")
            self.lora_dropout = 0.

    def _set_padding_strategy(self):
        """Set padding side based on model architecture.

        - Decoder-only (GPT, Llama): Left padding (for batch generation)
        - Encoder-decoder (T5, BART): Right padding
        """

        decoder_only_models: set[str] = { "llama", "mistral", "qwen", "opt", "phi", "gemma"} # deepseek uses "llama"
        model_type = getattr(self.base_model.config, "model_type", "").lower()  # type: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        if any(arch in model_type for arch in decoder_only_models):
            self.tokenizer.padding_side = "left"  # type: ignore[reportOptionalMemberAccess]
            logger.info(f"📍 Set padding_side='left' for decoder-only model ({model_type})")
        else:
            self.tokenizer.padding_side = "right"  # type: ignore[reportOptionalMemberAccess]
            logger.info(f"📍 Set padding_side='right' for encoder-decoder model ({model_type})")

        # ensure pad token exists
        if not (self.tokenizer.pad_token or self.tokenizer.pad_token_id):  # type: ignore[reportOptionalMemberAccess]
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[reportOptionalMemberAccess]
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # type: ignore[reportOptionalMemberAccess]

            logger.info(f"🔧 Set pad_token to eos_token: {self.tokenizer.eos_token}")  # type: ignore[reportOptionalMemberAccess]

    def _configure_tokenizer(self):
        """Configure tokenizer settings (chat template, padding, special tokens)."""
        model_name_lower = self.base_model_name.lower()

        # Check if DeepSeek model
        if "deepseek" in model_name_lower:
            logger.info("🔍 Detected DeepSeek model - applying custom chat template")

            # Override DeepSeek's default template to remove automatic system prompt
            self.tokenizer.chat_template = (  # type: ignore[reportOptionalMemberAccess]
                "{{ bos_token }}"
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ message['content'] + '\\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '### Instruction:\\n' + message['content'] + '\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '### Response:\\n' + message['content'] + '\\n<|EOT|>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '### Response:\\n' }}"
                "{% endif %}"
            )

            logger.info(
                "✅ Applied custom DeepSeek chat template (removed default system prompt)"
            )

        elif self.chat_template is not None:
            # Use Unsloth's built-in templates for non-DeepSeek models
            logger.info(
                f"🎨 Applying chat template for non-DeepSeek models: {self.chat_template}"
            )
            try:
                self.tokenizer = get_chat_template(
                    self.tokenizer, chat_template=self.chat_template
                )
            except ValueError as e:
                logger.error(f"Invalid chat template: {self.chat_template}")
                logger.error(f"Available templates: {list(CHAT_TEMPLATES.keys())}")
                raise ValueError(
                    f"Chat template '{self.chat_template}' not found"
                ) from e

        self._set_padding_strategy()

    def _load_base_model(self):
        logger.info(f"Initializing base model and tokenizer for '{self.base_model_name}'...")

        try:
            self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                load_in_4bit=not self.use_loftq,
                device_map="auto" if not self.use_deepspeed else None,  # type: ignore[reportArgumentType]
                attn_implementation="flash_attention_2",
            )

            if self.base_model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer failed to load")

            logger.info("✅ Base model and tokenizer loaded successfully")
            self._configure_tokenizer()

        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("💥 OUT OF MEMORY!")
            logger.critical(f"Model requires more VRAM than available.")
            logger.critical(f"Solutions:")
            logger.critical(f"  1. Use a smaller model")
            logger.critical(f"  2. Reduce max_seq_length (current: {self.max_seq_length})")
            logger.critical(f"  3. Use 8-bit quantization instead of 4-bit")
            torch.cuda.empty_cache()
            raise oom

        except Exception as e:
            logger.exception(f"❌ Failed to load model")
            raise e

        if getattr(self.base_model, "is_loaded_in_4bit", False):
                logger.info("✅ Verification successful: Model is loaded in 4-bit.")
        else:
            logger.warning(
                "⚠️ Verification warning: Model does not appear to be loaded in 4-bit."
                "This is normal since LoftQ is in use" if self.use_loftq else "Anomaly"
            )

    def patch_model(self) -> None:
        """Applies PEFT to the base model for LoRA fine-tuning."""

        if self.base_model is None:
            self._load_base_model()

        def _find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.modules.Linear4bit
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

        with rich_status(description=" 🩹 Patching model ...", spinner="arc"):
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

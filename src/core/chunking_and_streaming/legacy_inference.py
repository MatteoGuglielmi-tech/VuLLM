import os
import json
from dataclasses import dataclass, field
from typing import Optional, Any

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.generation.logits_process import LogitsProcessor

from .shared.typedef import ChatMsg
from .logits_processor import EnforceSingleTokenGeneration

import logging
from .shared.stdout import MY_LOGGER_NAME
logger = logging.getLogger(MY_LOGGER_NAME)

@dataclass
class InferencePipeline:
    """Handles loading a fine-tuned non-Unsloth model and running inference."""

    lora_model_dir: str
    max_seq_length: int = 100
    max_tokens_per_answer: int = 5
    use_double_quantization: bool = False

    # Attributes for the loaded model and tokenizer
    model: Optional[AutoPeftModelForCausalLM] = field(init=False, default=None)
    tokenizer: Optional[PreTrainedTokenizer] = field(init=False, default=None)
    output_dir: str = field(init=False)
    logits_processor: list[LogitsProcessor] = field(init=False)

    def __post_init__(self):
        """Initializes paths and loads the model."""
        results_path = self.lora_model_dir.replace("/lora_model", "").replace(
            "trainer", "results", 1
        )
        self.output_dir = results_path
        os.makedirs(self.output_dir, exist_ok=True)
        self._load_and_merge_model()
        # After the model is loaded, prepare the logits processor
        if self.tokenizer:
            # Important: Get the token IDs for the exact words you want
            yes_token_id: int = self.tokenizer.encode(
                text="YES", add_special_tokens=False
            )[0]
            no_token_id: int = self.tokenizer.encode(
                text="NO", add_special_tokens=False
            )[0]

            self.logits_processor = [
                EnforceSingleTokenGeneration(
                    allowed_token_ids=[yes_token_id, no_token_id]
                )
            ]

        self.bnb_config_arguments: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else None
            ),
            "bnb_4bit_use_double_quant": self.use_double_quantization,
        }

    def _load_and_merge_model(self):
        """Loads the fine-tuned PEFT model and merges it for inference."""
        logger.info(f"Loading and merging model from: {self.lora_model_dir}")

        # Step 1: Read the adapter's config to find the original base model name
        adapter_config_path = os.path.join(self.lora_model_dir, "adapter_config.json")
        if not os.path.exists(path=adapter_config_path):
            raise FileNotFoundError(
                f"Adapter config not found at {adapter_config_path}"
            )

        # Read the adapter's config to find the original base model name
        with open(file=adapter_config_path, mode="r") as f:
            adapter_config = json.load(fp=f)
        base_model_name: str = adapter_config.get("base_model_name_or_path")

        if not base_model_name:
            raise RuntimeError(
                "Could not determine base model from adapter_config.json"
            )

        print(f"Loading saved adapters with base model '{base_model_name}'...")

        try:
            # Load the PEFT model, which includes the base model and adapters
            peft_model = AutoPeftModelForCausalLM.from_pretrained(
                self.lora_model_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None,
                device_map="cuda:0",
            )

            # Merge the LoRA layers into the base model for faster inference
            self.model = peft_model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_dir)
            logger.info("✅ Model loaded and merged successfully onto GPU.")
        except torch.cuda.OutOfMemoryError:
            logger.warning("⚠️ OOM Error. Retrying with CPU offloading...")
            torch.cuda.empty_cache()
            peft_model = AutoPeftModelForCausalLM.from_pretrained(
                self.lora_model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Fallback to automatic offloading
            )
            self.model = peft_model.merge_and_unload()
            self.tokenizer = AutoTokenizer.from_pretrained(self.lora_model_dir)
            logger.info("✅ Model loaded and merged with CPU offloading.")

    def run_inference(self, prompt: str) -> str:
        """Performs inference on a single prompt."""

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        messages: ChatMsg = [{"role": "user", "content": prompt}]

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )  # BatchEncoding

        if isinstance(inputs, BatchEncoding):
            inputs = inputs.to(device="cuda")

        outputs: torch.Tensor = self.model.generate(  # type: ignore
            input_ids=inputs,
            logits_processor=self.logits_processor,
            max_new_tokens=(
                self.max_tokens_per_answer if not self.logits_processor else 1
            ),
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        decoded_output: str = self.tokenizer.batch_decode(
            sequences=outputs, skip_special_tokens=True
        )[0]

        # clean up response by removing the prompt
        # it returns a string when `tokenize=False`
        prompt_template: str = str(
            self.tokenizer.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=True
            )
        )

        # str cast to satisfy the checker.
        clean_response: str = decoded_output.replace(prompt_template, "").strip()

        return clean_response

    def evaluate_on_test_set(self, df_test_data: pd.DataFrame) -> pd.DataFrame:
        """Runs inference on a test DataFrame and parses the results."""
        logger.info("Evaluating on the test set...")

        predictions = [
            self.run_inference(prompt=str(row["text"]))
            for _, row in df_test_data.iterrows()
        ]

        results_df = df_test_data.copy()
        results_df["prediction_full_text"] = predictions
        results_df["predicted_label"] = (
            results_df["prediction_full_text"].str.strip().str.upper()
        )

        logger.info("Evaluation complete.")

        return results_df

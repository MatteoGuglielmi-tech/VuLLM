from unsloth import FastLanguageModel, is_bfloat16_supported

import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.quantization_config import BitsAndBytesConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .shared.typedef import ChatMsg
from .logits_processor import EnforceSingleTokenGeneration, LogitsProcessor


import logging
from .shared.stdout import MY_LOGGER_NAME
logger = logging.getLogger(MY_LOGGER_NAME)


@dataclass
class UnslothTestPipeline:
    """
    A class to handle loading a fine-tuned Unsloth model and running inference.

    This class loads saved LoRA adapters, prepares the model for efficient inference,
    evaluates it on a test set, and calculates performance metrics.

    Attributes:
        lora_model_dir (str): The path to the directory containing the saved LoRA adapters from training.
        max_seq_length (int): The maximum sequence length for the model's tokenizer. Must match the
            value used during training.
        max_tokens_per_answer (int): The maximum number of new tokens to generate during inference.
        model (FastLanguageModel): The loaded, inference-ready model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        output_dir (str): The automatically generated path where evaluation results will be saved.
    """

    lora_model_dir: str
    max_seq_length: int
    max_tokens_per_answer: int = 5
    use_double_quantization: bool = False

    output_dir: str = field(init=False)
    model: FastLanguageModel|None = field(init=False, default=None)
    tokenizer: PreTrainedTokenizer|None = field(init=False, default=None)
    logits_processor: list[LogitsProcessor] = field(init=False)

    def __post_init__(self) -> None:
        lora_path = Path(self.lora_model_dir)
        results_path = lora_path.parent.parent / "results"
        self.output_dir = str(results_path)
        os.makedirs(self.output_dir, exist_ok=True)
        self._unsloth_load_finetuned_model()

        # prepare the logits processor
        if self.tokenizer:
            # Important: get token IDs for the admitted answers
            yes_token_id: int = self.tokenizer.encode(text="YES", add_special_tokens=False)[0]
            no_token_id: int = self.tokenizer.encode(text="NO", add_special_tokens=False)[0]

            self.logits_processor = [EnforceSingleTokenGeneration(allowed_token_ids=[yes_token_id, no_token_id])]

    def _unsloth_load_finetuned_model(self) -> None:
        """Dynamically loads the fine-tuned LoRA model for inference, handling
        potential OOM errors with a fallback to CPU offloading.
        """

        bnb_config_arguments: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": (torch.bfloat16 if is_bfloat16_supported() else None),
            "bnb_4bit_use_double_quant": self.use_double_quantization,
        }

        if self.model:
            return # Avoid reloading if the model is already in memory

        logger.info(f"Loading fine-tuned LoRA model from '{self.lora_model_dir}'...")

        try:
            # Unsloth handles loading the base model and applying the adapter automatically
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name = self.lora_model_dir,
                max_seq_length = self.max_seq_length,
                quantization_config = BitsAndBytesConfig(**bnb_config_arguments),
                device_map = "auto",
                attn_implementation = "flash_attention_2",
            )
            logger.info("✅ Fine-tuned model and tokenizer loaded successfully.")
            # enable Unsloth's native 2x faster inference kernels
            FastLanguageModel.for_inference(self.model)
            logger.info("⚡ Unsloth inference mode enabled.")

        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("⚠️ OOM Error: Model does not fit entirely on GPU.")
            logger.critical(f"Please try a smaller model or use a GPU with more VRAM.")
            torch.cuda.empty_cache()  # Clear cache after OOM error
            raise oom

        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise e

    def run_inference(self, prompt: str) -> str:
        """Performs inference on a single prompt."""

        if not self.model or not self.tokenizer: raise RuntimeError("Model is not loaded.")

        messages: ChatMsg = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")  # BatchEncoding

        if isinstance(inputs, BatchEncoding): inputs = inputs.to(device="cuda")

        outputs: torch.Tensor = self.model.generate(  # type: ignore
            input_ids=inputs,
            logits_processor=self.logits_processor,
            max_new_tokens=(self.max_tokens_per_answer if not self.logits_processor else 1),
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded_output: str = self.tokenizer.batch_decode(sequences=outputs, skip_special_tokens=True)[0]
        # clean up response by removing the prompt, it returns a string when `tokenize=False`
        prompt_template: str = str(
            self.tokenizer.apply_chat_template(conversation=messages, tokenize=False, add_generation_prompt=True)
        )
        clean_response: str = decoded_output.replace(prompt_template, "").strip()

        return clean_response

    def evaluate_on_test_set(self, df_test_data: pd.DataFrame) -> pd.DataFrame:
        """Runs inference on the test dataset and returns the results."""

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        logger.info(f"Evaluating on {len(df_test_data)} test samples...")

        predictions = df_test_data["text"].apply(lambda prompt: self.run_inference(prompt=str(prompt)))

        results_df = df_test_data.copy()
        results_df["prediction_full_text"] = predictions
        results_df["predicted_label"] = (results_df["prediction_full_text"].str.strip().str.upper())

        logger.info("Evaluation complete.")

        return results_df

    @staticmethod
    def calculate_and_save_metrics(y_true: list[str], y_pred: list[str], output_dir: str) -> None:
        """Calculates classification metrics, generates a confusion matrix, and
        saves them to the output directory.

        This method is independent of the class instance.
        """

        # Define the labels and descriptive ticks for the plot
        class_labels: list[str] = ["YES", "NO"]
        tick_labels: list[str] = ["Vulnerable", "Not Vulnerable"]
        output_path: Path = Path(output_dir)

        # 1. Calculate Confusion Matrix and Per-Class Accuracy
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_accuracy = np.divide(cm.diagonal(), cm.sum(axis=1))
            per_class_accuracy[np.isnan(per_class_accuracy)] = 0 # Replace NaN with 0

        report_dict: dict[str, Any] = classification_report(
            y_true=y_true, y_pred=y_pred, labels=class_labels, output_dict=True
        ) # type: ignore

        report_dict["accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
        report_dict["per_class_accuracy"] = { label: acc for label, acc in zip(class_labels, per_class_accuracy) }

        logger.info("Calculating and saving metrics...")
        metrics_path: Path = Path(output_dir) / "classification_metrics.json"
        with open(file=metrics_path, mode="w") as f:
            json.dump(obj=report_dict, fp=f, indent=4)
        logger.info(f"Classification metrics saved to {metrics_path}")

        # 3. Generate and save the classification report for LaTeX
        report_df: pd.DataFrame = (pd.DataFrame(report_dict).transpose().round(decimals=2))
        latex_path: Path = output_path / "classification_report.tex"
        with open(file=latex_path, mode="w") as f:
            f.write(report_df.to_latex(float_format="%.2f", bold_rows=True))
        logger.info(f"LaTeX classification report saved to {latex_path}")

        # 4. Generate and save confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels)

        ax.set_title("Confusion Matrix", fontsize=17, pad=20)
        ax.set_xlabel("Prediction", fontsize=13)
        ax.set_ylabel("Ground truth", fontsize=13)

        # Move x-axis labels and ticks to the top
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()

        cm_path: Path = output_path / "confusion_matrix.svg"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.clf()

        logger.info(f"Confusion matrix saved to {cm_path}")

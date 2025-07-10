import os
import json
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any

import matplotlib.pyplot as plt
import seaborn as sns

# FIX: Unsloth must be imported before transformers and peft
from unsloth import FastLanguageModel, is_bfloat16_supported  # isort: ignore
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.generation.logits_process import LogitsProcessor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .typedef import ChatMsg
from .unsloth_model import fourbit_models
from .logits_processor import EnforceSingleTokenGeneration


# =================================================================================
# CLASS 2: INFERENCE PIPELINE
# =================================================================================
@dataclass
class InferencePipeline:
    """
    A class to handle loading a fine-tuned Unsloth model and running inference.

    This class loads saved LoRA adapters, prepares the model for efficient inference,
    evaluates it on a test set, and calculates performance metrics.

    Attributes:
        lora_model_dir (str): The path to the directory containing the saved LoRA adapters from training.
        max_seq_length (int): The maximum sequence length for the model's tokenizer. Must match the
            value used during training.
        max_tokens_per_answer (int): The maximum number of new tokens to generate during inference.
        PROMPT_TEMPLATE_INFERENCE (str): The f-string template for formatting inference prompts.
        model (FastLanguageModel): The loaded, inference-ready model.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        output_dir (str): The automatically generated path where evaluation results will be saved.
    """

    lora_model_dir: str
    max_seq_length: int = 100
    max_tokens_per_answer: int = 5

    PROMPT_TEMPLATE_INFERENCE: str = (
        "You are an AI system that analyzes C code for vulnerabilities.\n\n"
        "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
        "KEY:Code is chunked; reassemble by function signature to obtain full original source code.\n"
        "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
        "Function signature:\n{signature}\n\n"
        "Code Fragment:\n{subchunk}\n\n"
        "Answer 'YES' if vulnerable, 'NO' otherwise.\n"
        "Correct answer:\n"
    )

    # <---- Attributes for managing paths and models ---->
    model: Optional[FastLanguageModel] = field(init=False, default=None)
    tokenizer: Optional[PreTrainedTokenizer] = field(init=False, default=None)
    output_dir: str = field(init=False)
    logits_processor: list[LogitsProcessor] = field(init=False)

    def __post_init__(self) -> None:
        results_path = self.lora_model_dir.replace("/lora_model", "").replace(
            "trainer", "results", 1
        )
        self.output_dir = results_path
        os.makedirs(self.output_dir, exist_ok=True)
        self._unsloth_load_finetuned_model()

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
                torch.bfloat16 if is_bfloat16_supported() else None
            ),
            # " bnb_4bit_use_double_quant": True,
        }

    def _unsloth_load_finetuned_model(self) -> None:
        """
        Dynamically loads the fine-tuned LoRA model for inference, handling
        potential OOM errors with a fallback to CPU offloading.
        """

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

        # <---- Step 1: Attempt to load everything to GPU ---->
        try:
            print("Attempting to load model directly onto GPU...")
            load_kwargs = {
                "model_name": self.lora_model_dir,
                "max_seq_length": self.max_seq_length,
                "device_map": "cuda:0",
                "attn_implementation": "flash_attention_2",
            }
            if base_model_name in fourbit_models:
                load_kwargs["load_in_4bit"] = True
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    **self.bnb_config_arguments
                )

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                **load_kwargs
            )
            print("✅ Model fits entirely on GPU.")

        except torch.cuda.OutOfMemoryError:
            print("⚠️ OOM Error: Model does not fit entirely on GPU.")
            print("Retrying with automatic CPU offloading enabled...")
            torch.cuda.empty_cache()
            # gc.collect()

            # <---- Step 2: Fallback to dynamic offloading ---->
            try:
                # For offloading to work, we MUST provide the quantization config with the
                self.bnb_config_arguments["llm_int8_enable_fp32_cpu_offload"] = True

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.lora_model_dir,
                    quantization_config=BitsAndBytesConfig(**self.bnb_config_arguments),
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
                print("✅ Model loaded successfully with dynamic CPU offloading.")
            except Exception as e:
                print(f"❌ Failed to load model even with offloading enabled: {e}")
                raise e

        # Final step: prepare for inference
        if self.model:
            FastLanguageModel.for_inference(self.model)

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
        """Runs inference on the test dataset and returns the results."""

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded.")

        print("Evaluating on the test set...")
        predictions = [
            self.run_inference(prompt=str(row["text"]))
            for _, row in df_test_data.iterrows()
        ]

        results_df = df_test_data.copy()
        results_df["prediction_full_text"] = predictions
        results_df["predicted_label"] = (
            results_df["prediction_full_text"].str.strip().str.upper()
        )

        print("Evaluation complete.")

        return results_df

    @staticmethod
    def calculate_and_save_metrics(
        y_true: list[str], y_pred: list[str], output_dir: str
    ) -> None:
        """Calculates classification metrics, generates a confusion matrix, and
        saves them to the output directory.

        This method is independent of the class instance.
        """

        # Define the labels and descriptive ticks for the plot
        class_labels: list[str] = ["YES", "NO"]
        tick_labels: list[str] = ["Vulnerable", "Not Vulnerable"]

        # 1. Calculate Confusion Matrix and Per-Class Accuracy
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_labels)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # 2. Generate and save main metrics report (JSON)
        print("Calculating and saving metrics...")
        report_dict: dict[str, Any] = classification_report(
            y_true=y_true, y_pred=y_pred, labels=class_labels, output_dict=True
        )  # type: ignore

        report_dict["accuracy"] = accuracy_score(y_true=y_true, y_pred=y_pred)
        report_dict["per_class_accuracy"] = {
            label: acc for label, acc in zip(class_labels, per_class_accuracy)
        }

        metrics_path: str = os.path.join(output_dir, "classification_metrics.json")
        with open(file=metrics_path, mode="w") as f:
            json.dump(obj=report_dict, fp=f, indent=4)

        print(f"Classification metrics saved to {metrics_path}")

        # 3. Generate and save the classification report for LaTeX
        report_df: pd.DataFrame = (
            pd.DataFrame(report_dict).transpose().round(decimals=2)
        )
        latex_path: str = os.path.join(output_dir, "classification_report.tex")
        with open(file=latex_path, mode="w") as f:
            f.write(report_df.to_latex(float_format="%.2f", bold_rows=True))

        print(f"LaTeX classification report saved to {latex_path}")

        # 4. Generate and save confusion matrix heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )

        plt.title("Confusion Matrix", fontsize=17, pad=20)
        plt.ylabel("Ground truth", fontsize=13)
        plt.gca().xaxis.set_label_position("top")
        plt.xlabel("Prediction", fontsize=13)
        plt.gca().xaxis.tick_top()
        plt.gca().figure.subplots_adjust(bottom=0.2)
        plt.gca().figure.text(0.5, 0.05, "Prediction", ha="center", fontsize=13)
        cm_path: str = os.path.join(output_dir, "confusion_matrix.svg")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.clf()

        print(f"Confusion matrix saved to {cm_path}")

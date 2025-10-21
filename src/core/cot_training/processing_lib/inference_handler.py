from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import os
import json
import re
import torch
import logging
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer


logger = logging.getLogger(name=__name__)


@dataclass
class TestHandler:
    lora_model_dir: Path
    max_seq_length: int
    max_new_tokens: int
    chat_template: str

    model: FastLanguageModel | None = field(init=False, default=None)
    tokenizer: PreTrainedTokenizer | None = field(init=False, default=None)

    SYSTEM_PROMPT = (
        "You are an expert cybersecurity analyst specializing in C static code analysis. "
        "Your task is to analyze the provided code and produce a step-by-step reasoning "
        "chain explaining whether it contains a vulnerability."
    )

    PROMPT_SKELETON = (
        "**Analysis Instructions:**\n"
        "1.  **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
        "2.  **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
        "3.  **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
        "4.  **Conclude:** State your conclusion based on the analysis.\n\n"
        "**Output Format:**\n"
        "Produce a step-by-step list of your reasoning. After the list, your final answer must be "
        "prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'.\n"
        "--- CODE START ---\n"
        "{func_code}\n"
        "--- CODE END ---\n\n"
        "**Reasoning:**"
    ).strip()

    def __post_init__(self):
        self._load_finetuned_model()

    def _load_finetuned_model(self):
        """Loads the quantized base model first, then applies the fine-tuned LoRA adapter for efficient inference."""

        if self.model:
            return

        # find base-model name
        adapter_config_path = self.lora_model_dir / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"`adapter_config.json` not found in {self.lora_model_dir}"
            )

        with open(file=adapter_config_path, mode="r") as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

        if not base_model_name:
            raise ValueError("base_model_name_or_path not found in adapter_config.json")

        logger.info(f"Found base model '{base_model_name}' from adapter config.")
        logger.info(f"Loading fine-tuned LoRA model from '{self.lora_model_dir}'...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                device_map="auto",
                attn_implementation="flash_attention_2",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                ),
            )
            logger.info("✅ Base model and tokenizer loaded successfully.")

            if self.model is None or self.tokenizer is None:
                raise ValueError("Base model has not been loaded successfully")

            # apply LoRA adapters
            self.model.load_adapter(self.lora_model_dir)  # type: ignore
            logger.info("✅ LoRA adapter applied successfully.")

            # Enable native 2x faster inference
            FastLanguageModel.for_inference(model=self.model)
            logger.info("⚡ Unsloth inference mode enabled.")

            try:
                self.tokenizer = get_chat_template(
                    self.tokenizer, chat_template=self.chat_template
                )
            except ValueError:
                raise ValueError(
                    f"available chat templates are: {print(list(CHAT_TEMPLATES.keys()))}"
                )

            if self.tokenizer is not None: # I know, this is redundant but this is to silence the linter
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("⚠️ OOM Error: Model does not fit entirely on GPU.")
            logger.critical(f"Please try a smaller model or use a GPU with more VRAM.")
            torch.cuda.empty_cache()  # Clear cache after OOM error
            raise oom

        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise e

    def run_inference(self, c_code_input: str) -> str:
        """
        Performs inference on a single C code snippet using the CoT format.

        Parameters
        ----------
        c_code_input : str
            The raw C function code to be analyzed.

        Returns
        -------
        str
            The model's generated reasoning and final answer.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before running inference.")

        # build full prompt
        prompt = self.PROMPT_SKELETON.format(func_code=c_code_input)
        # build message structure
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]

        input_text = self.tokenizer.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,  # type: ignore
            return_tensors="pt",
            padding=True,
            max_length=self.max_seq_length,
            truncation=True,
        ).to(self.model.device)  # type: ignore

        outputs = self.model.generate( #type: ignore
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            min_p=0.1,
        )

        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded_output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return decoded_output.strip()

    def evaluate_on_test_set(
        self, test_dataset: Dataset, batch_size: int = 16
    ) -> Dataset:
        """Runs efficient, batched inference on the test dataset and returns the results.

        Parameters
        ----------
        test_dataset: Dataset
            The test data from a DatasetDict.
        batch_size: int, default 16
            The number of samples to process in each batch.

        Returns
        -------
        Dataset:
            The original dataset with added columns for predictions.
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError(
                "Model and tokenizer must be loaded before running evaluation."
            )

        logger.info(
            f"Evaluating on {len(test_dataset)} test samples with batch size {batch_size}..."
        )

        all_prompts: list[str] = [
            self.PROMPT_SKELETON.format(func_code=func) for func in test_dataset["func"]
        ]

        all_predictions: list[str] = []
        for i in tqdm(
            range(0, len(all_prompts), batch_size), desc="Evaluating Batches"
        ):
            batch_prompts = all_prompts[i : i + batch_size]
            batch_messages = [
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ]
                for prompt in batch_prompts
            ]

            input_text = self.tokenizer.apply_chat_template(
                batch_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,  # type: ignore
                return_tensors="pt",
                padding=True,
                max_length=self.max_seq_length,
                truncation=True,
            ).to(self.model.device)  # type: ignore

            try:
                outputs = self.model.generate(  # type: ignore
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    use_cache=True,
                    temperature=0.2,
                    top_p=0.95,
                    min_p=0.1,
                )

                # decode only the isolated tokens
                decoded_predictions = self.tokenizer.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                all_predictions.extend([p.strip() for p in decoded_predictions])

            except Exception as e:
                raise Exception(f"An error occurred during batch evaluation: {e}")

        results_dataset = test_dataset.add_column("model_prediction", all_predictions)  # type: ignore

        logger.info("✅ Evaluation complete.")

        return results_dataset


# Perform Detailed Error Analysis
# Find False Positives (model predicted 1, but the label was 0):
# false_positives = results.filter(
#     lambda ex: ex["target"] == 0 and ex["predicted_label"] == 1
# )
# Find False Negatives (model predicted 0, but the label was 1):
# false_negatives = results.filter(
#     lambda ex: ex["target"] == 1 and ex["predicted_label"] == 0
# )
#
# You can analyze if the model has biases or performs differently on subsets of your data.
# df = results.to_pandas()
# # Calculate accuracy for each project
# project_accuracy = df.groupby("project").apply(
#     lambda x: (x["target"] == x["predicted_label"]).mean()
# )
# print(project_accuracy)


    @staticmethod
    def quantitative_evaluation(all_predictions: list[str], test_dataset: Dataset):
        """Performs a comprehensive, two-step quantitative evaluation and saves all artifacts."""

        logger.info("\n" + "="*20 + " QUANTITATIVE EVALUATION " + "="*20)

        output_dir: Path = Path(__file__).parent.parent / "assets/"
        output_dir.mkdir(parents=True, exist_ok=True)

        def parse_prediction(prediction: str) -> tuple[str | None, list[str]]:
            """Robust parser for the model's output."""

            # search for the "Final Answer" section, case-insensitive
            match = re.search(pattern=r"Final Answer:.*", string=prediction, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                return None, []

            final_answer_text = match.group(0)

            label = None
            if re.search(pattern=r"\bYES\b|\bVULNERABLE\b", string=final_answer_text, flags=re.IGNORECASE):
                label = "YES"
            elif re.search(pattern=r"\bNO\b|\bNOT VULNERABLE\b", string=final_answer_text, flags=re.IGNORECASE):
                label = "NO"

            cwes = re.findall(pattern=r"CWE-\d+", string=final_answer_text, flags=re.IGNORECASE)

            return label, [c.upper() for c in cwes]

        ground_truth_labels = []
        predicted_labels = []
        ground_truth_cwes_list = []
        predicted_cwes_list = []
        parsing_failures = []

        for i, prediction_text in enumerate(all_predictions):
            pred_label, pred_cwes = parse_prediction(prediction=prediction_text)

            gt_label = "YES" if test_dataset[i]['target'] == 1 else "NO"
            gt_cwes = test_dataset[i]['cwe']

            # if parsing fails, count it as incorrect for the binary task
            if pred_label is None:
                predicted_labels.append("NO" if gt_label == "YES" else "YES")
                parsing_failures.append({"index": i, "prediction_text": prediction_text})
            else:
                predicted_labels.append(pred_label)

            ground_truth_labels.append(gt_label)
            # predicted_cwes_list.append(pred_cwes)
            # ground_truth_cwes_list.append(gt_cwes)
            if gt_label == "YES":
                predicted_cwes_list.append(pred_cwes)
                ground_truth_cwes_list.append(gt_cwes)


        # == Step 1: Evaluate Binary Classification (Vulnerable vs. Not Vulnerable) ==
        print("\n--- Step 1: Binary Classification Performance (YES/NO) ---\n")
        report = classification_report(
            ground_truth_labels, predicted_labels,
            target_names=["NO (Not Vulnerable)", "YES (Vulnerable)"], digits=4
        )
        print(report)

        # Save the report
        report_dict = classification_report(
            ground_truth_labels, predicted_labels,
            target_names=["NO (Not Vulnerable)", "YES (Vulnerable)"], output_dict=True
        )
        with open(file=(output_dir / "classification_report.json"), mode="w") as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"✅ Classification report saved to {output_dir}")

        logger.info(f"Found {len(parsing_failures)} parsing failures.")
        with open(file= (output_dir /"parsing_failures.json"), mode="w") as f:
            json.dump(parsing_failures, f, indent=4)

        # Generate and save the confusion matrix plot
        cm = confusion_matrix(ground_truth_labels, predicted_labels, labels=["YES", "NO"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["YES", "NO"], yticklabels=["YES", "NO"])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Binary Classification Confusion Matrix')
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        logger.info(f"✅ Confusion matrix plot saved to {cm_path}")
        plt.close()


        # == Step 2: Evaluate CWE Identification Performance ==
        print("\n--- Step 2: CWE Identification Performance (for correct YES predictions) ---\n")
        # correct_cwe_predictions = 0
        # total_predicted_cwes = 0
        # total_ground_truth_cwes = 0

        # for i, gt_label in enumerate(ground_truth_labels):
        #     # Only evaluate CWEs for samples that are truly vulnerable AND the model predicted as vulnerable
        #     if gt_label == "YES" and predicted_labels[i] == "YES":
        #         gt_set = set(ground_truth_cwes_list[i])
        #         pred_set = set(predicted_cwes_list[i])
        #
        #         correct_cwe_predictions += len(gt_set.intersection(pred_set))
        #         total_predicted_cwes += len(pred_set)
        #         total_ground_truth_cwes += len(gt_set)
        #
        # cwe_precision = correct_cwe_predictions / total_predicted_cwes if total_predicted_cwes > 0 else 0
        # cwe_recall = correct_cwe_predictions / total_ground_truth_cwes if total_ground_truth_cwes > 0 else 0
        # cwe_f1 = 2 * (cwe_precision * cwe_recall) / (cwe_precision + cwe_recall) if (cwe_precision + cwe_recall) > 0 else 0

        # print(f"CWE Precision: {cwe_precision:.4f}")
        # print(f"CWE Recall:    {cwe_recall:.4f}")
        # print(f"CWE F1-Score:  {cwe_f1:.4f}")
        #
        # cwe_metrics = {
        #     "cwe_precision": cwe_precision,
        #     "cwe_recall": cwe_recall,
        #     "cwe_f1_score": cwe_f1
        # }

        # build vocab
        all_cwes = set()
        for cwe_list in ground_truth_cwes_list:
            all_cwes.update(cwe_list)
        for cwe_list in predicted_cwes_list:
            all_cwes.update(cwe_list)

        sorted_cwes = sorted(list(all_cwes))

        mlb = MultiLabelBinarizer(classes=sorted_cwes)
        y_true_binarized = mlb.fit_transform(ground_truth_cwes_list)
        y_pred_binarized = mlb.transform(predicted_cwes_list)

        # generate the detailed classification report
        cwe_report = classification_report(
            y_true_binarized,
            y_pred_binarized,
            target_names=sorted_cwes,
            digits=4,
            zero_division=0.0 # type: ignore
        )
        print(cwe_report)

        cwe_report_dict = classification_report(
            y_true_binarized,
            y_pred_binarized,
            target_names=sorted_cwes,
            output_dict=True,
            digits=4,
            zero_division=0.0 # type: ignore
        )

        with open(file=(output_dir / "cwe_full_report.json"), mode="w") as f:
            json.dump(cwe_report_dict, f, indent=4)
        logger.info(f"✅ Full CWE performance report saved to {output_dir}")


    @staticmethod
    def qualitative_evaluation(
        all_predictions: list[str],
        test_dataset: Dataset,
        num_samples: int = 50
    ):
        """Saves a number of random samples to a text file for manual qualitative review."""
        logger.info("\n" + "="*20 + " QUALITATIVE EVALUATION " + "="*20)

        if num_samples > len(test_dataset):
            num_samples = len(test_dataset)

        sample_indices = random.sample(range(len(test_dataset)), num_samples)

        output_dir: Path = Path(__file__).parent.parent / "assets/"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "qualitative_review_samples.txt"

        with open(file=report_path, mode='w', encoding='utf-8') as f:
            f.write(f"Qualitative Review: {num_samples} Random Samples\n")
            f.write("="*50 + "\n\n")

            for idx in sample_indices:
                ground_truth_label = "YES" if test_dataset[idx]['target'] == 1 else "NO"
                ground_truth_cwes = ", ".join(test_dataset[idx]['cwe']) if test_dataset[idx]['cwe'] else "N/A"

                f.write(f"--- SAMPLE {idx} ---\n")
                f.write(f"GROUND TRUTH: {ground_truth_label} ({ground_truth_cwes})\n")
                f.write("-" * 20 + "\n")
                f.write("FUNCTION:\n")
                f.write(test_dataset[idx]['func'] + "\n")
                f.write("-" * 20 + "\n")
                f.write("MODEL'S FULL RESPONSE:\n")
                f.write(all_predictions[idx] + "\n\n")
                f.write("="*50 + "\n\n")

        logger.info(f"✅ Qualitative review file saved to: {report_path}")


    @staticmethod
    def analyze_misclassifications(
        all_predictions: list[str],
        test_dataset: Dataset,
    ):
        """
        Identifies misclassified samples and saves them to a file for error analysis.

        Usage example:
        # Run the standard evaluations
        TestHandler.quantitative_evaluation(results_dataset["model_prediction"], results_dataset, handler.output_dir)
        TestHandler.qualitative_evaluation(results_dataset["model_prediction"], results_dataset, handler.output_dir, num_samples=50)

        # Run the targeted error analysis
        TestHandler.analyze_misclassifications(results_dataset["model_prediction"], results_dataset, handler.output_dir)
        """

        logger.info("\n" + "="*20 + " MISCLASSIFICATION ANALYSIS " + "="*20)

        output_dir: Path = Path(__file__).parent.parent / "assets/"
        output_dir.mkdir(parents=True, exist_ok=True)

        def parse_prediction(prediction: str) -> str | None:
            """Robust parser for the model's output."""

            # search for the "Final Answer" section, case-insensitive
            match = re.search(pattern=r"Final Answer:.*", string=prediction, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                return None

            final_answer_text = match.group(0)

            label = None
            if re.search(pattern=r"\bYES\b|\bVULNERABLE\b", string=final_answer_text, flags=re.IGNORECASE):
                label = "YES"
            elif re.search(pattern=r"\bNO\b|\bNOT VULNERABLE\b", string=final_answer_text, flags=re.IGNORECASE):
                label = "NO"

            return label

        false_positives = []
        false_negatives = []

        for i, prediction_text in enumerate(all_predictions):
            pred_label = parse_prediction(prediction_text)
            gt_label = "YES" if test_dataset[i]['target'] == 1 else "NO"

            if pred_label is None or pred_label == gt_label:
                continue

            sample_details = {
                "index": i,
                "function": test_dataset[i]['func'],
                "ground_truth": gt_label,
                "ground_truth_cwes": ", ".join(test_dataset[i]['cwe']) if test_dataset[i]['cwe'] else "N/A",
                "model_response": prediction_text
            }

            if pred_label == "YES" and gt_label == "NO":
                false_positives.append(sample_details)
            elif pred_label == "NO" and gt_label == "YES":
                false_negatives.append(sample_details)

        report_path = output_dir / "misclassification_analysis.txt"
        with open(file=report_path, mode="w", encoding='utf-8') as f:
            f.write("="*20 + " Misclassification Analysis Report " + "="*20 + "\n\n")

            f.write(f"\n--- FALSE POSITIVES ({len(false_positives)} samples) ---\n")
            f.write("The model predicted YES, but the ground truth was NO.\n\n")
            for sample in false_positives:
                f.write(f"--- Sample Index: {sample['index']} ---\n")
                f.write(f"Ground Truth: {sample['ground_truth']}\n")
                f.write("Function:\n" + sample['function'] + "\n\n")
                f.write("Model's Full Response:\n" + sample['model_response'] + "\n\n")

            f.write(f"\n--- FALSE NEGATIVES ({len(false_negatives)} samples) ---\n")
            f.write("The model predicted NO, but the ground truth was YES.\n\n")
            for sample in false_negatives:
                f.write(f"--- Sample Index: {sample['index']} ---\n")
                f.write(f"Ground Truth: {sample['ground_truth']} ({sample['ground_truth_cwes']})\n")
                f.write("Function:\n" + sample['function'] + "\n\n")
                f.write("Model's Full Response:\n" + sample['model_response'] + "\n\n")

        logger.info(f"✅ Misclassification analysis report saved to: {report_path}")

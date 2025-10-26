import re
import json
import logging
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


logger = logging.getLogger(__name__)


@dataclass
class BinaryEvaluationResults:
    """Results from binary classification evaluation."""

    total_samples: int
    valid_samples: int
    unparsable_samples: int
    unparsable_indices: list[int]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]

    @property
    def accuracy(self) -> float:
        """Extract accuracy from classification report."""
        return self.classification_report.get("accuracy", 0.0)

    @property
    def f1_vulnerable(self) -> float:
        """F1-score for vulnerable (YES) class."""
        return self.classification_report.get("YES (Vulnerable)", {}).get("f1-score", 0.0)

@dataclass
class CWEEvaluationResults:
    """Results from CWE classification evaluation."""
    total_vulnerable_samples: int
    valid_samples: int
    samples_missing_cwes: int
    per_cwe_report: dict[str, Any]
    micro_avg_f1: float
    macro_avg_f1: float
    all_cwes: list[str]

    @property
    def micro_avg(self) -> float:
        """Extract accuracy from classification report."""
        return self.micro_avg_f1
    @property
    def macro_avg(self) -> float:
        """Extract accuracy from classification report."""
        return self.macro_avg_f1

    @property
    def encountered_cwe(self) -> list[str]:
        """F1-score for vulnerable (YES) class."""
        return self.all_cwes


@dataclass
class MisclassificationAnalysisResults:
    """Results from misclassification analysis."""

    total_samples: int
    false_positives: int
    false_negatives: int
    false_positive_samples: list[dict[str, Any]]
    false_negative_samples: list[dict[str, Any]]

    @property
    def total_errors(self) -> int:
        return self.false_positives + self.false_negatives

    @property
    def error_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_errors / self.total_samples


@dataclass
class Evaluator:
    """Handles evaluation, metrics computation, and visualization for model predictions."""

    output_dir: Path
    test_dataset: Dataset

    LABEL_YES: str = field(default="YES", init=False)
    LABEL_NO: str = field(default="NO", init=False)

    def __post_init__(self):
        """Setup output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📊 Evaluator initialized. Output dir: {self.output_dir}")

    def parse_prediction(self, prediction: str) -> dict[str, Any]:
        """Robust parser for the model's Chain-of-Thought output.

        Extracts:
            - Binary label (YES/NO)
            - CWE list (if vulnerable)
            - Reasoning chain
            - Parsing success status

        Parameters
        ----------
        prediction : str
            The model's complete generated output (reasoning + final answer).

        Returns
        -------
        dict
            {
                "label": "YES" | "NO" | None,
                "cwes": list[str],  # e.g., ["CWE-119", "CWE-787"]
                "reasoning": str,   # Everything before "Final Answer:"
                "final_answer_text": str,  # Everything after "Final Answer:"
                "parse_success": bool,
            }
        """

        result: dict[str, Any] = {
            "label": None,
            "cwes": [],
            "reasoning": "",
            "final_answer_text": "",
            "parse_success": False,
        }

        conclusion_patterns = [
            r"\bCONCLUSION\b(.+)",
            r"\bFINAL\s+ANSWER\b(.+)",
            r"\bANSWER\b(.+)",
            r"\bEND\b(.+)",
            r"\bVERDICT\b(.+)",
        ]

        answer_match: re.Match|None = None
        for pattern in conclusion_patterns:
            answer_match = re.search(pattern=pattern, string=prediction, flags=re.IGNORECASE|re.DOTALL)
            if answer_match: break

        if not answer_match:
            # logger.warning("No 'Final Answer:' found in prediction")
            result["reasoning"] = prediction.strip()
            return result

        # isolate components
        split_point = answer_match.start()
        result["reasoning"] = prediction[:split_point].strip()
        result["final_answer_text"] = answer_match.group(1).strip()
        final_answer = result["final_answer_text"]

        # ============================================================
        # CRITICAL: Check NO patterns FIRST (higher priority)
        # ============================================================
        no_patterns = [
            r"\bNO\b",                                              # "NO"
            r"\bNOT\s+VULNERABLE\b",                                # "NOT VULNERABLE"
            r"\bISN'T\s+VULNERABLE\b",                              # "ISN'T VULNERABLE"
            r"\bAIN'T\s+VULNERABLE\b",                              # "AIN'T VULNERABLE"
            r"\bIS\s+NOT\s+VULNERABLE\b",                           # "IS NOT VULNERABLE"
            r"\bNOT\s+VULN\b",                                      # "NOT VULN" (shorthand)
            r"\bNO\s+VULNERABILITY\b",                              # "NO VULNERABILITY"
            r"(?<!\bNOT\s)(?<!\bISN'T\s)(?<!\bIS\sNOT\s)\bSAFE\b"   # "SAFE"
        ]

        if any(re.search(pattern, final_answer, re.IGNORECASE) for pattern in no_patterns):
            result["label"] = "NO"
            result["parse_success"] = True
            return result

        # ============================================================
        # Only check YES patterns if NO patterns didn't match
        # ============================================================
        yes_patterns = [
            r"\bYES\b",                         # "YES"
            r"\bVULNERABLE\b",                  # "VULNERABLE"
            r"\bVULN\b",                        # "VULN" (shorthand)
            r"\bCONTAINS\s+VULNERABILITY\b",    # "CONTAINS VULNERABILITY"
            r"\bNOT\s+SAFE\b",                  # "NOT SAFE"
            r"\bISN'T\s+SAFE\b",                # "ISN'T SAFE"
            r"\bIS\s+NOT\s+SAFE\b",             # "IS NOT SAFE"
        ]

        if any(re.search(pattern, final_answer, re.IGNORECASE) for pattern in yes_patterns):
            result["label"] = "YES"
            result["parse_success"] = True

            # extract CWEs (e.g., "YES (CWE-119, CWE-787)")
            # cwe_match = re.search(pattern=r"\(([^)]+)\)", string=final_answer)
            # if cwe_match:
                # cwe_string = cwe_match.group(1) # group 1: extract parenthesis inner content
            # cwes: list[str] = re.findall(pattern=r"CWE-\d+", string=cwe_string, flags=re.IGNORECASE)
            cwes: list[str] = re.findall(pattern=r"CWE-\d+", string=final_answer, flags=re.IGNORECASE)
            if cwes:
                result["cwes"] = [cwe.upper() for cwe in cwes]
            # else:
                # logger.warning(f"Label is YES but no CWEs found in: {final_answer}")
            return result

        # -- UNABLE TO PARSE--
        # logger.warning(f"Could not parse label from: {final_answer[:100]}...")
        return result

    def evaluate_binary_classification(self, predictions: list[str], save_artifacts: bool = True) -> BinaryEvaluationResults:
        """Evaluate binary vulnerability detection (YES/NO).

        Parameters
        ----------
        predictions : list[str]
            Raw model predictions (reasoning + final answer).
        save_artifacts : bool
            Whether to save reports and plots.

        Returns
        -------
        BinaryEvaluationResults
            Comprehensive evaluation metrics.
        """

        parsed_results = self._parse_all_predictions(predictions)
        filtered_data = self._filter_unparsable_samples(parsed_results)

        # compute metrics on valid samples only
        report = self._compute_classification_report(y_true=filtered_data["ground_truth"], y_pred=filtered_data["predictions"])

        # compute confusion matrix
        cm = self._compute_confusion_matrix(y_true=filtered_data["ground_truth"], y_pred=filtered_data["predictions"])
        self._log_binary_summary(
            report=report,
            cm=cm,
            total_samples=len(predictions),
            valid_samples=len(filtered_data["predictions"]),
            unparsable_count=len(filtered_data["unparsable_indices"]),
        )

        if save_artifacts:
            self._save_binary_artifacts(
                report=report,
                cm=cm,
                unparsable_indices=filtered_data["unparsable_indices"],
                unparsable_texts=filtered_data["unparsable_texts"],
            )

        return BinaryEvaluationResults(
            total_samples=len(predictions),
            valid_samples=len(filtered_data["predictions"]),
            unparsable_samples=len(filtered_data["unparsable_indices"]),
            unparsable_indices=filtered_data["unparsable_indices"],
            classification_report=report,
            confusion_matrix=cm.tolist(),
        )

    def _parse_all_predictions(self, predictions: list[str]) -> list[dict[str, Any]]:
        """Parse all model predictions.

        Returns
        -------
        list[dict]
            List of parsed results with ground truth labels.
        """

        logger.info(f"Parsing {len(predictions)} predictions...")

        parsed_results = []
        for i, pred_text in enumerate(predictions):
            sample = self.test_dataset[i]
            gt_label = self.LABEL_YES if sample["target"] == 1 else self.LABEL_NO
            parse_result = self.parse_prediction(pred_text)

            parsed_results.append({
                "index": i,
                "ground_truth": gt_label,
                "predicted_label": parse_result.get("label"),
                "parse_success": parse_result.get("parse_success", False),
                "prediction_text": pred_text,
                "predicted_cwes": parse_result.get("cwes", []),
                "ground_truth_cwes": sample.get("cwe", []),
            })

        return parsed_results


    def _filter_unparsable_samples(self, parsed_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Filter out unparsable samples for unbiased evaluation.

        Returns
        -------
        dict
            {
                "ground_truth": list[str],
                "predictions": list[str],
                "unparsable_indices": list[int],
                "unparsable_texts": list[dict],
            }
        """

        ground_truth = []
        predictions = []
        unparsable_indices = []
        unparsable_texts = []

        for result in parsed_results:
            if result["parse_success"] and result["predicted_label"] is not None:
                ground_truth.append(result["ground_truth"])
                predictions.append(result["predicted_label"])
            else:
                # unparsable sample - exclude from evaluation
                unparsable_indices.append(result["index"])
                unparsable_texts.append({
                    "index": result["index"],
                    "ground_truth": result["ground_truth"],
                    "prediction_text": result["prediction_text"],
                    "predicted_label": result["predicted_label"],
                })

        logger.info(
            f"Valid samples: {len(predictions)} | "
            f"Unparsable: {len(unparsable_indices)} "
            f"({len(unparsable_indices)/len(parsed_results)*100:.2f}%)"
        )

        if unparsable_indices:
            logger.warning(
                f"⚠️  {len(unparsable_indices)} samples excluded from evaluation "
                f"due to parsing failures"
            )

        return {
            "ground_truth": ground_truth,
            "predictions": predictions,
            "unparsable_indices": unparsable_indices,
            "unparsable_texts": unparsable_texts,
        }

    def _compute_classification_report(self, y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
        """Compute classification report with all metrics.

        Returns
        -------
        dict
            Classification report containing accuracy, precision, recall, f1
            for each class, plus macro/weighted averages.
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=[f"{self.LABEL_YES} (Vulnerable)", f"{self.LABEL_NO} (Not Vulnerable)"],
            output_dict=True,
            zero_division=0.0,
        )

        return report

    def _compute_confusion_matrix(self, y_true: list[str], y_pred: list[str]) -> np.ndarray:
        """Compute confusion matrix.

        Returns
        -------
        np.ndarray
            Confusion matrix with shape (2, 2).
            [[TP, FN],
             [FP, TN]]
        """

        return confusion_matrix(y_true, y_pred, labels=[self.LABEL_YES, self.LABEL_NO])


    def _log_binary_summary(self, report: dict[str, Any], cm: np.ndarray, total_samples: int, valid_samples: int, unparsable_count: int) -> None:
        """Log evaluation summary to console."""

        yes_metrics = report[f"{self.LABEL_YES} (Vulnerable)"]
        no_metrics = report[f"{self.LABEL_NO} (Not Vulnerable)"]

        print(f"\n📊 Binary Classification Results:")
        print(f"   Total samples:      {total_samples}")
        print(f"   Valid samples:      {valid_samples}")
        print(f"   Unparsable:         {unparsable_count} ({unparsable_count/total_samples*100:.2f}%)")

        print(f"\n   Overall Accuracy:   {report['accuracy']:.4f}")

        print(f"\n   YES (Vulnerable) Class:")
        print(f"      Precision: {yes_metrics['precision']:.4f}")
        print(f"      Recall:    {yes_metrics['recall']:.4f}")
        print(f"      F1-Score:  {yes_metrics['f1-score']:.4f}")
        print(f"      Support:   {yes_metrics['support']}")

        print(f"\n   NO (Not Vulnerable) Class:")
        print(f"      Precision: {no_metrics['precision']:.4f}")
        print(f"      Recall:    {no_metrics['recall']:.4f}")
        print(f"      F1-Score:  {no_metrics['f1-score']:.4f}")
        print(f"      Support:   {no_metrics['support']}")

        # Confusion matrix breakdown
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]

        print(f"\n   Confusion Matrix:")
        print(f"      True Positives:  {tp:4d}")
        print(f"      True Negatives:  {tn:4d}")
        print(f"      False Positives: {fp:4d}")
        print(f"      False Negatives: {fn:4d}")

    def _save_binary_artifacts(self, report: dict[str, Any], cm: np.ndarray, unparsable_indices: list[int], unparsable_texts: list[dict]) -> None:
        """Save all binary classification artifacts to disk."""

        _ = unparsable_indices

        logger.info(f"Saving Artifacts ...")

        # classification report
        report_path = self.output_dir / "binary_classification_report.json"
        self._save_json(report_path, report)
        logger.info(f"✅ Classification report: {report_path.name}")

        # unparsable samples (if any)
        if unparsable_texts:
            unparsable_path = self.output_dir / "binary_unparsable_samples.json"
            self._save_json(unparsable_path, unparsable_texts)
            logger.info(f"✅ Unparsable samples ({len(unparsable_texts)}): {unparsable_path.name}")

        # confusion matrix plot
        cm_path = self.output_dir / "binary_confusion_matrix.png"
        self._plot_confusion_matrix(
            cm=cm,
            labels=[f"{self.LABEL_YES}\n(Vulnerable)", f"{self.LABEL_NO}\n(Not Vulnerable)"],
            title="Binary Vulnerability Detection",
            save_path=cm_path,
        )
        logger.info(f"✅ Confusion matrix: {cm_path.name}")

    def evaluate_cwe_classification(self, predictions: list[str], save_artifacts: bool = True) -> CWEEvaluationResults:
        """Evaluate CWE identification performance on vulnerable samples.

        Only evaluates samples where BOTH:
        1. Ground truth is vulnerable (YES)
        2. Prediction is vulnerable (YES) and parseable

        This ensures fair evaluation.

        Parameters
        ----------
        predictions : list[str]
            Raw model predictions.
        save_artifacts : bool
            Whether to save reports and plots.

        Returns
        -------
        CWEEvaluationResults
            Per-CWE metrics and aggregated scores.
        """

        logger.info("\n" + "=" * 60)
        logger.info("CWE CLASSIFICATION EVALUATION")
        logger.info("=" * 60)

        cwe_data = self._collect_cwe_pairs(predictions)
        all_cwes = self._build_cwe_vocabulary(cwe_data)
        y_true_bin, y_pred_bin = self._binarize_cwe_labels(cwe_data["ground_truth_cwes"], cwe_data["predicted_cwes"], all_cwes)
        per_cwe_report = self._compute_per_cwe_metrics(y_true_bin, y_pred_bin, all_cwes)
        micro_f1, macro_f1 = self._compute_aggregate_metrics(y_true_bin, y_pred_bin)

        self._log_cwe_summary(
            per_cwe_report=per_cwe_report,
            micro_f1=micro_f1,
            macro_f1=macro_f1,
            total_vulnerable=cwe_data["total_vulnerable"],
            valid_samples=cwe_data["valid_samples"],
            missing_cwes=cwe_data["samples_missing_cwes"],
        )

        if save_artifacts:
            self._save_cwe_artifacts(
                per_cwe_report=per_cwe_report,
                micro_f1=micro_f1,
                macro_f1=macro_f1,
                all_cwes=all_cwes,
                missing_cwe_samples=cwe_data["missing_cwe_samples"],
            )

        return CWEEvaluationResults(
            total_vulnerable_samples=cwe_data["total_vulnerable"],
            valid_samples=cwe_data["valid_samples"],
            samples_missing_cwes=cwe_data["samples_missing_cwes"],
            per_cwe_report=per_cwe_report,
            micro_avg_f1=micro_f1,
            macro_avg_f1=macro_f1,
            all_cwes=all_cwes,
        )

    def _collect_cwe_pairs(self, predictions: list[str]) -> dict[str, Any]:
        """Collect aligned ground truth and predicted CWE lists.

        Only includes samples where:
        - Ground truth is vulnerable (YES)
        - Prediction is vulnerable (YES) and parseable

        Returns
        -------
        dict
            {
                "ground_truth_cwes": list[list[str]],
                "predicted_cwes": list[list[str]],
                "total_vulnerable": int,
                "valid_samples": int,
                "samples_missing_cwes": int,
                "missing_cwe_samples": list[dict],
            }
        """

        logger.info("Collecting CWE pairs from vulnerable samples...")

        ground_truth_cwes = []
        predicted_cwes = []
        missing_cwe_samples = []

        total_vulnerable = 0  # Ground truth vulnerable
        valid_samples = 0  # Both GT and pred are YES with CWEs
        samples_missing_cwes = 0  # Pred is YES but no CWEs extracted

        for i, pred_text in enumerate(predictions):
            # ground-truth
            sample = self.test_dataset[i]
            gt_label = self.LABEL_YES if sample["target"] == 1 else self.LABEL_NO
            gt_cwes = sample.get("cwe", [])

            if gt_label != self.LABEL_YES: # skip safe samples
                continue
            total_vulnerable += 1

            # predictions
            parse_result = self.parse_prediction(pred_text)
            pred_label = parse_result.get("label")
            pred_cwes = parse_result.get("cwes", [])

            if not parse_result.get("parse_success") or pred_label != self.LABEL_YES:
                # skip: either unparseable or predicted as NOT vulnerable
                continue

            if not pred_cwes:
                samples_missing_cwes += 1
                missing_cwe_samples.append({
                    "index": i,
                    "ground_truth_cwes": gt_cwes,
                    "prediction_text": pred_text,
                })

            # even if pred_cwes is empty - it's a valid prediction
            ground_truth_cwes.append(gt_cwes)
            predicted_cwes.append(pred_cwes)
            valid_samples += 1

        logger.info(
            f"Collected {valid_samples} valid CWE pairs "
            f"from {total_vulnerable} vulnerable samples"
        )

        if samples_missing_cwes > 0:
            logger.warning(
                f"⚠️  {samples_missing_cwes} predictions marked YES "
                f"but failed to extract CWEs"
            )

        return {
            "ground_truth_cwes": ground_truth_cwes,
            "predicted_cwes": predicted_cwes,
            "total_vulnerable": total_vulnerable,
            "valid_samples": valid_samples,
            "samples_missing_cwes": samples_missing_cwes,
            "missing_cwe_samples": missing_cwe_samples,
        }

    def _build_cwe_vocabulary(self, cwe_data: dict[str, Any]) -> list[str]:
        """Build complete CWE vocabulary from ground truth and predictions.

        Returns
        -------
        list[str]
            Sorted list of unique CWE identifiers.
        """

        all_cwes = set()
        # collect
        for cwe_list in cwe_data["ground_truth_cwes"]:
            all_cwes.update(cwe_list)
        for cwe_list in cwe_data["predicted_cwes"]:
            all_cwes.update(cwe_list)

        sorted_cwes = sorted(list(all_cwes))

        logger.info(f"CWE vocabulary: {len(sorted_cwes)} unique CWEs")
        logger.debug(f"CWEs: {', '.join(sorted_cwes)}")

        return sorted_cwes

    def _binarize_cwe_labels(
        self,
        ground_truth_cwes: list[list[str]],
        predicted_cwes: list[list[str]],
        all_cwes: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Binarize multi-label CWE lists.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (y_true_binarized, y_pred_binarized)
            Shape: (n_samples, n_cwes)
        """

        mlb = MultiLabelBinarizer(classes=all_cwes)
        y_true_bin = mlb.fit_transform(ground_truth_cwes) # only on real ids
        y_pred_bin = mlb.transform(predicted_cwes)

        logger.debug(f"Binarized shape: {y_true_bin.shape}")

        return y_true_bin, y_pred_bin

    def _compute_per_cwe_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, cwe_names: list[str]) -> dict[str, Any]:
        """Compute precision, recall, F1 for each CWE.

        Returns
        -------
        dict
            Per-CWE metrics from classification_report.
        """

        report = classification_report(
            y_true,
            y_pred,
            target_names=cwe_names,
            output_dict=True,
            digits=4,
            zero_division=0.0,
        )

        return report

    def _compute_aggregate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Compute micro and macro averaged F1 scores.

        Returns
        -------
        tuple[float, float]
            (micro_f1, macro_f1)
        """

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0.0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

        return micro_f1, macro_f1

    def _log_cwe_summary(
        self,
        per_cwe_report: dict[str, Any],
        micro_f1: float,
        macro_f1: float,
        total_vulnerable: int,
        valid_samples: int,
        missing_cwes: int,
    ) -> None:
        """Log CWE evaluation summary."""

        print(f"\n📊 CWE Classification Results:")
        print(f"   Total vulnerable (GT): {total_vulnerable}")
        print(f"   Valid TP samples:      {valid_samples}")
        print(f"   Missing CWEs:          {missing_cwes}")

        print(f"\n   Aggregate Metrics:")
        print(f"      Micro-avg F1: {micro_f1:.4f}  (overall performance)")
        print(f"      Macro-avg F1: {macro_f1:.4f}  (per-CWE average)")

        cwe_f1_scores = [
            (cwe, metrics["f1-score"])
            for cwe, metrics in per_cwe_report.items()
            if cwe not in ["accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"]
        ]
        cwe_f1_scores.sort(key=lambda x: x[1], reverse=True)

        if len(cwe_f1_scores) > 0:
            print(f"\n   Top 5 CWEs by F1:")
            for cwe, f1 in cwe_f1_scores[:5]:
                support = per_cwe_report[cwe]["support"]
                print(f"      {cwe}: {f1:.4f} (n={support})")

            if len(cwe_f1_scores) > 5:
                print(f"\n   Bottom 5 CWEs by F1:")
                for cwe, f1 in cwe_f1_scores[-5:]:
                    support = per_cwe_report[cwe]["support"]
                    logger.info(f"      {cwe}: {f1:.4f} (n={support})")

    def _save_cwe_artifacts(
        self,
        per_cwe_report: dict[str, Any],
        micro_f1: float,
        macro_f1: float,
        all_cwes: list[str],
        missing_cwe_samples: list[dict],
    ) -> None:
        """Save CWE evaluation artifacts."""

        logger.info(f"--- Saving CWE Artifacts ---")

        report_path = self.output_dir / "cwe_classification_report.json"
        full_report = {
            "micro_avg_f1": micro_f1,
            "macro_avg_f1": macro_f1,
            "per_cwe_metrics": per_cwe_report,
        }
        self._save_json(report_path, full_report)
        logger.info(f"✅ CWE report: {report_path.name}")

        if missing_cwe_samples:
            missing_path = self.output_dir / "cwe_missing_samples.json"
            self._save_json(missing_path, missing_cwe_samples)
            logger.info(f"✅ Missing CWE samples ({len(missing_cwe_samples)}): {missing_path.name}")

        self._plot_per_cwe_performance(per_cwe_report, all_cwes)


    def analyze_misclassifications(
        self,
        predictions: list[str],
        save_artifacts: bool = True,
        include_code: bool = True,
        max_response_length: int = 1000,
    ) -> MisclassificationAnalysisResults:
        """Identify and analyze misclassified samples (False Positives & False Negatives).

        Parameters
        ----------
        predictions : list[str]
            Raw model predictions.
        save_artifacts : bool
            Whether to save detailed reports to disk.
        include_code : bool
            Whether to include full function code in reports.
        max_response_length : int
            Maximum length of model response to include (prevents huge files).

        Returns
        -------
        MisclassificationAnalysisResults
            Detailed breakdown of misclassifications.
        """

        logger.info("\n" + "=" * 60)
        logger.info("MISCLASSIFICATION ANALYSIS")
        logger.info("=" * 60)

        fp_samples, fn_samples = self._collect_misclassifications(
            predictions=predictions,
            include_code=include_code,
            max_response_length=max_response_length,
        )
        self._log_misclassification_summary(
            false_positives=fp_samples,
            false_negatives=fn_samples,
            total_samples=len(predictions),
        )
        if save_artifacts:
            self._save_misclassification_artifacts(fp_samples, fn_samples)

        return MisclassificationAnalysisResults(
            total_samples=len(predictions),
            false_positives=len(fp_samples),
            false_negatives=len(fn_samples),
            false_positive_samples=fp_samples,
            false_negative_samples=fn_samples,
        )

    def _collect_misclassifications(
        self,
        predictions: list[str],
        include_code: bool,
        max_response_length: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Collect False Positives and False Negatives.

        Returns
        -------
        tuple[list[dict], list[dict]]
            (false_positives, false_negatives)
        """

        logger.info(f"Analyzing {len(predictions)} predictions for misclassifications...")

        false_positives = []
        false_negatives = []

        for i, pred_text in enumerate(predictions):
            sample = self.test_dataset[i]
            gt_label = self.LABEL_YES if sample["target"] == 1 else self.LABEL_NO

            parse_result = self.parse_prediction(pred_text)
            pred_label = parse_result.get("label")

            # skip if unparseable and correct
            if not parse_result.get("parse_success") or pred_label is None: continue
            if pred_label == gt_label: continue

            truncated_response = (
                pred_text[:max_response_length] + "...[truncated]"
                if len(pred_text) > max_response_length
                else pred_text
            )

            sample_details = {
                "index": i,
                "ground_truth_label": gt_label,
                "predicted_label": pred_label,
                "ground_truth_cwes": sample.get("cwe", []),
                "predicted_cwes": parse_result.get("cwes", []),
                "model_response": truncated_response,
                "response_full_length": len(pred_text),
            }

            if include_code:
                sample_details["function_code"] = sample["func"]
            else:
                sample_details["function_length"] = len(sample["func"])

            if pred_label == self.LABEL_YES and gt_label == self.LABEL_NO:
                # False Positive: Model said vulnerable, but it's not
                false_positives.append(sample_details)
            elif pred_label == self.LABEL_NO and gt_label == self.LABEL_YES:
                # False Negative: Model said safe, but it's vulnerable
                false_negatives.append(sample_details)

        logger.info(
            f"Found {len(false_positives)} False Positives and "
            f"{len(false_negatives)} False Negatives"
        )

        return false_positives, false_negatives

    def _log_misclassification_summary(
        self,
        false_positives: list[dict[str, Any]],
        false_negatives: list[dict[str, Any]],
        total_samples: int,
    ) -> None:
        """Log summary statistics about misclassifications."""

        total_errors = len(false_positives) + len(false_negatives)
        error_rate = total_errors / total_samples if total_samples > 0 else 0.0

        print(f"\n📊 Misclassification Summary:")
        print(f"   Total samples:      {total_samples}")
        print(f"   Total errors:       {total_errors} ({error_rate:.2%})")
        print(f"   False Positives:    {len(false_positives)} ({len(false_positives)/total_samples:.2%})")
        print(f"   False Negatives:    {len(false_negatives)} ({len(false_negatives)/total_samples:.2%})")

        # what vulnerabilities were missed?
        if false_negatives:
            missed_cwes = {}
            for fn in false_negatives:
                for cwe in fn["ground_truth_cwes"]:
                    missed_cwes[cwe] = missed_cwes.get(cwe, 0) + 1

            print(f"\n   Missed CWEs (False Negatives):")
            for cwe, count in sorted(missed_cwes.items(), key=lambda x: x[1], reverse=True):
                print(f"      {cwe}: {count} times")

        # what did model hallucinate?
        if false_positives:
            hallucinated_cwes = {}
            for fp in false_positives:
                for cwe in fp["predicted_cwes"]:
                    hallucinated_cwes[cwe] = hallucinated_cwes.get(cwe, 0) + 1

            if hallucinated_cwes:
                print(f"\n   Hallucinated CWEs (False Positives):")
                for cwe, count in sorted(hallucinated_cwes.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {cwe}: {count} times")

    def _save_misclassification_artifacts(
        self,
        false_positives: list[dict[str, Any]],
        false_negatives: list[dict[str, Any]],
    ) -> None:
        """Save misclassification reports in multiple formats."""

        logger.info("--- Saving Misclassification Artifacts ---")

        json_path = self.output_dir / "misclassifications.json"
        json_data = {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "summary": {
                "total_fp": len(false_positives),
                "total_fn": len(false_negatives),
                "total_errors": len(false_positives) + len(false_negatives),
            }
        }
        self._save_json(json_path, json_data)
        logger.info(f"✅ JSON report: {json_path.name}")

        txt_path = self.output_dir / "misclassifications_report.txt"
        self._save_misclassification_text_report(
            txt_path,
            false_positives,
            false_negatives,
        )
        logger.info(f"✅ Text report: {txt_path.name}")

        csv_path = self.output_dir / "misclassifications.csv"
        self._save_misclassification_csv(
            csv_path,
            false_positives,
            false_negatives,
        )
        logger.info(f"✅ CSV report: {csv_path.name}")


    def _save_misclassification_text_report(
        self,
        filepath: Path,
        false_positives: list[dict[str, Any]],
        false_negatives: list[dict[str, Any]],
    ) -> None:
        """Save human-readable text report."""

        with open(file=filepath, mode="w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MISCLASSIFICATION ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Summary
            total_errors = len(false_positives) + len(false_negatives)
            f.write(f"Total Misclassifications: {total_errors}\n")
            f.write(f"  - False Positives: {len(false_positives)}\n")
            f.write(f"  - False Negatives: {len(false_negatives)}\n\n")

            # False Positives
            f.write("=" * 80 + "\n")
            f.write(f"FALSE POSITIVES ({len(false_positives)} samples)\n")
            f.write("=" * 80 + "\n")
            f.write("Model predicted VULNERABLE, but ground truth is NOT VULNERABLE\n\n")

            for i, sample in enumerate(false_positives, 1):
                f.write(f"\n{'─' * 80}\n")
                f.write(f"FALSE POSITIVE #{i} (Index: {sample['index']})\n")
                f.write(f"{'─' * 80}\n")
                f.write(f"Ground Truth:    {sample['ground_truth_label']}\n")
                f.write(f"Prediction:      {sample['predicted_label']}\n")
                f.write(f"Predicted CWEs:  {', '.join(sample['predicted_cwes']) if sample['predicted_cwes'] else 'None'}\n\n")

                if "function_code" in sample:
                    f.write("Function Code:\n")
                    f.write("-" * 40 + "\n")
                    f.write(sample["function_code"] + "\n")
                    f.write("-" * 40 + "\n\n")

                f.write("Model Response:\n")
                f.write("-" * 40 + "\n")
                f.write(sample["model_response"] + "\n")
                f.write("-" * 40 + "\n\n")

            # False Negatives
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"FALSE NEGATIVES ({len(false_negatives)} samples)\n")
            f.write("=" * 80 + "\n")
            f.write("Model predicted NOT VULNERABLE, but ground truth is VULNERABLE\n\n")

            for i, sample in enumerate(false_negatives, 1):
                f.write(f"\n{'─' * 80}\n")
                f.write(f"FALSE NEGATIVE #{i} (Index: {sample['index']})\n")
                f.write(f"{'─' * 80}\n")
                f.write(f"Ground Truth:      {sample['ground_truth_label']}\n")
                f.write(f"Prediction:        {sample['predicted_label']}\n")
                f.write(f"Ground Truth CWEs: {', '.join(sample['ground_truth_cwes'])}\n\n")

                if "function_code" in sample:
                    f.write("Function Code:\n")
                    f.write("-" * 40 + "\n")
                    f.write(sample["function_code"] + "\n")
                    f.write("-" * 40 + "\n\n")

                f.write("Model Response:\n")
                f.write("-" * 40 + "\n")
                f.write(sample["model_response"] + "\n")
                f.write("-" * 40 + "\n\n")


    def _save_misclassification_csv(
        self,
        filepath: Path,
        false_positives: list[dict[str, Any]],
        false_negatives: list[dict[str, Any]],
    ) -> None:
        """Save misclassifications as CSV for spreadsheet analysis."""
        with open(file=filepath, mode="w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "error_type",
                "index",
                "ground_truth_label",
                "predicted_label",
                "ground_truth_cwes",
                "predicted_cwes",
                "function_length",
                "response_length",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write False Positives
            for sample in false_positives:
                writer.writerow({
                    "error_type": "False Positive",
                    "index": sample["index"],
                    "ground_truth_label": sample["ground_truth_label"],
                    "predicted_label": sample["predicted_label"],
                    "ground_truth_cwes": ", ".join(sample["ground_truth_cwes"]),
                    "predicted_cwes": ", ".join(sample["predicted_cwes"]),
                    "function_length": sample.get("function_length", len(sample.get("function_code", ""))),
                    "response_length": sample["response_full_length"],
                })

            # Write False Negatives
            for sample in false_negatives:
                writer.writerow({
                    "error_type": "False Negative",
                    "index": sample["index"],
                    "ground_truth_label": sample["ground_truth_label"],
                    "predicted_label": sample["predicted_label"],
                    "ground_truth_cwes": ", ".join(sample["ground_truth_cwes"]),
                    "predicted_cwes": ", ".join(sample["predicted_cwes"]),
                    "function_length": sample.get("function_length", len(sample.get("function_code", ""))),
                    "response_length": sample["response_full_length"],
                })
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    def _save_json(self, filepath: Path, obj: dict | list) -> None:
        """Save object as JSON with error handling."""
        try:
            with open(file=filepath, mode="w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            raise

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: list[str],
        title: str,
        save_path: Path,
    ) -> None:
        """Generate and save confusion matrix heatmap."""

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            logger.error(f"Failed to save confusion matrix: {e}")
        finally:
            plt.close(fig)

    def _plot_per_cwe_performance(
        self,
        per_cwe_report: dict[str, Any],
        all_cwes: list[str],
    ) -> None:
        """Create visualizations for per-CWE performance.

        Generates:
        1. Bar chart: F1-scores per CWE
        2. Heatmap: Precision/Recall/F1 per CWE
        3. Support distribution: Sample counts per CWE

        Parameters
        ----------
        per_cwe_report : dict
            Classification report with per-CWE metrics.
        all_cwes : list[str]
            List of all CWE identifiers.
        """

        logger.info("Generating CWE performance visualizations...")

        cwe_metrics = []
        for cwe in all_cwes:
            if cwe in per_cwe_report:
                metrics = per_cwe_report[cwe]
                cwe_metrics.append({
                    "cwe": cwe,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"],
                })

        if not cwe_metrics:
            logger.warning("No CWE metrics to plot")
            return

        cwe_metrics.sort(key=lambda x: x["f1-score"], reverse=True)

        self._plot_f1_bar_chart(cwe_metrics)
        self._plot_metrics_heatmap(cwe_metrics)
        self._plot_support_distribution(cwe_metrics)
        self._plot_top_cwes_radar(cwe_metrics)

    def _plot_f1_bar_chart(self, cwe_metrics: list[dict[str, Any]]) -> None:
        """Bar chart showing F1-score for each CWE, sorted by performance."""

        cwes = [m["cwe"] for m in cwe_metrics]
        f1_scores = [m["f1-score"] for m in cwe_metrics]
        supports = [m["support"] for m in cwe_metrics]

        fig, ax = plt.subplots(figsize=(12, max(6, len(cwes) * 0.4)))

        bars = ax.barh(cwes, f1_scores, color="steelblue", edgecolor="black", linewidth=0.5)

        for bar, f1, support in zip(bars, f1_scores, supports):
            width = bar.get_width()
            ax.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{f1:.3f} (n={support})",
                ha="left",
                va="center",
                fontsize=9,
            )

        # Styling
        ax.set_xlabel("F1-Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("CWE", fontsize=12, fontweight="bold")
        ax.set_title("Per-CWE F1-Score Performance", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim(0, 1.1)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="0.5 threshold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.legend(loc="lower right")

        plt.tight_layout()

        save_path = self.output_dir / "cwe_f1_scores.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ F1 bar chart: {save_path.name}")
        plt.close(fig)

    def _plot_metrics_heatmap(self, cwe_metrics: list[dict[str, Any]]) -> None:
        """Heatmap showing Precision, Recall, and F1-score for each CWE."""

        cwes = [m["cwe"] for m in cwe_metrics]
        data = {
            "Precision": [m["precision"] for m in cwe_metrics],
            "Recall": [m["recall"] for m in cwe_metrics],
            "F1-Score": [m["f1-score"] for m in cwe_metrics],
        }

        df = pd.DataFrame(data, index=cwes)

        fig, ax = plt.subplots(figsize=(8, max(6, len(cwes) * 0.4)))
        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score"},
            linewidths=0.5,
            linecolor="gray",
            ax=ax,
        )

        # Styling
        ax.set_title("Per-CWE Metrics Heatmap", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
        ax.set_ylabel("CWE", fontsize=12, fontweight="bold")
        plt.yticks(rotation=0)

        plt.tight_layout()

        save_path = self.output_dir / "cwe_metrics_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Metrics heatmap: {save_path.name}")
        plt.close(fig)

    def _plot_support_distribution(self, cwe_metrics: list[dict[str, Any]]) -> None:
        """Bar chart showing sample count (support) for each CWE. Helps identify class imbalance."""

        sorted_metrics = sorted(cwe_metrics, key=lambda x: x["support"], reverse=True)

        cwes = [m["cwe"] for m in sorted_metrics]
        supports = [m["support"] for m in sorted_metrics]
        f1_scores = [m["f1-score"] for m in sorted_metrics]

        fig, ax = plt.subplots(figsize=(12, max(6, len(cwes) * 0.4)))
        bars = ax.barh(cwes, supports, edgecolor="black", linewidth=0.5)

        norm = plt.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
        sm.set_array([])

        for bar, f1 in zip(bars, f1_scores):
            bar.set_color(sm.to_rgba(f1))

        for bar, support in zip(bars, supports):
            width = bar.get_width()
            ax.text(
                width + max(supports) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{support}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        # Styling
        ax.set_xlabel("Sample Count (Support)", fontsize=12, fontweight="bold")
        ax.set_ylabel("CWE", fontsize=12, fontweight="bold")
        ax.set_title(
            "CWE Support Distribution (colored by F1-score)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Add colorbar
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("F1-Score", fontsize=10, fontweight="bold")

        plt.tight_layout()

        save_path = self.output_dir / "cwe_support_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Support distribution: {save_path.name}")
        plt.close(fig)

    def _plot_top_cwes_radar(self, cwe_metrics: list[dict[str, Any]], top_n: int = 5) -> None:
        """Radar chart showing Precision/Recall/F1 for top N CWEs by support."""
        top_metrics = sorted(cwe_metrics, key=lambda x: x["support"], reverse=True)[:top_n]

        if len(top_metrics) < 3:
            logger.warning("Not enough CWEs for radar chart (need at least 3)")
            return

        categories = ["Precision", "Recall", "F1-Score"]
        num_vars = len(categories)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
        for metric in top_metrics:
            values = [
                metric["precision"],
                metric["recall"],
                metric["f1-score"],
            ]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=f"{metric['cwe']} (n={metric['support']})")
            ax.fill(angles, values, alpha=0.15)

        # Styling
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title(f"Top {top_n} CWEs: Metrics Comparison", fontsize=14, fontweight="bold", pad=30)

        plt.tight_layout()

        save_path = self.output_dir / "cwe_top_radar.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Radar chart: {save_path.name}")
        plt.close(fig)

    def save_evaluation_summary(
        self,
        output_dir: Path,
        binary_results,
        cwe_results,
        misclass_results,
    ) -> None:
        """Save a consolidated evaluation summary."""

        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "binary_classification": {
                "accuracy": binary_results.accuracy,
                "f1_vulnerable": binary_results.f1_vulnerable,
                "total_samples": binary_results.total_samples,
                "valid_samples": binary_results.valid_samples,
                "unparsable_samples": binary_results.unparsable_samples,
            },
            "cwe_classification": {
                "macro_avg_f1": cwe_results.macro_avg_f1,
                "micro_avg_f1": cwe_results.micro_avg_f1,
                "num_unique_cwes": len(cwe_results.all_cwes),
                "valid_samples": cwe_results.valid_samples,
                "samples_missing_cwes": cwe_results.samples_missing_cwes,
            },
            "misclassifications": {
                "total_errors": misclass_results.total_errors,
                "error_rate": misclass_results.error_rate,
                "false_positives": misclass_results.false_positives,
                "false_negatives": misclass_results.false_negatives,
            },
        }

        summary_path = output_dir / "evaluation_summary.json"
        self._save_json(filepath=summary_path, obj=summary)
        logger.info(f"✅ Evaluation summary saved to: {summary_path.name}")

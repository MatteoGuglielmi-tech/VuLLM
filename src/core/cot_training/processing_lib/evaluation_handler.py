import csv
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import Normalize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Iterator, cast
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from ..utilities import (build_panel, build_table, rich_panel,
                         rich_panels_grid, rich_progress_manual, rich_status)
from .prompt_config import ParsedResponse

logger = logging.getLogger(__name__)


@dataclass
class CWEPair:
    cwes_gt: list[int]
    cwes_pred: list[int]

    @property
    def ground_truth(self) -> list[int]:
        return self.cwes_gt

    @property
    def predicted(self) -> list[int]:
        return self.cwes_pred

    @property
    def as_tuple(self) -> tuple[list[int], list[int]]:
        return self.cwes_gt, self.cwes_pred

    @property
    def as_flatten_tuple(self) -> tuple[int, ...]:
        return (*self.cwes_gt, *self.cwes_pred)

    def __iter__(self) -> Iterator[list[int]]:
        """Allow unpacking: gt, pred = pair"""
        yield self.cwes_gt
        yield self.cwes_pred


@dataclass
class CWEEvaluationResults:
    """Results from CWE classification evaluation."""
    per_cwe_metrics: dict[str, Any]
    aggreate_metrics: dict[str, float]
    vocabulary: list[int]
    n_samples: int
    n_classes: int

    @property
    def micro_avg(self) -> float:
        """Extract accuracy from classification report."""
        return self.aggreate_metrics["micro_avg_f1"]

    @property
    def macro_avg(self) -> float:
        """Extract accuracy from classification report."""
        return self.aggreate_metrics["macro_f1"]

    @property
    def cwe_vocabulary(self) -> list[int]:
        """F1-score for vulnerable (YES) class."""
        return self.vocabulary

    def to_dict(self):
        return {
            "per_cwe_metrics": self.per_cwe_metrics,
            "aggregate_metrics": self.aggreate_metrics,
            "vocabulary": self.vocabulary,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
        }


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
    """
    Handles evaluation, metrics computation, and visualization for model predictions.

    Expected dataset fields:
        - func: str - C function source code
        - target: int - Binary ground truth (0=safe, 1=vulnerable)
        - cwe: list[int] - Ground truth CWE IDs
        - model_prediction: ParsedResponse - Model's parsed JSON output
    """

    output_dir: Path
    test_dataset: Dataset

    results: list[dict] = field(default_factory=list, init=False, repr=False)
    parse_failures: list[dict] = field(default_factory=list, init=False, repr=False)
    metrics: Optional[dict] = field(default=None, init=False, repr=False)

    n_samples: int = field(default=0, init=False, repr=False)
    n_parse_failures: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        """Setup output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "errors").mkdir(exist_ok=True)

        if len(self.test_dataset) == 0:
            raise ValueError("Test dataset is empty")

        self.n_samples = len(self.test_dataset)

        sample: dict[str, Any] = self.test_dataset[0]
        required_fields: list[str] = ["func", "target", "cwe", "model_prediction"]
        missing_fields: list[str] = [f for f in required_fields if f not in sample]

        if missing_fields:
            raise ValueError(f"Dataset missing fields: {missing_fields}")

        if not isinstance(sample["model_prediction"], ParsedResponse):
            raise TypeError(
                f"Expected model_prediction to be ParsedResponse, "
                f"got {type(sample['model_prediction'])}"
            )

        logger.info("📊 Evaluator initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Test samples: {self.n_samples}")
        logger.info(f"  Required fields: ✓")

    def evaluate_binary_classification(self, save_artifacts: bool = True) -> None:
        """Evaluate binary vulnerability detection (YES/NO).

        Parameters
        ----------
        save_artifacts : bool
            Whether to save reports and plots.

        Returns
        -------
        BinaryEvaluationResults
            Comprehensive evaluation metrics.
        """

        logger.info("Starting binary classifiaction evaluation ...")
        self._parse_predictions()
        binary_metrics = self._compute_binary_metrics()

        if save_artifacts:
            self._save_binary_artifacts(binary_metrics=binary_metrics)

    def validate_cwe_format(self) -> None:
        """
        Validate that CWE fields are properly formatted in predictions.

        Checks:
        - CWEs are integers (not strings like "CWE-119")
        - CWEs are only present when is_vulnerable=True
        - CWE list is empty when is_vulnerable=False
        """

        invalid_cwes: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []  # For non-critical issues
        parse_errors: int = 0

        with rich_progress_manual(
            total=self.n_samples, description="Validating CWE format ..."
        ) as pbar:
            for idx in range(self.n_samples):
                sample: dict[str, Any] = self.test_dataset[idx]
                try:
                    prediction: ParsedResponse = sample["model_prediction"]
                    if prediction.parse_error:
                        parse_errors += 1
                        pbar.update(advance=1)
                        pbar.set_postfix({
                            "✗ Parse errors": parse_errors,
                            "✗ Format errors": len(invalid_cwes),
                            "⚠ Warnings": len(warnings),
                            "Error rate": (
                                f"{len(invalid_cwes)/self.n_samples:.1%}"
                                if invalid_cwes
                                else "0.0%"
                            ),
                        })

                    pred_cwes: list[int] = prediction.cwe_list
                    is_pred_vulnerable: Optional[bool] = prediction.is_vulnerable

                    if is_pred_vulnerable and pred_cwes:
                        warnings.append({
                            "index": idx,
                            "issue": "Predicted vulnerable but CWE list is empty",
                            "is_vulnerable": is_pred_vulnerable,
                            "cwe_list": pred_cwes,
                        })

                    if not is_pred_vulnerable and pred_cwes:
                        warnings.append({
                            "index": idx,
                            "issue": "Predicted safe but CWE list is non-empty",
                            "is_vulnerable": is_pred_vulnerable,
                            "cwe_list": pred_cwes
                        })

                    if pred_cwes and is_pred_vulnerable:
                        if not all(isinstance(cwe, int) for cwe in pred_cwes):
                            invalid_cwes.append({
                                "index": idx,
                                "issue": "CWE list contains non-integer values",
                                "pred_cwes": pred_cwes,
                                "types": [type(cwe).__name__ for cwe in pred_cwes],
                            })

                except Exception as e:
                    invalid_cwes.append({
                        "index": idx,
                        "issue": f"CWE validation error: {e}",
                        "raw_cwe": sample.get("cwe", "missing"),
                        "prediction": str(
                            sample.get("model_prediction", "missing")
                        ),
                    })
                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix({
                        "✗ Parse errors": parse_errors,
                        "✗ Format errors": len(invalid_cwes),
                        "⚠ Warnings": len(warnings),
                        "Format error rate": (
                            f"{len(invalid_cwes)/self.n_samples:.1%}"
                            if invalid_cwes
                            else "0.0%"
                        ),
                    })

            if invalid_cwes:
                logger.error(
                    f"❌ Found {len(invalid_cwes)} samples with INVALID CWE format"
                )

                invalid_file = self.output_dir / "errors" / "invalid_cwes.json"
                with open(file=invalid_file, mode="w", encoding="utf-8") as f:
                    json.dump(invalid_cwes, f, indent=2, default=str)

                logger.error(f"Invalid CWEs saved to {invalid_file}")

            if warnings:
                logger.warning(
                    f"⚠️  Found {len(warnings)} samples with CWE inconsistencies"
                )

                warnings_file = self.output_dir / "errors" / "cwe_warnings.json"
                with open(file=warnings_file, mode="w", encoding="utf-8") as f:
                    json.dump(warnings, f, indent=2, default=str)

                logger.warning(f"CWE warnings saved to {warnings_file}")

            if not invalid_cwes and not warnings:
                logger.info("✅ All CWEs properly formatted and consistent")

            stats = {
                "total_samples": self.n_samples,
                "invalid_format": len(invalid_cwes),
                "inconsistencies": len(warnings),
                "valid": self.n_samples - len(invalid_cwes),
            }

            tb = build_table(data=stats, columns=["Stat", "Value"])
            rich_panel(
                tables=tb,
                panel_title="CWE format statistics",
                border_style="light_steel_blue",
            )
            del tb

    def _parse_predictions(self) -> None:
        """
        Extract predictions and ground truths from dataset.
        Populates self.results and self.parse_failures.
        """

        with rich_progress_manual(
            total=self.n_samples,
            description="Extracting predictions and ground truths...",
        ) as pbar:
            for idx in range(self.n_samples):
                sample = self.test_dataset[idx]

                prediction: ParsedResponse = sample["model_prediction"]
                gt_binary_label: bool = bool(sample["target"])
                gt_cwes: list[int] = (
                    list(map(lambda x: int(x.split("-")[1]), sample["cwe"]))
                    if sample["cwe"]
                    else []
                )

                if (
                    prediction.parse_error
                    or prediction.is_vulnerable is None
                    or not isinstance(prediction.is_vulnerable, bool)
                ):
                    self.parse_failures.append(
                        {
                            "index": idx,
                            "func": sample["func"][:500],
                            "gt_vulnerable": gt_binary_label,
                            "gt_cwes": gt_cwes,
                            "error": "Parse failure - invalid JSON or missing verdict",
                            "raw_verdict": (
                                prediction.verdict if not prediction.parse_error else None
                            ),
                        }
                    )
                    self.n_parse_failures += 1

                    pbar.update(advance=1)
                    pbar.set_postfix({
                        "✓ Valid": len(self.results),
                        "✗ Failures": self.n_parse_failures,
                        "Error rate (%)": (
                            f"{self.n_parse_failures/self.n_samples:.1%}"
                            if self.n_parse_failures > 0
                            else "0%"
                        ),
                    })
                    continue

                is_pred_vulnerable: bool = prediction.is_vulnerable
                pred_cwe_list: list[int] = prediction.cwe_list
                pred_confidence: Optional[float] = prediction.confidence

                result = {
                    "index": idx,
                    "gt_vulnerable": gt_binary_label,
                    "gt_cwes": set(gt_cwes),
                    "pred_vulnerable": is_pred_vulnerable,
                    "pred_cwes": set(pred_cwe_list),
                    "confidence": (
                        pred_confidence if pred_confidence is not None else 0.0
                    ),
                    "correct_binary": (gt_binary_label == is_pred_vulnerable),
                    "correct_cwes": (
                        (set(gt_cwes) == set(pred_cwe_list))
                        if gt_binary_label and is_pred_vulnerable
                        else None
                    ),
                    "tp": gt_binary_label and is_pred_vulnerable,
                    "tn": (not gt_binary_label) and (not is_pred_vulnerable),
                    "fp": (not gt_binary_label) and is_pred_vulnerable,
                    "fn": gt_binary_label and (not is_pred_vulnerable),
                }

                self.results.append(result)

                pbar.update(advance=1)
                pbar.set_postfix(
                    {
                        "✓ Valid": len(self.results),
                        "✗ Failures": self.n_parse_failures,
                        "Error rate (%)": (
                            f"{self.n_parse_failures/self.n_samples:.1%}"
                            if self.n_parse_failures > 0
                            else "0%"
                        ),
                    }
                )

        logger.info(f"✓ Extracted {len(self.results)} valid predictions")
        logger.info(
            f"✗ Parse failures: {self.n_parse_failures} "
                f"({self.n_parse_failures/self.n_samples:.1%})"
        )
        # Save parse failures for analysis
        if self.parse_failures:
            self._save_parse_failures()
        else:
            logger.info("✅ No parse failures detected")

    def _analyze_confidence_distribution(self) -> None:
        """Analyze confidence score distribution for valid predictions."""
        if not self.results:
            logger.warning("No results to analyze confidence")
            return

        confidences: list[float] = [r["confidence"] for r in self.results]

        correct_confidences = [
            r["confidence"] for r in self.results if r["correct_binary"]
        ]
        incorrect_confidences = [
            r["confidence"] for r in self.results if not r["correct_binary"]
        ]

        mean_conf_correct = float(f"{np.mean(correct_confidences):.3f}")
        mean_conf_incorrect = float(f"{np.mean(incorrect_confidences):.3f}")

        stats = {
            "Mean": np.mean(confidences),
            "Median": np.median(confidences),
            "STD": np.std(confidences),
            "Min": np.min(confidences),
            "Max": np.max(confidences),
            "Q25": np.percentile(confidences, 25),
            "Q75": np.percentile(confidences, 75),
            "Q90": np.percentile(confidences, 90),
            "Q95": np.percentile(confidences, 95),
            "Q99": np.percentile(confidences, 99),
            "Mean (correct)": mean_conf_correct,
            "Mean (incorrect)": mean_conf_incorrect
        }

        tb = build_table(data=stats, columns=["Statistic index", "Value"])
        rich_panel(
            tables=tb,
            panel_title="Confidence statistics in predictions",
            border_style="pale_turquoise1",
        )
        del tb

        # Calibration check: High confidence should mean high accuracy
        high_conf = [r for r in self.results if r["confidence"] > 0.9]
        if high_conf:
            high_conf_accuracy = sum(r["correct_binary"] for r in high_conf) / len(
                high_conf
            )
            logger.info(
                f"  Accuracy when confidence > 0.9: {high_conf_accuracy:.2%} "
                f"({len(high_conf)} samples)"
            )

    def _save_parse_failures(self) -> None:
        """Save parse failures to JSON for debugging."""

        if self.n_parse_failures > 0:
            failures_file = self.output_dir / "errors" / "parse_failures.json"
            with open(file=failures_file, mode="w", encoding="utf-8") as f:
                json.dump(self.parse_failures, f, indent=2, default=str)

            logger.info(f"Parse failures saved to {failures_file}")

    def _compute_classification_report(self, y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
        """
        Compute classification report with all metrics.

        Parameters
        ----------
        y_true: list[str]
            Ground truth labels
        y_pred: list[str]
            Predicted labels

        Returns
        -------
        dict[str, Any]
            Classification report containing accuracy, precision, recall, f1 
            for each class, plus macro/weighted averages.
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=["Vulnerable", "Safe"],
            output_dict=True,
            zero_division=0.0, # type: ignore
        )

        return report # type: ignore

    def _compute_confusion_matrix(self, y_true: list[str], y_pred: list[str]) -> np.ndarray:
        """
        Compute confusion matrix.

        Parameters
        ----------
        y_true: list[str]
            Ground truth labels
        y_pred: list[str]
            Predicted labels

        Returns
        -------
        np.ndarray
            Confusion matrix with shape (2, 2).
            [[TP, FN],
             [FP, TN]]
        """

        return confusion_matrix(y_true, y_pred, labels=["Vulnerable", "Safe"])

    def _compute_binary_metrics(self) -> dict[str, Any]:
        """
        Compute binary classification metrics (accuracy, precision, recall, F1).

        Returns:
            Dictionary containing classification report and confusion matrix
        """
        if not self.results:
            logger.exception("No valid predictions found. Run _parse_predictions() first.")
            raise ValueError("No results available for metrics computation")

        with rich_status(
            description="Computing binary classification metrics...", spinner="arc"
        ):
            y_true: list[bool] = [r["gt_vulnerable"] for r in self.results]
            y_pred: list[bool] = [r["pred_vulnerable"] for r in self.results]

            y_true_labels = ["Vulnerable" if y else "Safe" for y in y_true]
            y_pred_labels = ["Vulnerable" if y else "Safe" for y in y_pred]

            report = self._compute_classification_report(
                y_true=y_true_labels, y_pred=y_pred_labels
            )
            cm = self._compute_confusion_matrix(
                y_true=y_true_labels, y_pred=y_pred_labels
            )

            tp = sum(r["tp"] for r in self.results)
            tn = sum(r["tn"] for r in self.results)
            fp = sum(r["fp"] for r in self.results)
            fn = sum(r["fn"] for r in self.results)

            # sanity check: sklearn confusion matrix should match tracking
            assert cm[0, 0] == tp, "TP mismatch between sklearn and tracked values"
            assert cm[0, 1] == fn, "FN mismatch between sklearn and tracked values"
            assert cm[1, 0] == fp, "FP mismatch between sklearn and tracked values"
            assert cm[1, 1] == tn, "TN mismatch between sklearn and tracked values"

        binary_metrics = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_dict": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
            "n_samples": len(self.results),
            "n_vulnerable": sum(y_true),
            "n_safe": len(y_true) - sum(y_true),
        }

        self._log_binary_summary(binary_metrics=binary_metrics)

        return binary_metrics

    def _save_binary_metrics(self, binary_metrics: dict[str, Any]) -> None:
        """Save binary classification metrics to JSON file."""

        metrics_file = self.output_dir / "binary_metrics.json"

        with open(file=metrics_file, mode="w", encoding="utf-8") as f:
            json.dump(binary_metrics, f, indent=2)

        logger.info(f"Binary metrics saved to {metrics_file}")

    def _log_binary_summary(self, binary_metrics: dict[str, Any]) -> None:
        """Log evaluation summary to console."""

        yes_metrics = binary_metrics["report"]["Vulnerable"]
        no_metrics = binary_metrics["report"]["Safe"]
        tp, tn, fp, fn = binary_metrics["confusion_matrix_dict"].values()

        overall_stats = {
            "Support (total)": binary_metrics["n_samples"],
            "Support (vulnerable)": binary_metrics["n_vulnerable"],
            "Support (safe)": binary_metrics["n_safe"],
            "Accuracy": f"{binary_metrics["accuracy"]:3.f}",
            "Confusion Matrix": f"TP={tp}, TN={tn}, FP={fp}, FN={fn}",
            "Overall Accuracy": f"{binary_metrics['accuracy']:.4f}"
        }

        tb = build_table(data=overall_stats, columns=["Index name", "Value"])
        overall_panel = build_panel(
            tb,
            panel_title="📊 Binary classification results",
            border_style="slate_blue1",
        )

        vul_stats = {
            "Support (vulnerable)": binary_metrics["n_vulnerable"],
            "Precision (vulnerable)": float(f"{yes_metrics["precision"]:3.f}"),
            "Recall (vulnerable)": float(f"{yes_metrics["recall"]:3.f}"),
            "F1-Score (vulnerable)": float(f"{yes_metrics["f1-score"]:3.f}"),
        }
        tb = build_table(data=vul_stats, columns=["Index name", "Value"])
        vul_panel = build_panel(
            tb,
            panel_title="Vulnearble class results",
            border_style="red1",
        )

        safe_stats = {
            "Support (safe)": binary_metrics["n_safe"],
            "Precision (safe)": f"{no_metrics["precision"]:3.f}",
            "Recall (safe)": f"{no_metrics["recall"]:3.f}",
            "F1-Score (safe)": f"{no_metrics["f1-score"]:3.f}",
        }
        tb = build_table(data=safe_stats, columns=["Index name", "Value"])
        safe_panel = build_panel(
            tb,
            panel_title="Safe class results",
            border_style="chartreuse1",
        )

        rich_panels_grid(
            panels=[overall_panel, vul_panel, safe_panel], grid_shape=(1, 3)
        )

        l: list[tuple[str, dict]] = [
            ("overall_stats", overall_stats),
            ("vul_stats", vul_stats),
            ("safe_stats", safe_stats),
        ]
        stats: dict[str,dict] = {k: v for k, v in l}
        self._save_json(filepath=(self.output_dir / "stats.json"), obj=stats)

    def _save_binary_artifacts(self, binary_metrics: dict[str, Any]) -> None:
        """Save all binary classification artifacts to disk."""

        with rich_status(description="Saving Artifacts ...", spinner="arc"):
            self._save_binary_metrics(binary_metrics=binary_metrics)
            # classification report as a separate file from `binary_metrics.json`
            report_path = self.output_dir / "binary_classification_report.json"
            self._save_json(report_path, binary_metrics["classification_report"])
            logger.info(f"✅ Classification report: {report_path.name}")

            cm_path = self.output_dir / "binary_confusion_matrix.png"
            self._plot_confusion_matrix(
                cm=binary_metrics["confusion_matrix"],
                title="Binary Vulnerability Detection",
                save_path=cm_path,
            )

    def evaluate_cwe_classification(self, save_artifacts: bool = True):

        logger.info("Starting CWE classifiaction evaluation ...")

        cwe_pairs = self._collect_cwe_pairs()
        cwe_vocabulary: list[int] = self._build_cwe_vocabulary(cwe_pairs=cwe_pairs)
        y_true_bin, y_pred_bin = self._binarize_cwe_labels(
            cwe_pairs=cwe_pairs, vocabulary=cwe_vocabulary
        )
        per_cwe_report = self._compute_per_cwe_metrics(
            y_true_bin, y_pred_bin, vocabulary=cwe_vocabulary  # type: ignore
        )
        micro_f1, macro_f1 = self._compute_aggregate_metrics(y_true_bin, y_pred_bin)  # type: ignore

        results = CWEEvaluationResults(
            per_cwe_metrics=per_cwe_report,
            aggreate_metrics={"micro_f1": micro_f1, "macro_f1": macro_f1},
            vocabulary=cwe_vocabulary,
            n_samples=len(cwe_pairs),
            n_classes=len(cwe_vocabulary),
        )

        if save_artifacts:
            self._save_cwe_artifacts(results)

    def _collect_cwe_pairs(self) -> list[CWEPair]:
        """Collect pairs of CWE indexes in a list of tuples where:
            - 1st element represents all ground truth cwe ids
            - 2nd element represents corresponding predicted cwe ids
        """
        return [
            CWEPair(cwes_gt=entry["gt_cwes"], cwes_pred=entry["pred_cwes"])
            for entry in self.results
            if (entry["gt_vulnerable"] and entry["pred_vulnerable"])
        ]

    def _build_cwe_vocabulary(self, cwe_pairs: list[CWEPair]) -> list[int]:
        """Build complete CWE vocabulary from ground truth and predictions.

        Returns
        -------
        list[str]
            Sorted list of unique CWE identifiers.
        """

        all_cwes: set[int] = set()
        for cwe_pair in cwe_pairs:
            all_cwes.update(cwe_pair.as_flatten_tuple)

        sorted_cwes = sorted(all_cwes)

        logger.info(f"CWE vocabulary: {len(sorted_cwes)} unique CWEs")
        logger.debug(f"CWEs: {', '.join(map(str,sorted_cwes))}")

        return sorted_cwes

    def _binarize_cwe_labels(self, cwe_pairs: list[CWEPair], vocabulary: list[int]):
        """Binarize multi-label CWE lists.

        Parameters
        ----------
        cwe_pairs : list[CWEPair]
            List of ground truth and predicted CWE pairs
        all_cwes : list[int]
            Complete vocabulary of CWE IDs

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (y_true_binarized, y_pred_binarized)
            Shape: (n_samples, n_cwes)
        """

        # must be iterable of iterables
        ground_truth_cwes: list[list[int]] = [pair.cwes_gt for pair in cwe_pairs]
        predicted_cwes: list[list[int]] = [pair.cwes_pred for pair in cwe_pairs]

        mlb = MultiLabelBinarizer(classes=vocabulary)
        y_true_bin = mlb.fit_transform(ground_truth_cwes)
        y_pred_bin = mlb.transform(predicted_cwes)

        assert (
            list(mlb.classes_) == vocabulary
        ), f"Class order mismatch: {list(mlb.classes_)} != {vocabulary}"

        logger.debug(f"Binarized shape: {y_true_bin.shape}")
        # logger.debug(f"Column order: {list(mlb.classes_[:5])}... (first 5)")

        return y_true_bin, y_pred_bin

    def _compute_per_cwe_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, vocabulary: list[int]
    ) -> dict[str, Any]:
        """Compute precision, recall, F1 for each CWE.

        Returns
        -------
        dict
            Per-CWE metrics from classification_report.
        """

        target_names = [f"CWE-{idx}" for idx in vocabulary]
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            digits=4,
            zero_division=0.0, # type: ignore
        )
        report = cast(dict[str, Any], report)

        logger.info(f"Computed metrics for {len(vocabulary)} CWE classes")

        self._log_top_bottom_performers(report, n=5)

        return report

    def _compute_aggregate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Compute micro and macro averaged F1 scores.

        Returns
        -------
        tuple[float, float]
            (micro_f1, macro_f1)
        """

        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0.0) # type: ignore
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0) # type: ignore

        return micro_f1, macro_f1

    def _log_top_bottom_performers(self, report: dict[str, Any], n: int = 5):
        """Log top and bottom performing CWEs."""

        cwe_metrics = {
            k: v
            for k, v in report.items()
            if k.startswith("CWE-") and isinstance(v, dict)
        }

        sorted_cwes = sorted(
            cwe_metrics.items(), key=lambda x: x[1].get("f1-score", 0), reverse=True
        )

        logger.info(f"\nTop {n} performing CWEs:")
        for cwe, metrics in sorted_cwes[:n]:
            logger.info(
                f"  {cwe}: F1={metrics['f1-score']:.4f}, "
                f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                f"Support={metrics['support']}"
            )

        # Bottom performers (with support > 0)
        bottom = [(cwe, m) for cwe, m in sorted_cwes if m["support"] > 0][-n:]

        if bottom:
            logger.info(f"\nBottom {n} performing CWEs (with support > 0):")
            for cwe, metrics in bottom:
                logger.info(
                    f"  {cwe}: F1={metrics['f1-score']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                    f"Support={metrics['support']}"
                )

    def _save_cwe_artifacts(self, results: CWEEvaluationResults) -> None:
        """Save CWE evaluation artifacts."""

        logger.info(f"--- Saving CWE Artifacts ---")

        report_path = self.output_dir / "cwe_classification_report.json"
        full_report = {
            "micro_avg_f1": results.aggreate_metrics["micro_avg_f1"],
            "macro_avg_f1": results.aggreate_metrics["macro_avg_f1"],
            "per_cwe_metrics": results.per_cwe_metrics,
        }
        self._save_json(report_path, full_report)
        logger.info(f"✅ CWE report: {report_path.name}")

        self._plot_per_cwe_performance(
            per_cwe_report=results.per_cwe_metrics, vocab=results.vocabulary
        )

    def analyze_misclassifications(
        self,
        predictions: list[ParsedResponse],
        save_artifacts: bool = True,
        include_code: bool = False,
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

        logger.info("MISCLASSIFICATION ANALYSIS")

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
        predictions: list[ParsedResponse],
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

        for i, parsed_resp in enumerate(predictions):
            sample = self.test_dataset[i]
            gt_label = self.LABEL_YES if sample["target"] == 1 else self.LABEL_NO

            pred_label = 1 if  parsed_resp.verdict.get("cwe_list") else 0

            # skip if unparseable and correct
            if not parsed_resp.parse_error or pred_label is None: continue
            if pred_label == gt_label: continue

            sample_details = {
                "index": i,
                "ground_truth_label": gt_label,
                "predicted_label": pred_label,
                "ground_truth_cwes": sample.get("cwe", []),
                "predicted_cwes": parsed_resp.verdict.get("cwe_list"),
                "model_response": parsed_resp.__repr__(),
                "response_full_length": len(parsed_resp.__str__()), # approx
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
        except Exception:
            logger.exception(f"Failed to save JSON to {filepath}")
            raise

    def _plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str,
        save_path: Path,
        labels: Optional[list[str]] = None,
    ) -> None:
        """Generate and save confusion matrix heatmap."""

        labels = labels if labels is not None else ["Vulnerable", "Safe"]
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()

        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        except Exception:
            logger.exception(f"Failed to save confusion matrix")
        finally:
            plt.close()

        logger.info(f"Confusion matrix plot saved to {save_path}")

    def _plot_per_cwe_performance(
        self, per_cwe_report: dict[str, Any], vocab: list[int]
    ) -> None:
        """Create visualizations for per-CWE performance.

        Generates:
        1. Bar chart: F1-scores per CWE (sorted by F1)
        2. Heatmap: Precision/Recall/F1 per CWE
        3. Support distribution: Sample counts per CWE
        4. Radar chart: Top performing CWEs

        Parameters
        ----------
        per_cwe_report : dict[str, Any]
            Classification report with per-CWE metrics.
            Keys are "CWE-{id}" strings.
        vocab : list[int]
            List of CWE IDs (integers).
        """

        logger.info("Generating CWE performance visualizations...")

        cwe_metrics = []
        for cwe_id in vocab:
            cwe_key = f"CWE-{cwe_id}"
            if cwe_key in per_cwe_report:
                metrics = per_cwe_report[cwe_key]
                cwe_metrics.append({
                    "cwe": cwe_id,
                    "cwe_label": cwe_key,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1-score": metrics["f1-score"],
                    "support": metrics["support"],
                })

        if not cwe_metrics:
            logger.warning("No CWE metrics to plot")
            return

        try:
            self._plot_f1_bar_chart(cwe_metrics)
            self._plot_metrics_heatmap(cwe_metrics)
            self._plot_support_distribution(cwe_metrics)
            logger.info(f"✅ Generated {4} CWE visualization plots")
        except Exception:
            logger.exception("Failed to generate CWE visualization")

    def _plot_f1_bar_chart(
        self,
        cwe_metrics: list[dict[str, Any]],
        max_to_display: Optional[int] = None,
    ) -> None:
        """Bar chart showing F1-score for each CWE, sorted by performance.

        Parameters
        ----------
        cwe_metrics : list[dict[str, Any]]
            List of CWE metrics dictionaries
        """

        sorted_metrics = sorted(cwe_metrics, key=lambda x: x["f1-score"], reverse=True)
        if max_to_display and len(sorted_metrics) > max_to_display:
            sorted_metrics = sorted_metrics[:max_to_display]
            title_suffix = f" (Top {max_to_display})"
        else:
            title_suffix = ""

        df = pd.DataFrame(data=sorted_metrics)

        fig, ax = plt.subplots(figsize=(12, min(max(6, len(sorted_metrics) * 0.4), 20)))
        sns.barplot(
            data=df,
            y="cwe_label",
            x="f1-score",
            palette="RdYlGn",
            edgecolot="black",
            linewidth=0.5,
            ax=ax
        )

        for i, (_, row) in enumerate(df.iterrows()):
            f1 = row["f1-score"]
            support = row["support"]

            if f1 > 0.15:
                x_pos = f1 - 0.02
                ha, color, weight = "right", "white", "bold"
            else:
                x_pos = f1 + 0.02
                ha, color, weight = "left", "black", "normal"

            ax.text(
                x_pos,
                i,
                f"{f1:.3f} (n={support:,})",
                ha=ha,
                va="center",
                fontsize=9,
                color=color,
                fontweight=weight,
            )

        # Styling
        ax.set_xlabel("F1-Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("CWE", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Per-CWE F1-Score Performance {title_suffix}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlim(0, 1.05)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5,
            alpha=0.5, label="0.5 threshold",
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.legend(loc="lower right")

        plt.tight_layout()

        save_path = self.output_dir / "cwe_f1_scores.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ F1 bar chart: {save_path.name}")
        plt.close(fig)

    def _plot_metrics_heatmap(self, cwe_metrics: list[dict[str, Any]], max_to_display: Optional[int]=30) -> None:
        """Heatmap showing Precision, Recall, and F1-score for each CWE."""

        sorted_metrics = sorted(cwe_metrics, key=lambda x: x["f1-score"], reverse=True)
        if max_to_display and len(sorted_metrics) > max_to_display:
            sorted_metrics = sorted_metrics[:max_to_display]

        cwes = [f"{m['cwe_label']} (n={m['support']:,})" for m in sorted_metrics]

        data = {
            "Precision": [m["precision"] for m in cwe_metrics],
            "Recall": [m["recall"] for m in cwe_metrics],
            "F1-Score": [m["f1-score"] for m in cwe_metrics],
        }

        df = pd.DataFrame(data, index=cwes) # type: ignore

        fig_height = min(max(6, len(cwes) * 0.5), 20)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        annot_fontsize = max(6, min(10, 100 / len(cwes)))

        sns.heatmap(
            df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score", "shrink": 0.8},
            linewidths=0.5,
            linecolor="gray",
            ax=ax,
            annot_kws={"fontsize": annot_fontsize},
        )

        # Styling
        ax.set_title(
            "Per-CWE Metrics Heatmap (Sorted by F1-Score)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
        ax.set_ylabel("CWE", fontsize=12, fontweight="bold")

        plt.xticks(rotation=0, fontsize=11)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        save_path = self.output_dir / "cwe_metrics_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Metrics heatmap: {save_path.name}")
        plt.close(fig)

    def _plot_metrics_clustermap(
        self, cwe_metrics: list[dict[str, Any]], max_to_display: Optional[int] = 30
    ) -> None:
        """Heatmap with clustering to show performance patterns."""

        sorted_metrics = sorted(cwe_metrics, key=lambda x: x["f1-score"], reverse=True)
        if max_to_display and len(sorted_metrics) > max_to_display:
            sorted_metrics = sorted_metrics[:max_to_display]

        cwes = [m["cwe_label"] for m in sorted_metrics]

        data = {
            "Precision": [m["precision"] for m in sorted_metrics],
            "Recall": [m["recall"] for m in sorted_metrics],
            "F1-Score": [m["f1-score"] for m in sorted_metrics],
            "Support": [
                m["support"] / max(m["support"] for m in sorted_metrics)
                for m in sorted_metrics
            ],
        }

        df = pd.DataFrame(data, index=cwes) # type: ignore

        fig_height = min(max(8, len(cwes) * 0.5), 20)

        g = sns.clustermap(
            df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            figsize=(12, fig_height),
            cbar_kws={"label": "Score"},
            linewidths=0.5,
            linecolor="gray",
            dendrogram_ratio=0.1,
            row_cluster=True,
            col_cluster=False,
        )

        g.figure.suptitle(
            "Per-CWE Metrics Heatmap (Clustered by Performance)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        save_path = self.output_dir / "cwe_metrics_heatmap_clustered.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Clustered heatmap: {save_path.name}")
        plt.close()

    def _plot_support_distribution(
        self, cwe_metrics: list[dict[str, Any]], max_to_display: Optional[int] = 30
    ) -> None:
        """Bar chart showing sample count (support) for each CWE. Helps identify class imbalance."""

        sorted_metrics = sorted(cwe_metrics, key=lambda x: x["support"], reverse=True)
        if max_to_display and len(sorted_metrics) > max_to_display:
            sorted_metrics = sorted_metrics[:max_to_display]

        df = pd.DataFrame(sorted_metrics)

        fig_height = min(max(6, len(sorted_metrics) * 0.4), 20)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        sns.barplot(
            data=df,
            y="cwe_label",
            x="support",
            palette="RdYlGn",
            hue="f1-score",
            dodge=False,
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            legend=False,
        )

        for i, (_, row) in enumerate(df.iterrows()):
            support = row["support"]
            ax.text(
                support + df["support"].max() * 0.01,
                i,
                f"{support:,}",
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
        ax.invert_xaxis()

        # Add colorbar
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("F1-Score", fontsize=10, fontweight="bold")

        plt.tight_layout()

        save_path = self.output_dir / "cwe_support_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Support distribution: {save_path.name}")
        plt.close(fig)

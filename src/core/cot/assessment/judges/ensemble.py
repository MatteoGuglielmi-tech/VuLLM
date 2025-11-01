import gc
import json
import jsonlines
import logging
import numpy as np

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from scipy.spatial.distance import pdist

from .judge import LLMJudge
from .judge_types import JudgeConfig
from ..datatypes import EvaluationResult, ReasoningSample
from ..utilities import iter_jsonl_samples, rich_table, rich_rule, rich_panel, build_table, progress_bar


logger = logging.getLogger(__name__)


class JudgeEnsemble:
    """Ensemble of LLM judges with equal-weighted voting"""

    def __init__(self, judge_configs: list[JudgeConfig], criterion_weights: dict[str, float]|None=None):
        self.judge_configs = judge_configs
        self.judges: list[LLMJudge] = [LLMJudge(config) for config in judge_configs]
        self.weights = np.ones(len(judge_configs)) / len(judge_configs) # equal weights for all judges

        if criterion_weights is not None:
            self.criterion_weights = criterion_weights
        else: #default
            self.criterion_weights = {
                "correctness": 0.35,
                "completeness": 0.25,
                "technical_accuracy": 0.20,
                "clarity": 0.10,
                "logical_flow": 0.10,
            }

        self.criterion_weights_vector = np.array(
            [self.criterion_weights[c] for c in EvaluationResult.get_criteria_names()]
        )

        data = {
            f"{config.model_name}": [
                f"{idx+1}",
                f"{config.specialization}",
                f"{self.weights[idx]:.3f}",
                None if config.description is None else f"{config.description}",
            ]
            for idx, config in enumerate(judge_configs)
        }
        table = build_table(
            data=data,
            title=f"Initialized ensemble with {len(judge_configs)} judges",
            columns=["Judge", "Position", "Specialization", "Weight", "Description"],
        )
        rich_table(data=table)
        rich_rule()
        tables = [
            build_table(
                jc.__dict__,
                columns=["Parameter", "Value"],
                title=""
            )
            for jc in judge_configs
        ]

        rich_panel(
            tables,
            panel_title="Judges configurations",
            subtitle="Judges correctly initialized",
            border_style="green",
            padding=(1, 20),
        )

        table = build_table(
            data=self.criterion_weights,
            title=f"Weights for computing overall quality from criteria",
            columns=["Criterion", "Weight"],
            expand=True
        )
        rich_table(data=table)
        rich_rule()
        del tables, table, data
        gc.collect()

    def get_ensemble_info(self) -> dict:
        """Get information about the ensemble composition."""
        return {
            "num_judges": len(self.judge_configs),
            "judges": [
                {
                    "model": config.model_name,
                    "specialization": config.specialization,
                    "weight": float(self.weights[idx]),
                    "description": config.description,
                }
                for idx, config in enumerate(self.judge_configs)
            ],
            "weighting_strategy": "equal",
        }

    def evaluate_sample(self, sample: ReasoningSample, method: str ="weighted_multidimensional") -> dict:
        """
        Evaluate sample with all judges and compute ensemble metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - sample_id: Sample identifier
            - evaluations: List of EvaluationResult dicts from each judge
            - ensemble_score: Final weighted quality score
            - ensemble_criteria: Average score for each criterion
            - agreement: Agreement score between judges
        """

        evaluations: list[EvaluationResult] = []
        for _, judge in enumerate(self.judges):
            judge.load()
            eval_result = judge.evaluate(sample)
            evaluations.append(eval_result)
            judge.unload()

        ensemble_score = self._compute_weighted_score(evaluations)
        agreement = self._compute_agreement(evaluations, method=method)

        ensemble_criteria = {
            "correctness": float(np.mean([e.correctness for e in evaluations])),
            "completeness": float(np.mean([e.completeness for e in evaluations])),
            "clarity": float(np.mean([e.clarity for e in evaluations])),
            "technical_accuracy": float(np.mean([e.technical_accuracy for e in evaluations])),
            "logical_flow": float(np.mean([e.logical_flow for e in evaluations])),
        }

        return {
            "sample_id": sample.sample_id,
            "evaluations": [e.to_dict() for e in evaluations],
            "ensemble_score": ensemble_score,
            "ensemble_criteria": ensemble_criteria,
            "agreement": agreement,
            "agreement_method": "weighted_multidimensional",
        }

    def _compute_weighted_score(self, evaluations: list[EvaluationResult]) -> float:
        """
        Compute weighted ensemble score using ALL criteria.

        Parameters
        ----------
        evaluations : List[EvaluationResult]
            List of evaluation results from all judges

        Returns
        -------
        float
            Weighted ensemble quality score [0, 1]
        """

        criterion_vectors = np.array([eval_result.get_criteria_vector() for eval_result in evaluations])
        confidences = np.array([eval_result.confidence for eval_result in evaluations])

        # Weight judges by their confidence
        judge_weights = self.weights * confidences
        judge_weights = judge_weights / judge_weights.sum()

        # Compute ensemble average for each criterion
        ensemble_criteria = np.dot(judge_weights, criterion_vectors)  # Shape: (n_criteria,)

        # Final weighted score
        final_score = np.dot(ensemble_criteria, self.criterion_weights_vector)

        return float(final_score)

    def _compute_agreement(self, evaluations: list[EvaluationResult], method: str) -> float:
        """
        Compute agreement between judges using criterion vectors.

        Parameters
        ----------
        evaluations : List[EvaluationResult]
            List of evaluation results from all judges
        method : str
            Agreement calculation method:
            - 'multidimensional': All criteria, equal weight
            - 'weighted_multidimensional': All criteria, weighted by importance

        Returns
        -------
        float
            Agreement score [0, 1], where 1 = perfect agreement
        """
        if len(evaluations) < 2:
            return 1.0

        if method == "multidimensional":
            # All criteria, equal weight
            criterion_vectors = np.array([e.get_criteria_vector() for e in evaluations])

            distances = pdist(criterion_vectors, metric="euclidean")

            # Normalize by max possible distance in 5D unit hypercube
            max_distance = np.sqrt(5)
            mean_distance = np.mean(distances)
            agreement = 1.0 - (mean_distance / max_distance)

            return float(np.clip(agreement, 0, 1))

        elif method == "weighted_multidimensional":
            # All criteria, weighted by importance
            criterion_vectors = np.array([e.get_criteria_vector() for e in evaluations])

            # Weight each dimension
            weighted_vectors = criterion_vectors * self.criterion_weights_vector[np.newaxis, :]

            distances = pdist(weighted_vectors, metric="euclidean")

            # max distance --> opposite corners in WEIGHTED space
            # point_1 = [0, 0, 0, 0, 0] * weights = [0, 0, 0, 0, 0]
            # point_2 = [1, 1, 1, 1, 1] * weights = [w₁, w₂, w₃, w₄, w₅]
            max_weighted_distance = np.linalg.norm(self.criterion_weights_vector)

            mean_distance = np.mean(distances)
            agreement = 1.0 - (mean_distance / max_weighted_distance)

            return float(np.clip(agreement, 0, 1))

        else:
            raise ValueError(f"Unknown agreement method: {method}")

    def aggregate_judge_evaluations(
        self, all_records: list[dict]
    ) -> dict[str, dict[str, float]]:
        """
        Aggregate per-judge criteria scores across all samples using median.

        Parameters
        ----------
        all_records : list[dict]
            List of all output records containing 'per_judge_evaluations'

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict: {judge_name: {criterion: median_score}}
            Example: {'Qwen-Coder': {'correctness': 0.82, 'completeness': 0.78, ...}}
        """

        # Collect all scores per judge per criterion
        # Structure: {judge_name: {criterion: [score1, score2, ...]}}
        judge_scores = defaultdict(lambda: defaultdict(list))

        criteria_names = EvaluationResult.get_criteria_names()
        criteria_names.append("quality_score")

        with progress_bar(
            all_records,
            description="🏗️ Collect all scores per judge per criterion 🏗️",
        ) as records:
            for record in records:
                for evaluation in record["per_judge_evaluations"]:
                    judge_name = evaluation["judge_name"]
                    for criterion in criteria_names:
                        if criterion in evaluation:
                            judge_scores[judge_name][criterion].append(
                                evaluation[criterion]
                            )

        judge_evaluations = {}
        with progress_bar(
            judge_scores, description="🧮 Computing median scores 🧮"
        ) as scores:
            for judge_name, criteria_dict in scores.items():
                judge_evaluations[judge_name] = {
                    criterion.replace("_", " ").title(): np.median(scores)
                    for criterion, scores in criteria_dict.items()
                }

        return judge_evaluations

    def filter_dataset_streaming(
        self,
        input_jsonl_path: Path,
        output_jsonl_path: Path,
        rejected_jsonl_path: Path,
        stats_json_path: Path,
        quality_threshold: float = 0.6,
        agreement_threshold: float = 0.7,
        save_interval: int = 10,
        agreement_method: str = "weighted_multidimensional"
    ) -> dict:
        """
        Filter dataset progressively with sequential judge loading.

        Parameters
        ----------
        input_jsonl_path : str
            Path to input JSONL file
        output_jsonl_path : str
            Path to save filtered (kept) samples
        rejected_jsonl_path : str
            Path to save rejected samples
        quality_threshold : float
            Minimum ensemble quality score (0-1)
        agreement_threshold : float
            Minimum judge agreement score (0-1)
        save_interval : int
            Flush to disk every N samples
        agreement_method : str ["multidimensional", "weighted_multidimensional"]
            Method to use to compute agreement
        hybrid_weight: float, (optional, default=0.75)
            How much to weight the range component in agreement computation if hybrid is
            selected as method

        Returns
        -------
        dict : Filtering statistics
        """

        logger.info("🚀 Starting dataset filtering...🚀 ")

        with open(file=input_jsonl_path, mode="r") as f:
            total_lines = sum(1 for _ in f)

        stats = {
            "total": 0,
            "kept": 0,
            "rejected": 0,
            "low_quality": 0,
            "low_agreement": 0,
            "both_issues": 0,
            "scores": [],
            "agreements": [],
        }

        kept_batch = []
        rejected_batch = []
        all_records = []
        with (
            jsonlines.open(output_jsonl_path, mode="w") as kept_writer,
            jsonlines.open(rejected_jsonl_path, mode="w") as rejected_writer,
            tqdm(total=total_lines, desc="🧪 Filtering samples", unit="sample") as pbar,
        ):

            for sample in iter_jsonl_samples(input_jsonl_path):
                stats["total"] += 1

                eval_result = self.evaluate_sample(sample, method=agreement_method)
                score: float = eval_result["ensemble_score"] # [0 - 1]
                agreement = eval_result["agreement"]

                stats["scores"].append(score)
                stats["agreements"].append(agreement)

                output_record = {
                    "project": sample.project,
                    "cwe": sample.cwe,
                    "target": sample.target,
                    "func": sample.func,
                    "cwe_desc": sample.cwe_desc,
                    "reasoning": sample.reasoning,
                    "quality_score": score,
                    "agreement": agreement,
                    "per_judge_evaluations": eval_result["evaluations"],
                }

                passes_quality: bool = score >= quality_threshold
                passes_agreement: bool = agreement >= agreement_threshold

                if passes_quality and passes_agreement:
                    stats["kept"] += 1
                    kept_batch.append(output_record)

                    if stats["kept"] % save_interval == 0: # save and flush
                        kept_writer.write_all(kept_batch)
                        all_records.extend(kept_batch)
                        kept_batch = []
                else:
                    stats["rejected"] += 1

                    rejection_reasons = []
                    if not passes_quality:
                        stats["low_quality"] += 1
                        rejection_reasons.append("low_quality")
                    if not passes_agreement:
                        stats["low_agreement"] += 1
                        rejection_reasons.append("low_agreement")
                    if not passes_quality and not passes_agreement:
                        stats["both_issues"] += 1

                    output_record["rejection_reason"] = rejection_reasons
                    rejected_batch.append(output_record)

                    # Flush batch
                    if len(rejected_batch) >= save_interval:
                        all_records.extend(rejected_batch)
                        rejected_writer.write_all(rejected_batch)
                        rejected_batch = []

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix( {
                    "kept": stats["kept"],
                    "rejected": stats["rejected"],
                    "keep_rate": (
                        f"{stats['kept']/stats['total']:.1%}"
                        if stats["total"] > 0
                        else "0%"
                    ),
                })

        # important: Flush remaining samples after loop ends
        if kept_batch:
            kept_writer.write_all(kept_batch)
            logger.info(f"Flushed final {len(kept_batch)} kept samples")

        if rejected_batch:
            rejected_writer.write_all(rejected_batch)
            logger.info(f"Flushed final {len(rejected_batch)} rejected samples")

        # Compute final statistics
        stats["keep_rate"] = stats["kept"] / stats["total"] if stats["total"] > 0 else 0
        stats["reject_rate"] = stats["rejected"] / stats["total"] if stats["total"] > 0 else 0
        stats["mean_score"] = float(np.mean(stats["scores"]))
        stats["std_score"] = float(np.std(stats["scores"]))
        stats["mean_agreement"] = float(np.mean(stats["agreements"]))
        stats["std_agreement"] = float(np.std(stats["agreements"]))
        stats["per_judge_metrics"] = self.aggregate_judge_evaluations(all_records=all_records)

        data = {
            "Total samples processed": [stats["total"], "/"],
            "Kept": [stats["kept"], f"{ stats["keep_rate"]:.2% }"],
            "Rejected": [stats["rejected"], f"{ stats["reject_rate"] :.2% }"],
            "Mean quality score": [f"{stats['mean_score']:.3f} ± {stats['std_score']:.3f}", "/"],
            "Mean quality score": [f"{stats['mean_agreement']:.3f} ± {stats['std_agreement']:.3f}", "/"],
        }
        res_table=build_table(data=data, title="FILTERING RESULTS", columns=["Metric", "Value", "Rate"])
        data = {
            "Total": stats["rejected"],
            "Low quality only": stats["low_quality"] - stats["both_issues"],
            "Low agreement only": stats["low_agreement"] - stats["both_issues"],
            "Both issues": stats["both_issues"],
        }
        rejected_table = build_table(data=data, title="REJECTED ANALYSIS", columns=["Support", "Value"])

        with open(file=stats_json_path, mode="w") as f:
            json.dump(
                {k: v for k, v in stats.items() if k not in ["scores", "agreements"]},
                f,
                indent=2,
            )

        rich_panel(
            tables=[res_table, rejected_table],
            panel_title="🏁 Experiment outcome 🏁",
            subtitle=f"\n✓ Statistics saved to: {stats_json_path}",
            border_style="dim purple",
            padding=(1, 35),
        )

        del data, res_table, rejected_table
        return stats

import gc
import json
import jsonlines
import logging
import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import Any

from .judge import LLMJudge
from .judge_types import JudgeConfig
from ..types import ReasoningSample
from ..utilities import iter_jsonl_samples, rich_table, rich_rule, rich_panel, build_table


logger = logging.getLogger(__name__)


class JudgeEnsemble:
    """Ensemble of LLM judges with equal-weighted voting"""

    def __init__(self, judge_configs: list[JudgeConfig]):
        self.judge_configs = judge_configs
        self.judges: list[LLMJudge] = [LLMJudge(config) for config in judge_configs]
        self.weights = np.ones(len(judge_configs)) / len(judge_configs) # equal weights for all judges
        self._scores_cache = None

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
        del tables, table, data
        gc.collect()

        exit()
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

    def evaluate_sample(
        self,
        sample: ReasoningSample,
        method: str = "range",
        hybrid_weight: float = 0.75,
    ) -> dict:
        """Get evaluations from all judges for a single sample.
        Loads/unloads each judge sequentially.
        """

        evaluations = []
        for _, judge in enumerate(self.judges):
            judge.load()
            eval_result = judge.evaluate(sample)
            evaluations.append(eval_result)
            judge.unload()

        return {
            "sample_id": sample.sample_id,
            "evaluations": evaluations,
            "ensemble_score": self._compute_weighted_score(evaluations),
            "agreement": self._compute_agreement(evaluations, method=method, hybrid_weight=hybrid_weight),
        }

    def _compute_weighted_score(self, evaluations: list[dict[str, Any]]) -> float:
        """Compute weighted ensemble score"""

        if self._scores_cache is None:
            self._scores_cache = [e["quality_score"] for e in evaluations]
        confidences = np.array([e.get("confidence", 1.0) for e in evaluations])

        # Combine equal weights with confidence
        combined_weights = self.weights * confidences
        combined_weights = combined_weights / combined_weights.sum()

        weighted_score = np.dot(np.array(self._scores_cache), combined_weights)

        return float(weighted_score)

    def _compute_agreement(
        self,
        evaluations: list[dict[str, Any]],
        method: str,
        hybrid_weight: float,
    ) -> dict:
        """Compute inter-judge agreement.

        Parameters
        ----------
        evaluations : list[dict]
            Judge evaluations
        method : str
            Agreement method: "range" (intuitive), "std" (sensitive), or "hybrid"
        hybrid_range_weight : float
            Weight for range component in hybrid mode (default 0.75)

        Returns
        -------
        float: Agreement score [0, 1]
        """

        if self._scores_cache is None:
            self._scores_cache = [e["quality_score"] for e in evaluations]

        if len(self._scores_cache) < 2:
            return {"agreement": 1.0, "method": "single_judge"}

        scores = np.array(self._scores_cache)

        match method:
            case "range":
                score_range = np.max(scores) - np.min(scores)
                range_agreement = float(np.clip(1.0 - score_range, 0, 1))
                return {
                    "agreement": range_agreement,
                    "method": method,
                    "range": score_range,
                    "scores": scores.tolist(),
                }
            case "std":
                std = np.std(scores)
                return {
                    "agreement": float(1.0 / (1.0 + std)),
                    "method": method,
                    "std": std,
                    "scores": scores.tolist(),
                }
            case "hybrid":
                score_range = np.max(scores) - np.min(scores)
                range_agreement = 1.0 - score_range  # weighted combination (hybrid)
                std = np.std(scores)
                std_agreement = 1.0 / (1.0 + std)

                agreement = hybrid_weight * range_agreement + (1 - hybrid_weight) * std_agreement

                return {
                    "method": "hybrid",
                    "range_agreement": range_agreement,
                    "std_agreement": std_agreement,
                    "weight_range": hybrid_weight,
                    "weight_std": (1.0 - hybrid_weight),
                    "agreement": float(np.clip(agreement, 0, 1)),
                    "range": score_range,
                    "std": std,
                    "scores": scores.tolist(),
                }
            case _:
                raise Exception

    def filter_dataset_streaming(
        self,
        input_jsonl_path: Path,
        output_jsonl_path: Path,
        rejected_jsonl_path: Path,
        stats_json_path: Path,
        quality_threshold: float = 0.6,
        agreement_threshold: float = 0.7,
        save_interval: int = 10,
        agreement_method: str = "hybrid",
        hybrid_weight: float = 0.75,
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
        agreement_method : str ["std", "range", "hybrid"]
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
        with (
            jsonlines.open(output_jsonl_path, mode="w") as kept_writer,
            jsonlines.open(rejected_jsonl_path, mode="w") as rejected_writer,
            tqdm(total=total_lines, desc="🧪 Filtering samples", unit="sample") as pbar,
        ):

            for sample in iter_jsonl_samples(input_jsonl_path):
                stats["total"] += 1
                self._scores_cache = None # flush cache

                eval_result = self.evaluate_sample(sample, method=agreement_method, hybrid_weight=hybrid_weight)
                score: float = eval_result["ensemble_score"] # [0 - 1]
                agreement: dict = eval_result["agreement"]

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
                    "judge_evaluations": eval_result["evaluations"],
                }

                passes_quality: bool = score >= quality_threshold
                passes_agreement: bool = agreement["agreement"] >= agreement_threshold

                if passes_quality and passes_agreement:
                    stats["kept"] += 1
                    kept_batch.append(output_record)

                    if stats["kept"] % save_interval == 0: # save and flush
                        kept_writer.write_all(kept_batch)
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

        data = {
            "Total samples processed": [stats["total"], "/"],
            "Kept": [stats["kept"], f"{ stats["keep_rate"]:.2% }"],
            "Rejected": [stats["rejected"], f"{ stats["reject_rate"] :.2% }"],
            "Mean quality score": [f"{stats['mean_score']:.3f} ± {stats['std_score']:.3f}", "/"],
            "Mean quality score": [f"{stats['mean_agreement']:.3f} ± {stats['std_agreement']:.3f}", "/"],
        }
        rich_table(data=data, title="FILTERING RESULTS", columns=["Metric", "Value", "Rate"])
        data = {
            "Total": stats["rejected"],
            "Low quality only": stats["low_quality"] - stats["both_issues"],
            "Low agreement only": stats["low_agreement"] - stats["both_issues"],
            "Both issues": stats["both_issues"],
        }
        rich_table(data=data, title="REJECTED ANALYSIS", columns=["Support", "Value"])
        del data

        with open(file=stats_json_path, mode="w") as f:
            json.dump(
                {k: v for k, v in stats.items() if k not in ["scores", "agreements"]},
                f,
                indent=2,
            )
        logger.info(f"\n✓ Statistics saved to: {stats_json_path}")
        return stats

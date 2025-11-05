import json
import argparse
import jsonlines
import logging
import numpy as np

from collections import defaultdict
from pathlib import Path

from .judges import JudgeConfig, JudgeEnsemble
from .datatypes import EvaluationResult
from .utilities import (
    iter_jsonl_samples,
    rich_progress,
    rich_progress_manual,
    progress_bar,
    build_table,
    rich_panel,
)

logger = logging.getLogger(__name__)


def aggregate_judge_evaluations(all_records: list[dict]) -> dict[str, dict[str, float]]:
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


def merge_and_filter(
    original_data: Path,
    judge_files: list[Path],
    output_kept: Path,
    output_rejected: Path,
    stats_json_path: Path,
    agreement_method: str = "weighted_multidimensional",
    agreement_threshold: float = 0.75,
    quality_threshold: float = 0.6,
    save_interval: int = 100
):
    """Merge judge evaluations and filter dataset."""

    logger.info("Loading judge evaluations...")

    # load evaluations: {sample_id: {judge_name: EvaluationResult}}
    sample_evals = defaultdict(dict[str, dict[str, EvaluationResult]])

    for judge_file in rich_progress(
        judge_files, total=len(judge_files), description="Loading judge evaluations"
    ):
        with jsonlines.open(judge_file) as reader:
            logger.info(f"Loading {judge_file}")
            for record in reader:
                sample_id = record["sample_id"]
                judge_name = record["judge_name"]  # ref_name
                evaluation = EvaluationResult(**record["evaluation"])
                sample_evals[sample_id][judge_name] = evaluation

    logger.info(f"Loaded evaluations for {len(sample_evals)} samples")

    # Create ensemble
    judge_configs: list[JudgeConfig] = [
        JudgeConfig(
            model_name="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
            ref_name="Qwen2.5-Coder-32B",
            chat_template="qwen-2.5",
            specialization="code",
            description="Specialized in C/C++ vulnerability patterns and code analysis",
        ),
        JudgeConfig(
            model_name="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
            ref_name="Llama-3.1-70B",
            chat_template="llama-3.1",
            specialization="reasoning",
            description="Deep reasoning model for logical flow and completeness",
        ),
        JudgeConfig(
            model_name="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
            chat_template="qwen-2.5",
            ref_name="DeepSeek-R1-Distill-Qwen-32B",
            temperature=0.6,
            top_p=0.95,
            min_p=0.05,
            specialization="logic",
            description="Mathematical and logical reasoning specialist",
        ),
    ]
    ensemble = JudgeEnsemble(judge_configs)

    # Process and filter
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
    judges_refnames = [jc.ref_name for jc in judge_configs]

    with open(file=original_data, mode="r") as f:
        total_lines = sum(1 for _ in f)
    with (
        jsonlines.open(output_kept, "w") as kept_writer,
        jsonlines.open(output_rejected, "w") as rejected_writer,
        rich_progress_manual(total=total_lines, description="🧪 Filtering samples") as pbar,
    ):
        for sample in iter_jsonl_samples(original_data):
            stats["total"] += 1

            evals_dict = sample_evals.get(sample.sample_id, {})
            if len(evals_dict) != 3:
                logger.warning(
                    f"Sample {sample.sample_id} has {len(evals_dict)} evaluations "
                    f"(expected 3), skipping"
                )
                continue

            evaluations: list[EvaluationResult] = [
                evals_dict[judges_refnames[0]],
                evals_dict[judges_refnames[1]],
                evals_dict[judges_refnames[2]],
            ]

            # Compute ensemble metrics
            ensemble_score = ensemble._compute_weighted_score(evaluations)
            agreement = ensemble._compute_agreement(
                evaluations, method=agreement_method
            )

            # Prepare output
            output_record = {
                "project": sample.project,
                "cwe": sample.cwe,
                "target": sample.target,
                "func": sample.func,
                "cwe_desc": sample.cwe_desc,
                "reasoning": sample.reasoning,
                "quality_score": ensemble_score,
                "agreement": agreement,
                "per_judge_evaluations": [e.to_dict() for e in evaluations],
            }

            # Filter
            passes_quality: bool = ensemble_score >= quality_threshold
            passes_agreement: bool = agreement >= agreement_threshold

            if passes_quality and passes_agreement:
                stats["kept"] += 1
                kept_batch.append(output_record)

                if len(kept_batch) >= save_interval:
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

                if len(rejected_batch) >= save_interval:
                    rejected_writer.write_all(rejected_batch)
                    all_records.extend(rejected_batch)
                    rejected_batch = []

            pbar.update(1)
            pbar.set_postfix(
                {
                    "kept": stats["kept"],
                    "rejected": stats["rejected"],
                    "keep_rate": (
                        f"{stats['kept']/stats['total']:.1%}"
                        if stats["total"] > 0
                        else "0%"
                    ),
                }
            )

        # Flush remaining
        if kept_batch:
            kept_writer.write_all(kept_batch)
        if rejected_batch:
            rejected_writer.write_all(rejected_batch)

    # Compute final statistics
    stats["keep_rate"] = stats["kept"] / stats["total"] if stats["total"] > 0 else 0
    stats["reject_rate"] = stats["rejected"] / stats["total"] if stats["total"] > 0 else 0
    stats["mean_score"] = float(np.mean(stats["scores"]))
    stats["std_score"] = float(np.std(stats["scores"]))
    stats["mean_agreement"] = float(np.mean(stats["agreements"]))
    stats["std_agreement"] = float(np.std(stats["agreements"]))
    stats["per_judge_metrics"] = aggregate_judge_evaluations(all_records=all_records)

    data = {
        "Total samples processed": [stats["total"], "/"],
        "Kept": [stats["kept"], f"{ stats["keep_rate"]:.2% }"],
        "Rejected": [stats["rejected"], f"{ stats["reject_rate"] :.2% }"],
        "Mean quality score": [f"{stats['mean_score']:.3f} ± {stats['std_score']:.3f}", "/", ],
        "Mean quality score": [ f"{stats['mean_agreement']:.3f} ± {stats['std_agreement']:.3f}", "/", ],
    }
    res_table = build_table(
        data=data, title="FILTERING RESULTS", columns=["Metric", "Value", "Rate"]
    )
    data = {
        "Total": stats["rejected"],
        "Low quality only": stats["low_quality"] - stats["both_issues"],
        "Low agreement only": stats["low_agreement"] - stats["both_issues"],
        "Both issues": stats["both_issues"],
    }
    rejected_table = build_table(
        data=data, title="REJECTED ANALYSIS", columns=["Support", "Value"]
    )

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

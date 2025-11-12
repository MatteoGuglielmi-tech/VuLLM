import json
import jsonlines
import logging
import numpy as np

from collections import defaultdict
from pathlib import Path

from .judges import JudgeConfig, JudgeEnsemble
from .datatypes import EvaluationResult
from .utilities import (
    iter_jsonl_samples,
    count_jsonl_lines,
    rich_progress,
    rich_progress_manual,
    rich_panel,
    rich_rule,
    build_table,
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
    # Structure: {"judge_name": {"criterion": [score1, score2, ...]}}
    judge_scores = defaultdict(lambda: defaultdict(list))

    criteria_names = EvaluationResult.get_criteria_names()
    criteria_names.append("quality_score")

    for record in rich_progress(
        all_records,
        description="🏗️ Collecting all scores",
        status_fn=lambda _: "Running ...",
    ):
        # for record in records:
        for evaluation in record["per_judge_evaluations"]:
            judge_name = evaluation["judge_name"]
            for criterion in criteria_names:
                if criterion in evaluation:
                    judge_scores[judge_name][criterion].append(evaluation[criterion])

    judge_evaluations = {}
    for judge_name, criteria_dict in rich_progress(
        judge_scores.items(),
        description="📟 Computing median scores",
        status_fn=lambda _: "Running ...",
    ):
        judge_evaluations[judge_name] = {
            criterion.replace("_", " ").title(): np.median(scores)
            for criterion, scores in criteria_dict.items()
        }

    return judge_evaluations


def merge_and_filter(
    original_data: Path,
    judge_files: list[Path],
    judge_configs: list[JudgeConfig],
    output_kept: Path,
    output_rejected: Path,
    output_stats: Path,
    agreement_method: str = "weighted_multidimensional",
    agreement_threshold: float = 0.75,
    quality_threshold: float = 0.6,
    save_interval: int = 100,
):
    """Merge judge evaluations and filter dataset."""

    assert len(judge_files) == len(
        judge_configs
    ), "Mismatch between specified judge files and judge configurations."

    # load evaluations: {sample_id: {judge_name: EvaluationResult}}
    sample_evals = defaultdict(dict[str, dict[str, EvaluationResult]])
    for judge_file in rich_progress(
        judge_files,
        # total=len(judge_files),
        description="⏳ Loading judge evaluations",
        status_fn=lambda f: f"Loading {f.name}",
    ):
        with jsonlines.open(judge_file) as reader:
            for record in reader:
                sample_id = record["sample_id"]
                judge_name = record["judge_name"]  # ref_name
                evaluation = EvaluationResult(**record["evaluation"])
                sample_evals[sample_id][judge_name] = evaluation

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

    with (
        jsonlines.open(file=output_kept, mode="w") as kept_writer,
        jsonlines.open(file=output_rejected, mode="w") as rejected_writer,
        rich_progress_manual(
            total=count_jsonl_lines(original_data), description="🧪 Filtering samples"
        ) as pbar,
    ):
        for sample in iter_jsonl_samples(original_data):
            stats["total"] += 1

            evals_dict = sample_evals.get(sample.sample_id, {})
            if len(evals_dict) != len(judge_files):
                logger.warning(
                    f"Sample {sample.sample_id} has {len(evals_dict)} evaluations "
                    f"(expected {len(judge_files)}), skipping"
                )
                continue

            evaluations: list[EvaluationResult] = [
                evals_dict[judge_name] for judge_name in judges_refnames
            ]

            # Compute ensemble metrics
            ensemble_score = ensemble._compute_weighted_score(evaluations)
            agreement = ensemble._compute_agreement(
                evaluations, method=agreement_method
            )
            stats["scores"].append(ensemble_score)
            stats["agreements"].append(agreement)

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
    stats["reject_rate"] = (
        stats["rejected"] / stats["total"] if stats["total"] > 0 else 0
    )
    stats["mean_score"] = float(np.mean(stats["scores"]))
    stats["std_score"] = float(np.std(stats["scores"]))
    stats["mean_agreement"] = float(np.mean(stats["agreements"]))
    stats["std_agreement"] = float(np.std(stats["agreements"]))
    stats["per_judge_metrics"] = aggregate_judge_evaluations(all_records=all_records)

    data = {
        "Total samples processed": [stats["total"], "/"],
        "Kept": [stats["kept"], f"{stats["keep_rate"]:.2%}"],
        "Rejected": [stats["rejected"], f"{stats["reject_rate"]:.2%}"],
        "Mean quality score": [
            f"{stats['mean_score']:.3f} ± {stats['std_score']:.3f}",
            "/",
        ],
        "Mean quality score": [
            f"{stats['mean_agreement']:.3f} ± {stats['std_agreement']:.3f}",
            "/",
        ],
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

    with open(file=output_stats, mode="w") as f:
        json.dump(
            {k: v for k, v in stats.items() if k not in ["scores", "agreements"]},
            f,
            indent=2,
        )

    rich_rule(style="medium_purple1")
    rich_panel(
        tables=[res_table, rejected_table],
        panel_title="🏁 Experiment outcome 🏁",
        subtitle=f"\n✓ Statistics saved to: {output_stats}",
        border_style="medium_purple1",
        panel_padding=(1, 3),
        grid_padding=(1, 5),
    )
    rich_rule(style="medium_purple1")

    del data, res_table, rejected_table

    return stats


def finalize_jury_merge_mode(
    judge_files: list[Path],
    max_lengths: list[int],
    judge_configs: dict[str, JudgeConfig],
) -> dict[str, JudgeConfig]:
    """Dynamically assign max_seq_length attribute in `JudgeConfig` based on order of `judge_files`."""

    def _clean_string(text: str):
        text = text.replace("_vul", "")
        text = text.replace("_safe", "")
        return text.replace("_", "-")

    # this assumes judge files have the same names as the judges
    judge_names: list[str] = [_clean_string(p.stem) for p in judge_files]
    for key_name, length_val in zip(judge_names, max_lengths):
        judge_configs[key_name].max_seq_length = length_val

    return {k: v for k, v in judge_configs.items() if k in judge_names}

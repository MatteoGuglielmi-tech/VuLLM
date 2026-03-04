import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import cast

from .utilities import stateless_progress

logger = logging.getLogger(__name__)


JUDGE_WEIGHTS = {"Qwen-Coder": 0.25, "Qwen": 0.25, "Phi": 0.25, "DeepSeek": 0.25}

METRIC_WEIGHTS = {
    "correctness": 0.35,
    "completeness": 0.25,
    "clarity": 0.15,
    "technical_accuracy": 0.15,
    "logical_flow": 0.10,
}


def calculate_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative quality scores for each entry.

    Two-step process:
    1. Aggregate scores across judges: ŝ_k = Σ w_i · s_{i,k}
    2. Compute final score: Q_final = Σ λ_k · ŝ_k

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: entry_id, judge_short, and metric scores

    Returns
    -------
    pd.DataFrame
        DataFrame with entry_id and quality scores
    """
    quality_data = []

    for entry_id in df["entry_id"].unique():
        entry_evals = cast(pd.DataFrame, df[df["entry_id"] == entry_id])

        # Step 1: Aggregate across judges (equation 1)
        # ŝ_k = Σ w_i · s_{i,k}
        aggregated_scores = {}

        for metric in METRIC_WEIGHTS.keys():
            weighted_sum = 0
            total_weight = 0

            for _, row in entry_evals.iterrows():
                judge = cast(str, row["judge_short"])
                weight = JUDGE_WEIGHTS.get(judge, 0)
                score = row[metric]

                weighted_sum += weight * score
                total_weight += weight

            # Normalize by total weight (in case some judges missing)
            aggregated_scores[metric] = (
                weighted_sum / total_weight if total_weight > 0 else 0
            )

        # Step 2: Compute final quality score (equation 2)
        # Q_final = Σ λ_k · ŝ_k
        final_score = sum(
            METRIC_WEIGHTS[metric] * aggregated_scores[metric]
            for metric in METRIC_WEIGHTS.keys()
        )

        quality_data.append(
            {
                "entry_id": entry_id,
                **{
                    f"agg_{metric}": aggregated_scores[metric]
                    for metric in METRIC_WEIGHTS.keys()
                },
                "quality_score_final": final_score,
            }
        )

    return pd.DataFrame(quality_data)


def plot_quality_score_cdf(quality_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot cumulative distribution function (CDF) of quality scores.

    Parameters
    ----------
    quality_df : pd.DataFrame
        DataFrame with quality scores
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(25, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

    main_color = "#2E86AB"

    # 1. CDF of final quality score (main plot)
    ax1 = fig.add_subplot(gs[0, :])

    scores = cast(np.ndarray, quality_df["quality_score_final"].values)
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    ax1.plot(sorted_scores, cdf, linewidth=2.5, color=main_color, label="CDF")
    ax1.fill_between(sorted_scores, 0, cdf, alpha=0.3, color=main_color)

    # Add percentile lines
    percentiles = [25, 50, 75, 90, 95]
    percentile_values = np.percentile(scores, percentiles)

    colors_percentile = ["#FFA500", "#FF6347", "#8B0000", "#4B0082", "#000000"]

    for p, val, color in zip(percentiles, percentile_values, colors_percentile):
        ax1.axvline(
            val,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            color=color,
            label=f"P{p}: {val:.3f}",
        )
        ax1.axhline(p / 100, linestyle=":", linewidth=1, alpha=0.5, color=color)

    # Statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)

    ax1.axvline(
        mean_score,  # type: ignore[reportArgumentType]
        linestyle="-",
        linewidth=1,
        color="purple",
        label=f"Mean: {mean_score:.3f}",
    )
    ax1.axvline(
        median_score,  # type: ignore[reportArgumentType]
        linestyle="-",
        linewidth=1,
        color="green",
        label=f"Median: {median_score:.3f}",
    )
    ax1.axvline(
        0.85,  # type: ignore[reportArgumentType]
        linestyle="--",
        linewidth=1,
        color="red",
        label=f"Quality threshold: {0.85}",
    )

    ax1.set_xlabel("Final Quality Score (Q_final)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cumulative Probability", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Cumulative Distribution of Final Quality Scores",
        fontsize=15,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=14, ncol=2)
    ax1.set_xlim(sorted_scores[0] - 0.02, sorted_scores[-1] + 0.02)
    ax1.set_ylim(0, 1.05)

    # 2. Histogram with CDF overlay
    ax2 = fig.add_subplot(gs[1, 0])

    ax2.hist(
        scores,
        bins=50,
        alpha=0.6,
        color=main_color,
        edgecolor="black",
        density=True,
        label="PDF",
    )

    # Overlay CDF on secondary axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(sorted_scores, cdf, color="red", linewidth=1.2, label="CDF")
    ax2_twin.set_ylabel(
        "Cumulative Probability", fontsize=11, fontweight="bold", color="red"
    )
    ax2_twin.tick_params(axis="y", labelcolor="red")
    ax2_twin.set_ylim(0, 1.05)

    ax2.set_xlabel("Final Quality Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("Distribution (PDF + CDF)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    ax2_twin.legend(loc="upper right", fontsize=9)

    # 3. Per-metric aggregated score CDFs
    ax3 = fig.add_subplot(gs[1, 1])

    metric_colors = sns.color_palette("husl", len(METRIC_WEIGHTS))

    for i, metric in enumerate(METRIC_WEIGHTS.keys()):
        metric_scores = cast(np.ndarray, quality_df[f"agg_{metric}"].values)
        sorted_metric = np.sort(metric_scores)
        cdf_metric = np.arange(1, len(sorted_metric) + 1) / len(sorted_metric)

        ax3.plot(
            sorted_metric,
            cdf_metric,
            linewidth=2,
            label=metric.replace("_", " ").title(),
            color=metric_colors[i],
        )

    ax3.set_xlabel("Aggregated Score (ŝ_k)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cumulative Probability", fontsize=11, fontweight="bold")
    ax3.set_title(
        "CDF per Metric (Aggregated Across Judges)", fontsize=12, fontweight="bold"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1.05)

    # 4. Box plot of aggregated metrics
    ax4 = fig.add_subplot(gs[2, 0])

    metric_data = [quality_df[f"agg_{m}"].values for m in METRIC_WEIGHTS.keys()]
    metric_labels = [m.replace("_", " ").title() for m in METRIC_WEIGHTS.keys()]

    bp = ax4.boxplot(
        metric_data,
        labels=metric_labels,  # type: ignore[reportCallIssue]
        patch_artist=True,
        showmeans=True,
    )
    ax4.axhline(
        0.85,  # type: ignore[reportArgumentType]
        linestyle="--",
        linewidth=1,
        color="red",
        label=f"Quality threshold: {0.85}",
    )

    for patch, color in zip(bp["boxes"], metric_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel("Aggregated Score", fontsize=11, fontweight="bold")
    ax4.set_title("Distribution of Aggregated Metrics", fontsize=12, fontweight="bold")
    ax4.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Statistics table
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    stats_data = []
    stats_data.append(
        [
            "Final Score",
            f"{mean_score:.4f}",
            f"{median_score:.4f}",
            f"{std_score:.4f}",
            f"{np.min(scores):.4f}",
            f"{np.max(scores):.4f}",
        ]
    )

    # Per-metric stats
    for metric in METRIC_WEIGHTS.keys():
        metric_vals = cast(np.ndarray, quality_df[f"agg_{metric}"].values)
        stats_data.append(
            [
                metric.replace("_", " ").title(),
                f"{np.mean(metric_vals):.4f}",
                f"{np.median(metric_vals):.4f}",
                f"{np.std(metric_vals):.4f}",
                f"{np.min(metric_vals):.4f}",
                f"{np.max(metric_vals):.4f}",
            ]
        )

    table = ax5.table(
        cellText=stats_data,
        colLabels=["Metric", "Mean", "Median", "Std", "Min", "Max"],
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight final score row
    for i in range(6):
        table[(1, i)].set_facecolor("#FFD700")
        table[(1, i)].set_alpha(0.5)

    # Color other rows
    for i in range(2, len(stats_data) + 1):
        for j in range(6):
            table[(i, j)].set_facecolor(metric_colors[i - 2])
            table[(i, j)].set_alpha(0.3)

    ax5.set_title("Quality Score Statistics", fontsize=12, fontweight="bold", pad=20)

    # Add weight information as text
    weight_text = "Weights:\n\n"
    weight_text += "Judge Weights (w_i):\n"
    for judge, weight in JUDGE_WEIGHTS.items():
        weight_text += f"  {judge}: {weight:.2f}\n"
    weight_text += "\nMetric Weights (λ_k):\n"
    for metric, weight in METRIC_WEIGHTS.items():
        weight_text += f"  {metric}: {weight:.2f}\n"

    fig.text(
        0.91,
        0.35,
        weight_text,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="lightyellow",
            edgecolor="black",
            alpha=0.9,
        ),
        family="monospace",
        transform=fig.transFigure,
    )

    plt.savefig(output_dir / "quality_score_cdf.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    print(f"✅ Saved quality score CDF to {output_dir / 'quality_score_cdf.png'}")

    # Print summary
    print(f"\n📊 Quality Score Summary:")
    print(f"  Mean: {mean_score:.4f}")
    print(f"  Median: {median_score:.4f}")
    print(f"  Std: {std_score:.4f}")
    print(f"  Min: {np.min(scores):.4f}")
    print(f"  Max: {np.max(scores):.4f}")
    print(f"\n  Percentiles:")
    for p, val in zip(percentiles, percentile_values):
        print(f"    P{p}: {val:.4f}")

    return quality_df  # type: ignore[reportReturnType]


def export_quality_scores(quality_df: pd.DataFrame, output_dir: Path) -> None:
    """Export quality scores to CSV."""
    output_dir = Path(output_dir)
    output_path = output_dir / "quality_scores.csv"

    quality_df.to_csv(output_path, index=False)
    print(f"💾 Saved quality scores to {output_path}")


def run_quality_analysis(
    df: pd.DataFrame, output_dir: Path, names_map: dict[str, str]
) -> None:
    """Run complete quality score analysis."""

    df = df.copy()
    df["judge_short"] = df["judge_name"].map(names_map)  # type: ignore[reportArgumentType]

    with stateless_progress(description="📐 Calculating quality scores...") as status:
        quality_df = calculate_quality_scores(df)
        status.stop()

    with stateless_progress(description="📊 Generating CDF plots...") as status:
        plot_quality_score_cdf(quality_df, output_dir)

        status.update("💾 Exporting quality scores...")
        export_quality_scores(quality_df, output_dir)

        status.stop()

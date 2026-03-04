import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pathlib import Path
from typing import cast
from matplotlib.ticker import MultipleLocator

from .utilities import rich_progress

logger = logging.getLogger(__name__)


def count_jsonl_lines(jsonl_path: Path | str) -> int:
    with open(file=jsonl_path, mode="r") as f:
        return sum(1 for _ in f)


def load_evaluations(jsonl_path: Path) -> pd.DataFrame:
    """Load JSONL data into DataFrame with expanded evaluations."""
    data: list[dict[str, float | bool]] = []

    with open(file=jsonl_path, mode="r") as f:
        for idx, line in rich_progress(
            enumerate(f),
            total=count_jsonl_lines(jsonl_path=jsonl_path),
            description="🔍 Loading data...",
            status_fn=lambda _: "Running..."
        ):
            entry = json.loads(line)
            for eval in entry.get("per_judge_evaluations", []):
                row = {
                    "entry_id": idx,
                    "judge_name": eval["judge_name"],
                    "quality_score": eval["quality_score"],
                    "correctness": eval["correctness"],
                    "completeness": eval["completeness"],
                    "clarity": eval["clarity"],
                    "technical_accuracy": eval["technical_accuracy"],
                    "logical_flow": eval["logical_flow"],
                    "confidence": eval["confidence"],
                }
                data.append(row)

    return pd.DataFrame(data)


def create_name_mapping_legend(ax, name_map: dict) -> None:
    """
    Add a legend box explaining short names.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add legend to
    name_map : dict
        Mapping of full names to short names
    """
    legend_text = "Model Names:\n" + "\n".join(
        f"  {short}: {full}" for full, short in name_map.items()
    )

    ax.text(
        1.02,
        0.5,  # Position (right of plot)
        legend_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="lightgray",
            edgecolor="black",
            alpha=0.8,
        ),
    )


def plot_judge_confidence_summary(
    df: pd.DataFrame, output_dir: Path, names_map: dict[str, str]
) -> None:
    """
    Create comprehensive confidence analysis per judge.

    Shows: mean, median, std, and distribution of confidence scores.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["judge_short"] = df["judge_name"].map(names_map)  # type: ignore[reportArgumentType]

    # Calculate statistics per judge
    stats = (
        df.groupby("judge_short")["confidence"]
        .agg(  # apply multiple aggregation functions
            [
                ("mean", "mean"),  # Calculate mean, name the column "mean"
                ("median", "median"),  # Calculate median, name the column "median"
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
                ("count", "count"),
            ]
        )
        .round(3)
    )

    # Sort by mean confidence
    stats = stats.sort_values("mean", ascending=False)  # type: ignore[reportCallIssue]

    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(
        3,
        3,  # Changed to 3 columns for legend
        hspace=0.35,
        wspace=0.5,
        width_ratios=[2, 2, 0.8],  # Legend column is narrower
        height_ratios=[1.5, 1.2, 1],
    )

    judges = stats.index.tolist()
    colors = sns.color_palette("husl", len(judges))

    # 1. Bar plot with error bars (mean ± std)
    ax1 = fig.add_subplot(gs[0, :2])
    x_pos = np.arange(len(judges))

    bars = ax1.bar(
        x_pos,
        stats["mean"],
        yerr=stats["std"],
        capsize=5,
        alpha=0.7,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, stats["mean"], stats["std"])):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.01,
            f"{mean:.3f}\n±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_xlabel("Judge Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Confidence Score", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Mean Confidence per Judge (with Std Dev)", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks(x_pos)
    # ax1.set_xticklabels(judges, rotation=45, ha="right", fontsize=11)
    ax1.set_xticklabels(judges, fontsize=12)
    ax1.set_ylim(0, 1.25)
    # ax1.grid(True, alpha=0.3, axis="y")
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.grid(True, which="minor", linestyle=":", alpha=0.15, axis="y", color="darkgray")
    ax1.grid(True, which="major", linestyle="-", alpha=0.3, axis="y", color="darkgray")

    ax1.axhline(
        y=0.9,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="High confidence threshold: 0.9",
    )
    ax1.legend(fontsize=11)

    # Add name mapping legend
    # create_name_mapping_legend(ax1, name_map=names_map)

    # 2. Violin plot - distribution of confidence
    ax2 = fig.add_subplot(gs[1, :2])

    df_sorted = df.copy()
    df_sorted["judge_short"] = pd.Categorical(
        df_sorted["judge_short"], categories=judges, ordered=True
    )

    parts = ax2.violinplot(
        [df[df["judge_short"] == judge]["confidence"].values for judge in judges],  # type: ignore [reportAttributeAccessIssue, reportArgumentType]
        positions=x_pos,
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):  # type: ignore[reportArgumentType]
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax2.set_xlabel("Judge Model", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Confidence Score", fontsize=12, fontweight="bold")
    ax2.set_title("Confidence Distribution per Judge", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    # ax2.set_xticklabels(judges, rotation=45, ha="right")
    ax2.set_xticklabels(judges, ha="right")

    # Calculate global min/max across all judges with some padding
    all_confidence = cast(np.ndarray, df["confidence"].values)
    data_min = np.min(all_confidence)
    data_max = np.max(all_confidence)
    padding = (data_max - data_min) * 0.15  # 15% padding

    ax2.set_ylim(data_min - padding, data_max + padding)
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax2.grid(True, which="minor", linestyle=":", alpha=0.15, axis="y", color="darkgray")
    ax2.grid(True, which="major", linestyle="-", alpha=0.3, axis="y", color="darkgray")

    # 3. Box plot with statistics table
    ax3 = fig.add_subplot(gs[2, 0])

    bp = ax3.boxplot(
        [df[df["judge_short"] == judge]["confidence"].values for judge in judges],  # type: ignore[reportArgumentAccessIssue]
        labels=judges,  # type: ignore[reportCallIssue]
        patch_artist=True,
        showmeans=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax3.set_xlabel("Judge Model", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Confidence Score", fontsize=11, fontweight="bold")
    ax3.set_title("Confidence Range per Judge", fontsize=12, fontweight="bold")
    ax3.set_xticklabels(judges, fontsize=9)
    ax3.set_ylim(data_min - padding, data_max + padding)
    ax3.yaxis.set_major_locator(MultipleLocator(0.05))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.025))
    # ax3.grid(True, alpha=0.3, axis="y")
    ax3.grid(True, which="minor", linestyle=":", alpha=0.15, axis="y", color="darkgray")
    ax3.grid(True, which="major", linestyle="-", alpha=0.3, axis="y", color="darkgray")

    # 4. Statistics table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")

    # Create table data with short names
    table_data = []
    for judge in judges:
        row = stats.loc[judge]
        table_data.append(
            [
                judge,  # Short name
                f"{row['mean']:.3f}",
                f"{row['median']:.3f}",
                f"{row['std']:.3f}",
                f"{row['min']:.3f}",
                f"{row['max']:.3f}",
                f"{int(row['count'])}",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Judge", "Mean", "Median", "Std", "Min", "Max", "N"],
        cellLoc="center",
        loc="center",
        # colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1],
        colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.13, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)

    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Color rows
    for i in range(1, len(judges) + 1):
        for j in range(7):
            table[(i, j)].set_facecolor(colors[i - 1])
            table[(i, j)].set_alpha(0.3)

    ax4.set_title(
        "Confidence Statistics Summary", fontsize=13, fontweight="bold", pad=20
    )

    # 5. Legend box (separate column)
    ax_legend = fig.add_subplot(gs[:, 2])  # Span all rows, last column
    ax_legend.axis("off")

    # 5. Add legend
    legend_items = []
    for full, short in names_map.items():
        legend_items.append(f"{short}: {full}")

    legend_text = "Model Names:\n\n" + "\n".join(legend_items)

    # Position in figure coordinates (0-1 range)
    # x=0.82 means 82% from left edge
    # Adjust this value to move left/right
    fig.text(
        0.72,
        0.5,  # 72% from left edge
        legend_text,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="left",
        bbox=dict(
            boxstyle="round,pad=1",
            facecolor="lightblue",
            edgecolor="black",
            alpha=0.9,
            linewidth=2,
        ),
        linespacing=1.8,
        transform=fig.transFigure,
    )

    plt.savefig(
        output_dir / "judge_confidence_analysis.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()

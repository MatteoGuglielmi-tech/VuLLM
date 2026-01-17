import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal, cast
from pathlib import Path
from matplotlib.ticker import MultipleLocator


def calculate_disagreement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate disagreement metrics across judges for each entry.

    Returns DataFrame with entry_id and disagreement metrics.
    """
    disagreement_data = []

    for entry_id in df["entry_id"].unique():
        entry_evals = cast(pd.DataFrame, df[df["entry_id"] == entry_id])

        metrics = [
            "quality_score",
            "correctness",
            "completeness",
            "clarity",
            "technical_accuracy",
            "logical_flow",
            "confidence",
        ]

        disagreement_scores = {}

        for metric in metrics:
            values = cast(np.ndarray, entry_evals[metric].values)

            # Calculate disagreement measures
            std = np.std(values)
            range_val = np.max(values) - np.min(values)
            cv = (
                std / np.mean(values) if np.mean(values) > 0 else 0
            )  # Coefficient of variation

            disagreement_scores[f"{metric}_std"] = std
            disagreement_scores[f"{metric}_range"] = range_val
            disagreement_scores[f"{metric}_cv"] = cv

        # Overall disagreement (average across all metrics)
        avg_std = np.mean([disagreement_scores[f"{m}_std"] for m in metrics])
        avg_range = np.mean([disagreement_scores[f"{m}_range"] for m in metrics])

        disagreement_data.append(
            {
                "entry_id": entry_id,
                "num_judges": len(entry_evals),
                "avg_std": avg_std,
                "avg_range": avg_range,
                **disagreement_scores,
            }
        )

    return pd.DataFrame(disagreement_data)


def categorize_disagreement(
    avg_range: float,
) -> Literal["Strong Agreement", "Slight Disagreement", "Strong Disagreement"]:
    """
    Categorize disagreement level based on average range.

    Thresholds:
    - [0, 0.1): Strong Agreement
    - [0.1, 0.3): Slight Disagreement
    - [0.3, 1.0]: Strong Disagreement
    """
    if avg_range < 0.1:
        return "Strong Agreement"
    elif avg_range < 0.3:
        return "Slight Disagreement"
    else:
        return "Strong Disagreement"


def plot_disagreement_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Comprehensive disagreement analysis visualization.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disagreement_df = calculate_disagreement_metrics(df)
    disagreement_df["category"] = disagreement_df["avg_range"].apply(
        categorize_disagreement
    )
    category_counts = disagreement_df["category"].value_counts()

    n_categories = len(category_counts)
    color_map = {
        "Strong Agreement": "#2ecc71",  # Green
        "Slight Disagreement": "#f39c12",  # Orange
        "Strong Disagreement": "#e74c3c",  # Red
    }

    # Get colors for present categories only
    colors_pie = [color_map[cat] for cat in category_counts.index]  # type: ignore[reportArgumentType]
    explode = tuple([0.05] * n_categories)

    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # 1. Pie chart - Overall disagreement distribution
    ax1 = fig.add_subplot(gs[0, 0])

    _, _, autotexts = ax1.pie(  # type: ignore[reportAssigmentType]
        category_counts.values,  # type: ignore[reportArgumentType]
        labels=category_counts.index,  # type: ignore[reportArgumentType]
        autopct=lambda pct: f"{pct:.1f}%\n({round(pct/100 * category_counts.sum())})",
        colors=colors_pie,
        explode=explode,
        shadow=True,
        startangle=90,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    ax1.set_title(
        f"Disagreement Categories\n(N={len(disagreement_df)} entries)",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # 2. Histogram - Distribution of average range
    ax2 = fig.add_subplot(gs[0, 1:])

    _, bins, patches = ax2.hist(
        disagreement_df["avg_range"],
        bins=50,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    # Color bins by category
    for i, patch in enumerate(patches):  # type: ignore[repoortUndefinedVariable]
        bin_center = (bins[i] + bins[i + 1]) / 2
        if bin_center < 0.1:
            patch.set_facecolor("#2ecc71")  # Green
        elif bin_center < 0.3:
            patch.set_facecolor("#f39c12")  # Orange
        else:
            patch.set_facecolor("#e74c3c")  # Red

    # Add threshold lines
    ax2.axvline(
        0.1, color="orange", linestyle="--", linewidth=2, label="Slight threshold"
    )
    ax2.axvline(0.3, color="red", linestyle="--", linewidth=2, label="Strong threshold")

    # Add statistics
    mean_range = cast(np.float64, disagreement_df["avg_range"].mean())
    median_range = cast(np.float64, disagreement_df["avg_range"].median())

    ax2.axvline(
        mean_range,
        color="blue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {mean_range:.3f}",
    )
    ax2.axvline(
        median_range,
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Median: {median_range:.3f}",
    )

    ax2.set_xlabel("Average Range Across Metrics", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax2.set_title("Distribution of Disagreement Levels", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper right")
    ax2.xaxis.set_major_locator(MultipleLocator(0.05))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax2.grid(True, which="minor", linestyle=":", alpha=0.3, axis="x", color="darkgray")
    ax2.grid(True, which="major", linestyle="-", alpha=0.5, axis="x", color="darkgray")
    ax2.grid(True, alpha=0.5, axis="y")

    # 3. Per-metric disagreement (heatmap-style bar chart)
    ax3 = fig.add_subplot(gs[1, :])

    metrics = [
        "quality_score",
        "correctness",
        "completeness",
        "clarity",
        "technical_accuracy",
        "logical_flow",
        "confidence",
    ]

    metric_stds = cast(
        list[np.float64], [disagreement_df[f"{m}_std"].mean() for m in metrics]
    )
    metric_labels = [m.replace("_", " ").title() for m in metrics]

    bars = ax3.barh(
        metric_labels, metric_stds, color=sns.color_palette("RdYlGn_r", len(metrics))
    )

    # Add value labels
    for bar, val in zip(bars, metric_stds):
        ax3.text(
            val + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax3.set_xlabel("Average Standard Deviation", fontsize=12, fontweight="bold")
    ax3.set_title("Disagreement by Metric", fontsize=13, fontweight="bold")
    # ax3.grid(True, alpha=0.3, axis="x")
    ax3.xaxis.set_major_locator(MultipleLocator(0.005))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.0025))
    ax3.grid(True, which="minor", linestyle=":", alpha=0.3, axis="x", color="darkgray")
    ax3.grid(True, which="major", linestyle="-", alpha=0.5, axis="x", color="darkgray")

    # 4. Scatter: Avg Range vs Entry ID (to see patterns)
    ax4 = fig.add_subplot(gs[2, 0])

    scatter_colors = disagreement_df["category"].map(
        {  # type: ignore[reportArgumentType]
            "Strong Agreement": "#2ecc71",  # Green
            "Slight Disagreement": "#f39c12",  # Orange
            "Strong Disagreement": "#e74c3c",  # Red
        }
    )

    ax4.scatter(
        range(len(disagreement_df)),
        disagreement_df["avg_range"],
        c=scatter_colors,
        alpha=0.6,
        s=20,
    )

    ax4.axhline(0.1, color="orange", linestyle="--", alpha=0.7)
    ax4.axhline(0.3, color="red", linestyle="--", alpha=0.7)

    ax4.set_xlabel("Entry Index", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Avg Range", fontsize=11, fontweight="bold")
    ax4.set_title("Disagreement Across Entries", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # 5. Box plot - Range distribution per metric
    ax5 = fig.add_subplot(gs[2, 1:])

    range_data = [disagreement_df[f"{m}_range"].values for m in metrics]

    bp = ax5.boxplot(
        range_data,
        labels=metric_labels,  # type: ignore[reportCallIssue]
        patch_artist=True,
        showmeans=True,
    )

    for patch, color in zip(bp["boxes"], sns.color_palette("RdYlGn_r", len(metrics))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax5.set_ylabel("Range (Max - Min)", fontsize=11, fontweight="bold")
    ax5.set_title("Range Distribution per Metric", fontsize=12, fontweight="bold")
    ax5.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=9)
    ax5.yaxis.set_major_locator(MultipleLocator(0.05))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax5.grid(True, which="minor", linestyle=":", alpha=0.3, axis="y", color="darkgray")
    ax5.grid(True, which="major", linestyle="-", alpha=0.6, axis="y", color="darkgray")

    plt.savefig(
        output_dir / "disagreement_analysis.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()

    # Print summary statistics
    print("\n📊 Disagreement Summary:")
    print(f"  Total entries: {len(disagreement_df)}")
    print(f"\n  Categories:")
    for cat, count in category_counts.items():
        pct = count / len(disagreement_df) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

    print(f"\n  Average Range Statistics:")
    print(f"    Mean: {disagreement_df['avg_range'].mean():.4f}")
    print(f"    Median: {disagreement_df['avg_range'].median():.4f}")
    print(f"    Std: {disagreement_df['avg_range'].std():.4f}")

    return disagreement_df  # type: ignore[reportReturnType]

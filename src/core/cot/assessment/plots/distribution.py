import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_filtering_distributions(stats, quality_threshold: float, agreement_threshold: float, output_dir: Path):
    """Plot score distributions with decision boundaries."""

    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    scores = stats["scores"]

    if isinstance(stats["agreements"], dict):
        agreements = [stat_dict["agreement"] for stat_dict in stats["agreements"]]
    else:
        agreements = stats["agreements"]

    # Quality Score Distribution
    ax1 = axes[0, 0]
    ax1.hist(scores, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(
        quality_threshold, color="red",
        linestyle="--", linewidth=2,
        label=f"Threshold: {quality_threshold}",
    )
    ax1.axvline(
        stats["mean_score"], color="green",
        linestyle="--", linewidth=2,
        label=f'Mean: {stats["mean_score"]:.3f}',
    )
    ax1.set_xlabel("Quality Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Quality Score Distribution", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Agreement Score Distribution
    ax2 = axes[0, 1]
    ax2.hist(agreements, bins=50, alpha=0.7, color="coral", edgecolor="black")
    ax2.axvline(
        agreement_threshold, color="red",
        linestyle="--", linewidth=2,
        label=f"Threshold: {agreement_threshold}",
    )
    ax2.axvline(
        stats["mean_agreement"], color="green",
        linestyle="--", linewidth=2,
        label=f'Mean: {stats["mean_agreement"]:.3f}',
    )
    ax2.set_xlabel("Agreement Score", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Judge Agreement Distribution", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 2D Scatter: Quality vs Agreement (most informative)
    ax3 = axes[1, 0]

    # Color code by decision
    kept_mask = (np.array(scores) >= quality_threshold) & (np.array(agreements) >= agreement_threshold)
    ax3.scatter(
        np.array(scores)[kept_mask], np.array(agreements)[kept_mask],
        alpha=0.5, c="green", label="Kept", s=20
    )
    ax3.scatter(
        np.array(scores)[~kept_mask], np.array(agreements)[~kept_mask],
        alpha=0.5, c="red", label="Rejected", s=20
    )

    # Draw decision boundaries
    ax3.axvline(quality_threshold, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax3.axhline(agreement_threshold, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Shade regions
    ax3.axvspan(0, quality_threshold, alpha=0.1, color="red")
    ax3.axhspan(0, agreement_threshold, alpha=0.1, color="red")

    ax3.set_xlabel("Quality Score", fontsize=12)
    ax3.set_ylabel("Agreement Score", fontsize=12)
    ax3.set_title("Quality vs Agreement (Decision Space)", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Rejection Reasons Breakdown
    ax4 = axes[1, 1]

    categories = ['Low Quality\nOnly', 'Low Agreement\nOnly', 'Both Issues']
    values = [
        stats['low_quality'] - stats['both_issues'],
        stats['low_agreement'] - stats['both_issues'],
        stats['both_issues']
    ]
    colors = ["#ff6b6b", "#ffa07a", "#cd5c5c"]

    bars = ax4.bar(categories, values, color=colors, edgecolor="black", linewidth=1.5)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.set_title("Rejection Reasons Breakdown", fontsize=14, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig((output_dir / "filtering_analysis.png"), dpi=300, bbox_inches="tight")
    plt.show()


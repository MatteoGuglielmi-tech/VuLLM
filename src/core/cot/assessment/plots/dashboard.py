from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_summary_dashboard(stats, output_dir: Path):
    """Professional summary dashboard."""

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # 1. Big Numbers (Top Row)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(
        0.5,
        0.5,
        f"{stats['total']}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color="#2c3e50",
    )
    ax1.text(
        0.5,
        0.15,
        "Total Samples",
        ha="center",
        va="center",
        fontsize=14,
        color="#7f8c8d",
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(
        0.5,
        0.5,
        f"{stats['kept']}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color="#27ae60",
    )
    ax2.text(
        0.5,
        0.15,
        f"Kept ({stats['keep_rate']:.1%})",
        ha="center",
        va="center",
        fontsize=14,
        color="#27ae60",
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(
        0.5,
        0.5,
        f"{stats['rejected']}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color="#e74c3c",
    )
    ax3.text(
        0.5,
        0.15,
        f"Rejected ({1-stats['keep_rate']:.1%})",
        ha="center",
        va="center",
        fontsize=14,
        color="#e74c3c",
    )
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.text(
        0.5,
        0.5,
        f"{stats['mean_score']:.3f}",
        ha="center",
        va="center",
        fontsize=48,
        fontweight="bold",
        color="#3498db",
    )
    ax4.text(
        0.5,
        0.15,
        f"Mean Quality (±{stats['std_score']:.3f})",
        ha="center",
        va="center",
        fontsize=14,
        color="#3498db",
    )
    ax4.axis("off")

    # 2. Pie Chart: Keep vs Reject
    ax5 = fig.add_subplot(gs[1, 0:2])
    sizes = [stats["kept"], stats["rejected"]]
    colors = ["#27ae60", "#e74c3c"]
    explode = (0.05, 0)

    wedges, texts, autotexts = ax5.pie(
        sizes,
        explode=explode,
        labels=["Kept", "Rejected"],
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    ax5.set_title("Overall Filtering Result", fontsize=14, fontweight="bold", pad=20)

    # 3. Stacked Bar: Rejection Breakdown
    ax6 = fig.add_subplot(gs[1, 2:4])

    low_qual_only = stats["low_quality"] - stats["both_issues"]
    low_agree_only = stats["low_agreement"] - stats["both_issues"]
    both = stats["both_issues"]

    categories = ["Rejected Samples"]
    x = np.arange(len(categories))
    width = 0.6

    p1 = ax6.barh(x, [low_qual_only], width, label="Low Quality Only", color="#ff6b6b")
    p2 = ax6.barh(
        x,
        [low_agree_only],
        width,
        left=[low_qual_only],
        label="Low Agreement Only",
        color="#ffa07a",
    )
    p3 = ax6.barh(
        x,
        [both],
        width,
        left=[low_qual_only + low_agree_only],
        label="Both Issues",
        color="#cd5c5c",
    )

    ax6.set_yticks(x)
    ax6.set_yticklabels(categories)
    ax6.set_xlabel("Count", fontsize=12)
    ax6.set_title("Rejection Reasons", fontsize=14, fontweight="bold")
    ax6.legend(loc="upper right", fontsize=10)
    ax6.grid(axis="x", alpha=0.3)

    # Add counts on bars
    for p in [p1, p2, p3]:
        for bar in p:
            width = bar.get_width()
            if width > 0:
                ax6.text(
                    bar.get_x() + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )

    plt.suptitle(
        "LLM Judge Ensemble - Filtering Summary", fontsize=18, fontweight="bold", y=0.98
    )

    plt.savefig((output_dir / "filtering_summary.png"), dpi=300, bbox_inches="tight")
    plt.show()

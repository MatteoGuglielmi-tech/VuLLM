"""Overview plot with project counts and percentages."""

from pathlib import Path
import matplotlib.pyplot as plt

from ..types import SplitStats


def plot_distribution_overview(
    stats: dict[str, SplitStats], total_projects: int, output_dir: Path
) -> None:
    """
    Create overview plot with project counts and percentages.

    Parameters
    ----------
    stats : dict[str, SplitStats]
        Statistics per split
    total_projects : int
        Total unique projects
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Project Distribution Across Splits", fontsize=16, fontweight="bold")

    splits = list(stats.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # 1. Bar plot - Number of projects
    ax1 = axes[0, 0]
    project_counts = [stats[split]["projects"] for split in splits]
    bars = ax1.bar(
        splits,
        project_counts,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, count in zip(bars, project_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax1.set_ylabel("Number of Projects", fontsize=11, fontweight="bold")
    ax1.set_title("Projects per Split", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. Pie chart - Percentage distribution
    ax2 = axes[0, 1]
    percentages = [stats[split]["percentage"] for split in splits]

    _, texts, autotexts = ax2.pie(
        percentages,
        labels=[s.capitalize() for s in splits],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.05, 0.05, 0.05),
        shadow=True,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight("bold")

    ax2.set_title("Project Distribution (%)", fontsize=12, fontweight="bold")

    # 3. Bar plot - Average samples per project
    ax3 = axes[1, 0]
    avg_samples = [stats[split]["avg_samples_per_project"] for split in splits]
    bars = ax3.bar(
        splits, avg_samples, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )

    for bar, avg in zip(bars, avg_samples):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{avg:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add horizontal line for overall average
    overall_avg = sum(stats[s]["total_samples"] for s in splits) / total_projects
    ax3.axhline(
        y=overall_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Avg: {overall_avg:.1f}",
    )

    ax3.set_ylabel("Avg Samples per Project", fontsize=11, fontweight="bold")
    ax3.set_title("Average Project Size", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=10)

    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis("off")

    table_data = []
    for split in splits:
        s = stats[split]
        table_data.append(
            [
                split.capitalize(),
                f"{s['projects']}",
                f"{s['percentage']:.1f}%",
                f"{s['avg_samples_per_project']:.1f}",
                f"{s['total_samples']}",
            ]
        )

    # Add total row
    table_data.append(
        [
            "Total",
            str(total_projects),
            "100.0%",
            f"{sum(stats[s]['total_samples'] for s in splits) / total_projects:.1f}",
            str(sum(stats[s]["total_samples"] for s in splits)),
        ]
    )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Split", "Projects", "% Total", "Avg Size", "Samples"],
        cellLoc="center",
        loc="center",
        colWidths=[0.20, 0.15, 0.15, 0.20, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style data rows
    for i in range(1, 4):
        for j in range(5):
            table[(i, j)].set_facecolor(colors[i - 1])
            table[(i, j)].set_alpha(0.3)

    # Style total row
    for j in range(5):
        table[(4, j)].set_facecolor("#FFD700")
        table[(4, j)].set_alpha(0.5)
        table[(4, j)].set_text_props(weight="bold")

    ax4.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(
        output_dir / "project_distribution_overview.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()

    print(f"  ✅ Saved overview to {output_dir / 'project_distribution_overview.pdf'}")

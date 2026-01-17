from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_size_comparison(
    sample_counts: dict[str, dict[str, int]], output_dir: Path
) -> None:
    """
    Create detailed comparison of project sizes across splits.

    Parameters
    ----------
    sample_counts : dict[str, dict[str, int]]
        Sample counts per project per split
    output_dir : Path
        Output directory
    """
    _, axes = plt.subplots(1, 2, figsize=(14, 6))

    splits = list(sample_counts.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # 1. Violin plot
    ax1 = axes[0]

    violin_data = [list(sample_counts[split].values()) for split in splits]
    parts = ax1.violinplot(
        violin_data,
        positions=range(len(splits)),
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax1.set_xticks(range(len(splits)))
    ax1.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
    ax1.set_ylabel("Samples per Project", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Project Size Distribution (Violin Plot)", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")

    # 2. CDF comparison
    ax2 = axes[1]

    for split, color in zip(splits, colors):
        counts = sorted(sample_counts[split].values())
        cdf = np.arange(1, len(counts) + 1) / len(counts)

        ax2.plot(
            counts,
            cdf,
            linewidth=2.5,
            color=color,
            label=f"{split.capitalize()} (n={len(counts)})",
        )

    ax2.set_xlabel("Samples per Project", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Probability", fontsize=12, fontweight="bold")
    ax2.set_title("Cumulative Distribution (CDF)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(
        output_dir / "project_size_comparison.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()

    print(f"  ✅ Saved size comparison to {output_dir / 'project_size_comparison.pdf'}")

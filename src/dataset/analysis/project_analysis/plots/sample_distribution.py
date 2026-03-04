from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_sample_distribution(
    sample_counts: dict[str, dict[str, int]], output_dir: Path
) -> None:
    """
    Plot distribution of samples per project for each split.

    Parameters
    ----------
    sample_counts : dict[str, dict[str, int]]
        Sample counts per project per split
    output_dir : Path
        Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Samples per Project Distribution", fontsize=16, fontweight="bold")

    splits = list(sample_counts.keys())
    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    # Plot histogram for each split
    for idx, split in enumerate(splits):
        ax = axes[idx // 2, idx % 2]

        counts = list(sample_counts[split].values())

        if not counts:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(f"{split.capitalize()} Split", fontsize=12, fontweight="bold")
            continue

        # Histogram
        _ = ax.hist(
            counts,
            bins=30,
            color=colors[idx],
            alpha=0.7,
            edgecolor="black",
            linewidth=1,
        )

        # Statistics
        mean_val = np.mean(counts)
        median_val = np.median(counts)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.1f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.1f}",
        )

        ax.set_xlabel("Samples per Project", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{split.capitalize()} Split (n={len(counts)} projects)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    # Combined box plot in 4th subplot
    ax4 = axes[1, 1]

    box_data = [list(sample_counts[split].values()) for split in splits]
    bp = ax4.boxplot(
        box_data,
        labels=[s.capitalize() for s in splits],
        patch_artist=True,
        showmeans=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel("Samples per Project", fontsize=11, fontweight="bold")
    ax4.set_title("Comparison Across Splits", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        output_dir / "samples_per_project_distribution.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"  ✅ Saved sample distribution to {output_dir / 'samples_per_project_distribution.pdf'}"
    )

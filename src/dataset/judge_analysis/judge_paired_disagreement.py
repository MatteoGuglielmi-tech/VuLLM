import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def plot_judge_pair_disagreement(
    df: pd.DataFrame, output_dir: Path, names_map: dict[str, str]
) -> None:
    """
    Analyze disagreement between specific judge pairs.

    Creates heatmap showing pairwise disagreement.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["judge_short"] = df["judge_name"].map(names_map)  # type: ignore[reportArgumentType]

    judges = df["judge_short"].unique()
    n_judges = len(judges)

    # Initialize disagreement matrix
    disagreement_matrix = np.zeros((n_judges, n_judges))

    # Calculate pairwise disagreements
    for entry_id in df["entry_id"].unique():
        entry_evals = df[df["entry_id"] == entry_id]

        if len(entry_evals) < 2:
            continue

        # Get confidence scores for this entry
        for i, judge1 in enumerate(judges):
            for j, judge2 in enumerate(judges):
                if i >= j:
                    continue

                score1 = entry_evals[entry_evals["judge_short"] == judge1]["confidence"]
                score2 = entry_evals[entry_evals["judge_short"] == judge2]["confidence"]

                if len(score1) > 0 and len(score2) > 0:
                    diff = abs(score1.values[0] - score2.values[0])  # type: ignore[reportAttributeAccessIssue]
                    disagreement_matrix[i, j] += diff
                    disagreement_matrix[j, i] += diff

    # Normalize by number of entries
    disagreement_matrix /= len(df["entry_id"].unique())

    # Create heatmap
    _, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        disagreement_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.3
    )

    # Set ticks
    ax.set_xticks(np.arange(n_judges))
    ax.set_yticks(np.arange(n_judges))
    ax.set_xticklabels(judges, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(judges, fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Average Disagreement (Confidence)",
        rotation=270,
        labelpad=20,
        fontsize=12,
        fontweight="bold",
    )

    for i in range(n_judges):
        for j in range(n_judges):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{disagreement_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if disagreement_matrix[i, j] > 0.15 else "black",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_title(
        "Pairwise Judge Disagreement (Confidence Scores)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "judge_pair_disagreement.pdf", format="pdf", bbox_inches="tight"
    )
    plt.close()

    # print(f"✅ Saved pairwise disagreement to {output_dir / 'judge_pair_disagreement.png'}")

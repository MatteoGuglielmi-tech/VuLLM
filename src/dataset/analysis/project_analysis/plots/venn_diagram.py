from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def plot_venn_diagram(project_names: dict[str, set[str]], output_dir: Path) -> None:
    """
    Create Venn diagram showing project overlaps (or lack thereof).

    Parameters
    ----------
    project_names : dict[str, set[str]]
        Project names per split
    output_dir : Path
        Output directory
    """
    _, ax = plt.subplots(figsize=(10, 8))

    venn = venn3(
        [project_names["train"], project_names["validation"], project_names["test"]],
        set_labels=("Train", "Validation", "Test"),
        ax=ax,
    )

    region_colors = {
        "100": "#3498db",  # Only Train
        "010": "#e74c3c",  # Only Validation
        "001": "#2ecc71",  # Only Test
        "110": "#9b59b6",  # Train ∩ Validation
        "101": "#e67e22",  # Train ∩ Test
        "011": "#f39c12",  # Validation ∩ Test
        "111": "#c0392b",  # All three
    }

    # Apply colors only to existing patches
    for region_id, color in region_colors.items():
        try:
            patch = venn.get_patch_by_id(region_id)
            if patch is not None:
                patch.set_color(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)
        except KeyError:
            # Patch doesn't exist (region is empty)
            continue

    # Style labels
    if venn.set_labels:
        for text in venn.set_labels:
            if text is not None:
                text.set_fontsize(13)
                text.set_fontweight("bold")

    if venn.subset_labels:
        for text in venn.subset_labels:
            if text is not None:
                text.set_fontsize(11)
                text.set_fontweight("bold")

    ax.set_title(
        "Project Overlap Across Splits\n(Venn Diagram)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Calculate overlaps
    train_val = project_names["train"].intersection(project_names["validation"])
    train_test = project_names["train"].intersection(project_names["test"])
    val_test = project_names["validation"].intersection(project_names["test"])
    all_three = train_val.intersection(project_names["test"])

    total_overlap = len(train_val | train_test | val_test)

    # Add status annotation
    if total_overlap == 0:
        # Perfect - no overlap
        status_text = "✓ Zero Data Leakage: No project appears in multiple splits"
        box_color = "lightgreen"
        edge_color = "darkgreen"
        text_color = "green"
    else:
        # Has overlap - show details
        overlap_details = []
        if train_val:
            overlap_details.append(f"Train∩Val: {len(train_val)}")
        if train_test:
            overlap_details.append(f"Train∩Test: {len(train_test)}")
        if val_test:
            overlap_details.append(f"Val∩Test: {len(val_test)}")
        if all_three:
            overlap_details.append(f"All: {len(all_three)}")

        status_text = f'⚠ Data Leakage: {", ".join(overlap_details)}'
        box_color = "lightyellow"
        edge_color = "red"
        text_color = "red"

    ax.text(
        0.5,
        -0.15,
        status_text,
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=text_color,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=2,
            alpha=0.8,
        ),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "project_venn_diagram.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved Venn diagram to {output_dir / 'project_venn_diagram.png'}")

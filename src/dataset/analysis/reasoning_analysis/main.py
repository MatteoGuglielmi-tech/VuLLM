import sys
import json
import logging
import tiktoken

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from .logging_config import setup_logger
from .ui import rich_progress

logger = logging.getLogger(__name__)


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count the number of tokens in a text string."""
    return len(encoder.encode(text))


def analyze_reasoning_tokens(jsonl_path: str, output_dir: str = ".") -> None:
    """Analyze reasoning tokens and generate plots."""

    path = Path(jsonl_path)
    if not path.exists():
        logger.error(f"Error: File not found: {jsonl_path}")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # encoder = tiktoken.get_encoding("cl100k_base")
    encoder = tiktoken.encoding_for_model("gpt-4")

    func_tokens = []
    reasoning_tokens = []

     
    with open(file=path, mode="r") as f:
        total_samples: int = sum(1 for _ in f)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in rich_progress(
            iterable=enumerate(f, 1),
            total=total_samples,
            description="Processing dataset...",
        ):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue

            func = entry["func"]
            reasoning = entry["reasoning"]
            if entry["target"] == 1:
                continue

            if not reasoning:
                logger.warning(f"Warning: Missing 'reasoning' field at line {line_num}")
                continue

            func_tokens.append(count_tokens(func, encoder))
            reasoning_tokens.append(count_tokens(reasoning, encoder))

    func_tokens = np.array(func_tokens)
    reasoning_tokens = np.array(reasoning_tokens)

    # Calculate statistics
    stats = {
        "count": len(reasoning_tokens),
        "min": int(np.min(reasoning_tokens)),
        "max": int(np.max(reasoning_tokens)),
        "mean": float(np.mean(reasoning_tokens)),
        "median": float(np.median(reasoning_tokens)),
        "std": float(np.std(reasoning_tokens)),
    }

    print("\n" + "=" * 60)
    print("Reasoning Token Statistics")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.capitalize():>8}: {value:,.2f}")
        else:
            print(f"  {key.capitalize():>8}: {value:,}")

    # Calculate correlation
    correlation = np.corrcoef(func_tokens, reasoning_tokens)[0, 1]
    print(f"\nCorrelation (func vs reasoning tokens): {correlation:.4f}")

    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Plot 1: Violin Plot ---
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    violin_parts = ax1.violinplot(
        reasoning_tokens,
        positions=[1],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    # Customize violin plot colors
    for pc in violin_parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.7)

    violin_parts["cmeans"].set_color("#C44E52")
    violin_parts["cmeans"].set_linewidth(2)
    violin_parts["cmedians"].set_color("#55A868")
    violin_parts["cmedians"].set_linewidth(2)
    violin_parts["cbars"].set_color("#333333")
    violin_parts["cmaxes"].set_color("#333333")
    violin_parts["cmins"].set_color("#333333")

    ax1.set_ylabel("Token Count", fontsize=12)
    ax1.set_title(
        "Distribution of Reasoning Length (in Tokens)", fontsize=14, fontweight="bold"
    )
    ax1.set_xticks([1])
    ax1.set_xticklabels(["Reasoning"])

    # Add legend for mean/median lines
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], color="#C44E52", linewidth=2, label=f"Mean: {stats['mean']:,.2f}"
        ),
        Line2D(
            [0],
            [0],
            color="#55A868",
            linewidth=2,
            label=f"Median: {stats['median']:,.2f}",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Add text box with statistics
    textstr = "\n".join(
        [
            f"N = {stats['count']:,}",
            f"Min = {stats['min']:,}",
            f"Max = {stats['max']:,}",
            f"Mean = {stats['mean']:,.2f}",
            f"Median = {stats['median']:,.2f}",
            f"Std Dev = {stats['std']:,.2f}",
        ]
    )
    props = dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
    )
    ax1.text(
        0.02,
        0.98,
        textstr,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=props,
    )

    plt.tight_layout()
    violin_path = output_path / "reasoning_token_distribution.png"
    fig1.savefig(violin_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved violin plot: {violin_path}")
    plt.close(fig1)

    # --- Plot 2: Correlation Scatter Plot ---
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    # Use hexbin for better visualization with large datasets
    if len(func_tokens) > 5000:
        hb = ax2.hexbin(
            func_tokens,
            reasoning_tokens,
            gridsize=50,
            cmap="Blues",
            mincnt=1,
            linewidths=0.2,
        )
        cb = plt.colorbar(hb, ax=ax2, label="Count")
    else:
        ax2.scatter(
            func_tokens,
            reasoning_tokens,
            alpha=0.5,
            s=20,
            c="#4C72B0",
            edgecolors="none",
        )

    # Add regression line
    z = np.polyfit(func_tokens, reasoning_tokens, 1)
    p = np.poly1d(z)
    x_line = np.linspace(func_tokens.min(), func_tokens.max(), 100)
    ax2.plot(
        x_line,
        p(x_line),
        "r--",
        linewidth=2,
        label=f"Linear fit (r = {correlation:.3f})",
    )

    ax2.set_xlabel("Function Length (tokens)", fontsize=12)
    ax2.set_ylabel("Reasoning Length (tokens)", fontsize=12)
    ax2.set_title(
        "Correlation: Function Length vs Reasoning Length",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(loc="upper left")

    # Add correlation text box
    corr_text = f"Pearson r = {correlation:.4f}\nR² = {correlation**2:.4f}"
    props = dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
    )
    ax2.text(
        0.98,
        0.02,
        corr_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=props,
    )

    plt.tight_layout()
    corr_path = output_path / "func_reasoning_correlation.png"
    fig2.savefig(corr_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation plot: {corr_path}")
    plt.close(fig2)


if __name__ == "__main__":
    setup_logger()
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_enhanced_diversevul.jsonl> [output_dir]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    analyze_reasoning_tokens(jsonl_path, output_dir)

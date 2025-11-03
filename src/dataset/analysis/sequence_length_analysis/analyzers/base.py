import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from collections.abc import Generator
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

from src.core.cot.assessment.utilities.utils import build_table, rich_panel

from ..utilities import rich_rule
from ..datatypes import ReasoningSample, TokensStats
from ..loader_config import Loader

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


@dataclass
class TokenStats:
    """Statistics for token counts across dataset."""

    system_tokens: int = 0
    user_tokens: list[int] = field(default_factory=list)
    reasoning_tokens: list[int] = field(default_factory=list)
    answer_tokens: list[int] = field(default_factory=list)
    total_tokens: list[int] = field(default_factory=list)

    def add_sample(
        self,
        system_len: int,
        user_len: int,
        reasoning_len: int,
        answer_len: int,
        total_len: int,
    ):
        """Add a sample's token counts."""
        if self.system_tokens == 0:
            self.system_tokens = system_len

        self.user_tokens.append(user_len)
        self.reasoning_tokens.append(reasoning_len)
        self.answer_tokens.append(answer_len)
        self.total_tokens.append(total_len)

    def get_summary(self) -> dict:
        """Get statistical summary."""
        return {
            "system_tokens": {"value": self.system_tokens},
            "user_tokens": self._compute_stats(self.user_tokens),
            "reasoning_tokens": self._compute_stats(self.reasoning_tokens),
            "answer_tokens": self._compute_stats(self.answer_tokens),
            "total_tokens": self._compute_stats(self.total_tokens),
        }

    @staticmethod
    def _compute_stats(values: list[int]) -> dict:
        """Compute statistics for a list of values."""
        if not values:
            return {}

        arr = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


class BaseSequenceLengthAnalyzer(ABC):
    """Base class for analyzing token distribution in datasets."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def format_sample(self, sample: ReasoningSample) -> tuple[str, str, str]:
        """Format a dataset entry into system, user, and assistant messages.

        Returns
        -------
        tuple[str, str, str]
            (system_content, user_content, assistant_content)
        """
        pass

    @abstractmethod
    def count_tokens_for_sample(self, sample: ReasoningSample) -> TokensStats:
        """Count tokens for each component of a sample.

        Returns
        -------
        dict[str, int]
            Dictionary with token counts (keys vary by implementation)
        """
        pass

    def stream_jsonl(self, filepath: Path) -> Generator[dict[str, Any], None, None]:
        """Memory-efficient JSONL streaming.

        Parameters
        ----------
        filepath, Path:
            Path to JSONL file

        Yields
        ------
            dict for each line in JSONL
        """

        with open(file=filepath, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

    def _encode_and_count(self, text, add_special_tokens: bool = False) -> int:
        """Helper method for token counting."""
        return len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def apply_template(self, messages: list[dict[str,str]]):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def analyze_dataset(
        self,
        jsonl_path: Path,
        max_samples: int | None = None,
        output_dir: Path = Path("./sequence_analysis"),
    ) -> TokenStats:
        """Analyze entire dataset and generate comprehensive report.

        Parameters
        ----------
        jsonl_path, Path
            Path to JSONL dataset file
        max_samples, int (optional, default=None)
            Maximum samples to analyze (None = all)
        output_dir, Path (optional, default="./sequence_analysis")
            Directory to save results

        Returns
        -------
            TokenStats object with all statistics
        """

        logger.info(f"Starting analysis of {jsonl_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = TokenStats()
        sample_count = 0

        for entry in tqdm(self.stream_jsonl(jsonl_path), total=23836, desc="Analyzing samples"):
            if max_samples and sample_count >= max_samples:
                break

            sample = ReasoningSample(**entry)
            try:
                token_counts = self.count_tokens_for_sample(sample)
                stats.add_sample(
                    system_len=token_counts.system_tokens,
                    user_len=token_counts.user_tokens,
                    reasoning_len=token_counts.reasoning_tokens,
                    answer_len=token_counts.answer_tokens,
                    total_len=token_counts.total_tokens,
                )
                sample_count += 1
            except Exception as e:
                logger.warning(f"Error processing sample {sample_count}: {e}")
                continue

        logger.info(f"✅ Analyzed {sample_count} samples")

        # Generate all outputs
        summary = stats.get_summary()
        self._save_summary(summary, output_dir)
        self._generate_plots(stats, output_dir)
        self._generate_recommendations(summary, output_dir)

        return stats

    def _save_summary(self, summary: dict, output_dir: Path):
        """Save statistical summary to JSON."""

        output_file = output_dir / "token_statistics.json"
        with open(file=output_file, mode="w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📄 Statistics saved to {output_file}")

    def _generate_plots(self, stats: TokenStats, output_dir: Path):
        """Generate comprehensive visualizations."""

        with Loader("📊 Generating plots...", f"✅ All plots saved to {output_dir}", logger=logger):
            self._plot_distribution(data=stats.total_tokens, title="Distribution of Total Sequence Length",
                xlabel="Total Tokens", output_path=output_dir / "total_tokens_distribution.png",
            )
            self._plot_component_breakdown(stats, output_dir / "component_breakdown.png")
            self._plot_cumulative_distribution(stats.total_tokens, output_path=output_dir / "cumulative_distribution.png")
            self._plot_truncation_impact(stats.total_tokens, output_path=output_dir / "truncation_impact.png")
            self._plot_component_correlation( stats, output_dir / "component_correlation.png")

    def _plot_distribution(self, data: list[int], title: str, xlabel: str, output_path: Path):
        """Plot histogram with statistics overlay."""

        _, ax = plt.subplots(figsize=(12, 6))
        _ = ax.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

        mean_val = np.mean(data).astype(np.float64)
        median_val = np.median(data).astype(np.float64)
        p95_val = np.percentile(data, 95).astype(np.float64)
        p99_val = np.percentile(data, 99).astype(np.float64)

        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.0f}")
        ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.0f}")
        ax.axvline(p95_val, color="orange", linestyle="--", linewidth=2, label=f"95th percentile: {p95_val:.0f}")
        ax.axvline(p99_val, color="purple", linestyle="--", linewidth=2, label=f"99th percentile: {p99_val:.0f}")

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_component_breakdown(self, stats: TokenStats, output_path: Path):
        """Box plot showing token distribution for each component."""

        _, ax = plt.subplots(figsize=(12, 6))

        data_to_plot = [ stats.user_tokens, stats.reasoning_tokens, stats.answer_tokens, stats.total_tokens ]
        labels = ["User\nPrompt", "Reasoning", "Answer", "Total"]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            label=labels,
            patch_artist=True,
            showmeans=True,
        )

        # Color boxes
        colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Token Count", fontsize=12)
        ax.set_title("Token Distribution by Component", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        for i, data in enumerate(data_to_plot, start=1):
            mean_val = np.mean(data).astype(np.float64)
            ax.text(i, mean_val, f"{mean_val:.0f}", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_cumulative_distribution(self, data: list[int], output_path: Path):
        """Plot cumulative distribution function."""

        _, ax = plt.subplots(figsize=(12, 6))

        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100

        ax.plot(sorted_data, cumulative, linewidth=2, color="steelblue")

        for percentile in [90, 95, 99]:
            val = np.percentile(data, percentile).astype(np.float64)
            ax.axvline(val, color="red", linestyle="--", alpha=0.5)
            ax.axhline(percentile, color="red", linestyle="--", alpha=0.5)
            ax.text(val, percentile + 2, f"{percentile}th: {val:.0f}", fontsize=9, ha="left")

        ax.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax.set_ylabel("Cumulative Percentage (%)", fontsize=12)
        ax.set_title(
            "Cumulative Distribution of Sequence Lengths",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim((0.0, 105.0))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_truncation_impact(self, data: list[int], output_path: Path):
        """Show impact of different max_seq_length values."""

        _, ax = plt.subplots(figsize=(12, 6))

        test_lengths = [1024, 2048, 3072, 4096, 6144, 8192]
        truncation_percentages = []

        for max_len in test_lengths:
            truncated = sum(1 for x in data if x > max_len)
            pct = (truncated / len(data)) * 100
            truncation_percentages.append(pct)

        bars = ax.bar(
            range(len(test_lengths)),
            truncation_percentages,
            color="coral",
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xticks(range(len(test_lengths)))
        ax.set_xticklabels([str(x) for x in test_lengths])
        ax.set_xlabel("max_seq_length", fontsize=12)
        ax.set_ylabel("Percentage of Samples Truncated (%)", fontsize=12)
        ax.set_title(
            "Truncation Impact for Different max_seq_length Values",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add percentage labels on bars
        for bar, pct in zip(bars, truncation_percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_component_correlation(self, stats: TokenStats, output_path: Path):
        """Scatter plot showing correlation between components."""

        _, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: User (code) vs Reasoning
        axes[0].scatter(
            stats.user_tokens,
            stats.reasoning_tokens,
            alpha=0.5,
            s=20,
            color="steelblue",
        )
        axes[0].set_xlabel("User Tokens (Code Length)", fontsize=11)
        axes[0].set_ylabel("Reasoning Tokens", fontsize=11)
        axes[0].set_title(
            "Code Length vs Reasoning Length", fontsize=12, fontweight="bold"
        )
        axes[0].grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(stats.user_tokens, stats.reasoning_tokens)[0, 1]
        axes[0].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axes[0].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 2: Reasoning vs Answer
        axes[1].scatter(
            stats.reasoning_tokens, stats.answer_tokens, alpha=0.5, s=20, color="coral"
        )
        axes[1].set_xlabel("Reasoning Tokens", fontsize=11)
        axes[1].set_ylabel("Answer Tokens", fontsize=11)
        axes[1].set_title(
            "Reasoning Length vs Answer Length", fontsize=12, fontweight="bold"
        )
        axes[1].grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(stats.reasoning_tokens, stats.answer_tokens)[0, 1]
        axes[1].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axes[1].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_recommendations(self, summary: dict, output_dir: Path):
        """Generate recommendations for max_seq_length."""

        total_stats = summary["total_tokens"]

        recommendations = {
            "analysis_summary": {
                "total_samples": total_stats["count"],
                "mean_length": total_stats["mean"],
                "median_length": total_stats["median"],
                "p95_length": total_stats["p95"],
                "p99_length": total_stats["p99"],
                "max_length": total_stats["max"],
            },
            "recommendations": [],
        }

        # GPU tensor core alignment -> 5-10% faster training 
        # Flash Attention blocks -> Fewer partial blocks
        # Memory coalescing -> Better cache utilization
        cuda_performance_alignment: int = 512

        # Conservative (cover 95% of samples)
        rec_95 = int(np.ceil(total_stats["p95"] /  cuda_performance_alignment) * cuda_performance_alignment)
        truncated_95 = 5.0  # By definition
        recommendations["recommendations"].append(
            {
                "strategy": "Conservative (cover 95%)",
                "max_seq_length": rec_95,
                "samples_truncated_pct": truncated_95,
                "description": "Recommended for production. Only 5% of samples truncated.",
                "recommended": True,
            }
        )

        # Balanced (cover 99% of samples)
        rec_99 = int(np.ceil(total_stats["p99"] / cuda_performance_alignment) * cuda_performance_alignment)
        truncated_99 = 1.0
        recommendations["recommendations"].append(
            {
                "strategy": "Balanced (cover 99%)",
                "max_seq_length": rec_99,
                "samples_truncated_pct": truncated_99,
                "description": "Good balance. Only 1% of samples truncated.",
                "recommended": rec_99 <= 8192,  # Recommend if reasonable
            }
        )

        # Comprehensive (cover all samples)
        rec_max = int(np.ceil(total_stats["max"] / cuda_performance_alignment) * cuda_performance_alignment)
        recommendations["recommendations"].append(
            {
                "strategy": "Comprehensive (cover 100%)",
                "max_seq_length": rec_max,
                "samples_truncated_pct": 0.0,
                "description": "No truncation. May be memory-intensive.",
                "recommended": rec_max <= 8192,
            }
        )

        # Save recommendations
        output_file = output_dir / "recommendations.json"
        with open(file=output_file, mode="w") as f:
            json.dump(recommendations, f, indent=2)

        logger.info(f"✅ Recommendations saved to {output_file}")

        # Print recommendations
        table_data = {
            "Total samples analyzed": int(total_stats["count"]),
            "Mean sequence length": round(total_stats["mean"], 3),
            "Median sequence length": round(total_stats["median"], 3),
            "95th percentile": round(total_stats["p95"], 3),
            "99th percentile": round(total_stats["p99"], 3),
            "Maximum length": int(total_stats["max"]),
        }
        table = build_table(data=table_data, columns=["Metric", "Value [tokens]"])
        rich_rule()
        rich_panel(
            table,
            panel_title="📋 MAX_SEQ_LENGTH RECOMMENDATIONS 📋",
            border_style="royal_blue1",
            justify="center",
        )
        rich_rule()

        tables = []
        for rec in recommendations["recommendations"]:
            trunc_pct = f"{rec['samples_truncated_pct']:.2f}%"
            d = {
                "Strategy": rec["strategy"],
                "Max seq length": rec["max_seq_length"],
                "Truncated": trunc_pct,
                "Description": rec["description"],
                "Recommended?": rec["recommended"]
            }
            tables.append(build_table(data=d, columns=["Info","Value"]))

        rich_panel(
            tables,
            panel_title="🚨 RECOMMENDATIONS 🚨",
            border_style="yellow1",
        )
        rich_rule()

"""
Sequence Length Analyzer
Analyzes token distribution in CoT dataset to determine optimal max_seq_length.

Features:
- Memory-efficient streaming (handles large JSONL files)
- Multiple tokenizer support
- Comprehensive statistics and visualizations
- Truncation impact analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from collections.abc import Generator
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from loader_config import Loader


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


class SequenceLengthAnalyzer:
    """Analyzes token distribution in CoT dataset to determine optimal max_seq_length."""

    SYSTEM_PROMPT = (
        "You are an expert cybersecurity analyst specializing in C static code analysis. "
        "Your task is to analyze the provided code and produce a step-by-step reasoning "
        "chain explaining whether it contains a vulnerability."
    )

    PROMPT_SKELETON = (
        "**Analysis Instructions:**\n"
        "1. **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
        "2. **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
        "3. **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
        "4. **Conclude:** State your conclusion based on the analysis.\n\n"
        "**Output Format:**\n"
        "Produce a step-by-step list of your reasoning. After the list, your final answer must be "
        "prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'.\n"
        "--- CODE START ---\n"
        "{func_code}\n"
        "--- CODE END ---\n\n"
        "**Reasoning:**\n"
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str | None = None,
        prompt_skeleton: str | None = None,
    ):
        """Initialize analyzer.

        Parameters
        ----------
        tokenizer, PreTrainedTokenizer:
            Tokenizer to use for counting tokens
        system_prompt, str (optional, default=None):
            System prompt (uses default if None)
        prompt_skeleton, str (optional, default=None):
            User prompt template (uses default if None)
        """

        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.prompt_skeleton = prompt_skeleton or self.PROMPT_SKELETON

    @staticmethod
    def stream_jsonl(filepath: Path) -> Generator[dict[str, Any], None, None]:
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

    def format_sample(self, entry: dict) -> tuple[str, str, str]:
        """Format a dataset entry into system, user, and assistant messages.

        Parameters
        ----------
        entry, dict
            dictionary representing one line with keys: func, cwe, target, reasoning

        Returns
        -------
        tuple[str,str,str]:
            Tuple of (system_content, user_content, assistant_content)
        """

        system_content = self.system_prompt
        user_content = self.prompt_skeleton.format(func_code=entry["func"])

        if entry["target"] == 1 and entry.get("cwe"):
            cwe_string = ", ".join(entry["cwe"])
            final_answer = f" YES ({cwe_string})"
        else:
            final_answer = " NO"

        assistant_content = f"{entry['reasoning']}\n\nFinal Answer:{final_answer}"

        return system_content, user_content, assistant_content

    def count_tokens_for_sample(self, entry: dict) -> dict[str, int]:
        """Count tokens for each component of a sample.

        Parametrs
        ---------
        entry, Dict[str, Any]
            Dataset entry

        Returns
        -------
        dict[str, int]:
            Dict with token counts for each component
        """

        system_content, user_content, assistant_content = self.format_sample(entry)

        # Count individual components
        # System
        system_messages = [{"role": "system", "content": system_content}]
        system_formatted = self.tokenizer.apply_chat_template(
            system_messages, tokenize=False, add_generation_prompt=False
        )
        system_tokens = len(self.tokenizer.encode(system_formatted, add_special_tokens=False))  # type: ignore

        # System + User (to get user contribution)
        system_user_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        system_user_formatted = self.tokenizer.apply_chat_template(
            system_user_messages, tokenize=False, add_generation_prompt=False
        )
        system_user_tokens = len(self.tokenizer.encode(system_user_formatted, add_special_tokens=True))  # type: ignore
        user_tokens = system_user_tokens - system_tokens

        # Full conversation (system + user + assistant)
        full_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        total_tokens = len(self.tokenizer.encode(full_formatted, add_special_tokens=True))  # type: ignore

        # Assistant tokens = total - (system + user)
        assistant_tokens = total_tokens - system_user_tokens

        # Split assistant into reasoning and answer
        reasoning = entry["reasoning"]
        if entry["target"] == 1 and entry.get("cwe"):
            cwe_string = ", ".join(entry["cwe"])
            final_answer_str = f"\n\nFinal Answer: YES ({cwe_string})"
        else:
            final_answer_str = "\n\nFinal Answer: NO"

        # Count reasoning and answer separately (approximate)
        # These are approximate because we can't easily separate them post-tokenization
        reasoning_tokens_approx = len(
            self.tokenizer.encode(reasoning, add_special_tokens=False)
        )
        answer_tokens_approx = len(
            self.tokenizer.encode(final_answer_str, add_special_tokens=False)
        )

        # Adjust proportionally to match actual assistant token count
        total_approx = reasoning_tokens_approx + answer_tokens_approx
        if total_approx > 0:
            reasoning_tokens = int(
                assistant_tokens * (reasoning_tokens_approx / total_approx)
            )
            answer_tokens = assistant_tokens - reasoning_tokens
        else:
            reasoning_tokens = assistant_tokens
            answer_tokens = 0

        return {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "assistant_tokens": assistant_tokens,
            "total_tokens": total_tokens,
        }

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

        # Stream and analyze
        for entry in tqdm(
            self.stream_jsonl(jsonl_path), total=23836, desc="Analyzing samples"
        ):
            if max_samples and sample_count >= max_samples:
                break

            try:
                token_counts = self.count_tokens_for_sample(entry)
                stats.add_sample(
                    system_len=token_counts["system_tokens"],
                    user_len=token_counts["user_tokens"],
                    reasoning_len=token_counts["reasoning_tokens"],
                    answer_len=token_counts["answer_tokens"],
                    total_len=token_counts["total_tokens"],
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

            self._plot_distribution(
                stats.total_tokens,
                title="Distribution of Total Sequence Length",
                xlabel="Total Tokens",
                output_path=output_dir / "total_tokens_distribution.png",
            )

            self._plot_component_breakdown(stats, output_dir / "component_breakdown.png")
            self._plot_cumulative_distribution(stats.total_tokens, output_path=output_dir / "cumulative_distribution.png")
            self._plot_truncation_impact(stats.total_tokens, output_path=output_dir / "truncation_impact.png")
            self._plot_component_correlation( stats, output_dir / "component_correlation.png")


    def _plot_distribution(
        self,
        data: list[int],
        title: str,
        xlabel: str,
        output_path: Path,
    ):
        """Plot histogram with statistics overlay."""

        _, ax = plt.subplots(figsize=(12, 6))

        # Histogram
        _ = ax.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

        # Statistics lines
        mean_val = np.mean(data).astype(np.float64)
        median_val = np.median(data).astype(np.float64)
        p95_val = np.percentile(data, 95).astype(np.float64)
        p99_val = np.percentile(data, 99).astype(np.float64)

        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.0f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {median_val:.0f}",
        )
        ax.axvline(
            p95_val,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"95th percentile: {p95_val:.0f}",
        )
        ax.axvline(
            p99_val,
            color="purple",
            linestyle="--",
            linewidth=2,
            label=f"99th percentile: {p99_val:.0f}",
        )

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

        data_to_plot = [
            stats.user_tokens,
            stats.reasoning_tokens,
            stats.answer_tokens,
            stats.total_tokens,
        ]
        labels = ["User\n(Code)", "Reasoning", "Answer", "Total"]

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

        # Add mean values as text
        for i, data in enumerate(data_to_plot, 1):
            mean_val = np.mean(data).astype(np.float64)
            ax.text(
                i,
                mean_val,
                f"{mean_val:.0f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

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

        # Test various max_seq_length values
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
                f"{pct:.1f}%",
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

        # Print recommendations
        print("\n" + "=" * 80)
        print("📋 MAX_SEQ_LENGTH RECOMMENDATIONS")
        print("=" * 80)
        print(f"Total samples analyzed: {total_stats['count']}")
        print(f"Mean sequence length: {total_stats['mean']:.0f} tokens")
        print(f"Median sequence length: {total_stats['median']:.0f} tokens")
        print(f"95th percentile: {total_stats['p95']:.0f} tokens")
        print(f"99th percentile: {total_stats['p99']:.0f} tokens")
        print(f"Maximum length: {total_stats['max']:.0f} tokens")
        print("\n" + "-" * 80)
        print("RECOMMENDATIONS:")
        print("-" * 80)

        for rec in recommendations["recommendations"]:
            symbol = "✅" if rec["recommended"] else "⚠️"
            print(f"\n{symbol} {rec['strategy']}")
            print(f"   max_seq_length = {rec['max_seq_length']}")
            print(f"   Truncated: {rec['samples_truncated_pct']:.1f}%")
            print(f"   {rec['description']}")

        print("\n" + "=" * 80)
        print(f"✅ Recommendations saved to {output_file}")
        print("=" * 80 + "\n")


# ============================================================================
# UTILITIES
# ============================================================================
def analyze_single_tokenizer(
    dataset_path: Path,
    tokenizer_name: str,
    output_dir: Path,
    max_samples: int | None = None,
):
    """Analyze dataset with a single tokenizer."""
    logger.info(f"🚀 Starting analysis with tokenizer: {tokenizer_name}")

    try:
        with Loader(
            f"📦 Loading tokenizer...",
            f"✅ Tokenizer `{tokenizer_name}` loaded",
            logger=logger
        ):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        with Loader(
            f"📊 Initializing analyzer",
            f"✅ Initialized analyzer with tokenizer: `{tokenizer.name_or_path}`",
            logger=logger,
        ):
            analyzer = SequenceLengthAnalyzer(tokenizer=tokenizer)

        stats = analyzer.analyze_dataset(
            jsonl_path=dataset_path, max_samples=max_samples, output_dir=output_dir
        )

        logger.info("✅ Analysis complete!")
        return stats
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}")
        raise


def compare_tokenizers(
    dataset_path: Path,
    output_dir: Path,
    tokenizer_names: list[str],
    max_samples: int | None = None,
):
    """Compare multiple tokenizers."""

    logger.info(f"🔍 Comparing {len(tokenizer_names)} tokenizers")

    results = {}

    for tokenizer_name in tokenizer_names:
        # logger.info(f"\n{'='*80}")
        # logger.info(f"Analyzing with: {tokenizer_name}")
        # logger.info(f"{'='*80}\n")

        tokenizer_output_dir = output_dir / tokenizer_name.replace("/", "_")

        try:
            stats = analyze_single_tokenizer(
                dataset_path=dataset_path,
                tokenizer_name=tokenizer_name,
                output_dir=tokenizer_output_dir,
                max_samples=max_samples,
            )
            results[tokenizer_name] = stats.get_summary()
        except Exception as e:
            logger.error(f"Failed to analyze with {tokenizer_name}: {e}")
            continue

    # Generate comparison report
    if len(results) > 1:
        generate_comparison_report(results, output_dir)

    return results


def generate_comparison_report(results: dict, output_dir: Path):
    """Generate a comparison report for multiple tokenizers."""

    print("\n" + "=" * 80)
    print("📊 TOKENIZER COMPARISON")
    print("=" * 80)

    comparison = {"tokenizers": list(results.keys()), "comparison": {}}

    print(f"\n{'Tokenizer':<50} {'Mean':<12} {'P95':<12} {'P99':<12} {'Max':<12}")
    print("-" * 100)

    for tokenizer_name, stats in results.items():
        total_stats = stats["total_tokens"]
        display_name = (
            tokenizer_name[-47:] if len(tokenizer_name) > 47 else tokenizer_name
        )

        print(
            f"{display_name:<50} "
            f"{total_stats['mean']:>10.0f}  "
            f"{total_stats['p95']:>10.0f}  "
            f"{total_stats['p99']:>10.0f}  "
            f"{total_stats['max']:>10.0f}"
        )

        comparison["comparison"][tokenizer_name] = {
            "mean_total_tokens": total_stats["mean"],
            "p95_total_tokens": total_stats["p95"],
            "p99_total_tokens": total_stats["p99"],
            "max_total_tokens": total_stats["max"],
        }

    # Save comparison
    comparison_file = output_dir / "tokenizer_comparison.json"
    with open(file=comparison_file, mode="w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Comparison saved to {comparison_file}")
    print("=" * 80 + "\n")

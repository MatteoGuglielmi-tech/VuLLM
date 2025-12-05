from .analyzers import BaseSequenceLengthAnalyzer

import logging
import json

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utilities import rich_status, build_table, rich_panel, rich_rule

logger = logging.getLogger(__name__)


def analyze_single_tokenizer(
    dataset_path: Path,
    analyzer_type: type[BaseSequenceLengthAnalyzer],
    tokenizer_name: str,
    chat_template: str,
    output_dir: Path,
    max_samples: int | None = None,
):
    """Analyze dataset with a single tokenizer."""
    logger.info(f"🚀 Starting analysis with tokenizer: {tokenizer_name}")

    try:
        with rich_status(description=f"📦 Loading tokenizer..."):
            tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        with rich_status(description=f"📊 Initializing analyzer"):
            analyzer = analyzer_type(tokenizer=tokenizer, chat_template=chat_template)

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
    analyzer_type: type[BaseSequenceLengthAnalyzer],
    output_dir: Path,
    tokenizer_config: list[tuple[str,str]],
    max_samples: int | None = None,
):
    """Compare multiple tokenizers."""

    logger.info(f"🔍 Comparing {len(tokenizer_config)} tokenizers")

    results = {}

    for tokenizer_name, chat_template in tokenizer_config:
        tokenizer_output_dir = output_dir / tokenizer_name.replace("/", "_")

        try:
            stats = analyze_single_tokenizer(
                dataset_path=dataset_path,
                analyzer_type=analyzer_type,
                tokenizer_name=tokenizer_name,
                chat_template=chat_template,
                output_dir=tokenizer_output_dir,
                max_samples=max_samples,
            )
            results[tokenizer_name] = stats.get_summary()
        except Exception as e:
            logger.error(f"Failed to analyze with {tokenizer_name}: {e}")
            continue

    if len(results) > 1:
        generate_comparison_report(results, output_dir)

    return results


def generate_comparison_report(results: dict, output_dir: Path):
    """Generate a comparison report for multiple tokenizers."""

    comparison = {"tokenizers": list(results.keys()), "comparison": {}}

    tables = []
    for tokenizer_name, stats in results.items():
        total_stats = stats["total_tokens"]

        comparison["comparison"][tokenizer_name] = {
            "mean_total_tokens": total_stats["mean"],
            "p95_total_tokens": total_stats["p95"],
            "p99_total_tokens": total_stats["p99"],
            "max_total_tokens": total_stats["max"],
        }
        tables.append(
            build_table(
                data=comparison["comparison"][tokenizer_name],
                title=tokenizer_name,
                columns=["Metric", "Value"],
            )
        )

    rich_panel(
        tables,
        panel_title="📊 TOKENIZER COMPARISON 📊",
        border_style="light_slate_blue",
        layout="horizontal"
    )

    # Save comparison
    comparison_file = output_dir / "tokenizer_comparison.json"
    with open(file=comparison_file, mode="w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"✅ Comparison saved to {comparison_file}")
    rich_rule()

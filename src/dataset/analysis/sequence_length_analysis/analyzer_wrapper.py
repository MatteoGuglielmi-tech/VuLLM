from unsloth import get_chat_template

from .analyzers import AnalyzerVersion, AnalyzerFactory, TokenStats
from .datatypes import AssumptionMode, PromptPhase

import logging
import json

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from .utilities import rich_status, build_table, rich_panel, rich_rule

logger = logging.getLogger(__name__)


def _configure_tokenizer(
    tokenizer_name: str, chat_template: str | None
) -> PreTrainedTokenizer:
    """Configure tokenizer settings (chat template, padding, special tokens)."""

    model_name_lower = tokenizer_name.lower()
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # DeepSeek models
    if "deepseek" in model_name_lower:
        logger.info("🔍 Detected DeepSeek model - applying fixed chat template")

        tokenizer.chat_template = (
            "{% if not add_generation_prompt is defined %}"
            "{% set add_generation_prompt = false %}"
            "{% endif %}"
            "{{ bos_token }}"
            "{%- for message in messages %}"
            "    {%- if message['role'] == 'system' %}"
            "{{ message['content'] + '\\n' }}"
            "    {%- else %}"
            "        {%- if message['role'] == 'user' %}"
            "{{ '### Instruction:\\n' + message['content'] + '\\n' }}"
            "        {%- else %}"
            "{{ '### Response:\\n' + message['content'] + '\\n' + eos_token + '\\n' }}"
            "        {%- endif %}"
            "    {%- endif %}"
            "{%- endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '### Response:\\n' }}"
            "{% endif %}"
        )
        logger.info("✅ Applied fixed DeepSeek chat template")
        tokenizer.padding_side = "left"

    # CodeLlama models
    elif "codellama" in model_name_lower:
        logger.info("🔍 Detected CodeLlama model - applying Llama 2 chat template")
        tokenizer = get_chat_template(tokenizer, chat_template="llama")
        logger.info("✅ Applied Llama 2 chat template")
        tokenizer.padding_side = "left"

    elif chat_template is not None:
        logger.info(f"🎨 Applying chat template: {chat_template}")
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
            if "qwen" in tokenizer_name.lower():
                tokenizer.padding_side = "left"
            logger.info(f"✅ Applied {chat_template} chat template")
        except ValueError as e:
            logger.error(f"Invalid chat template: {chat_template}")
            raise ValueError(f"Chat template '{chat_template}' not found") from e

    else:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            logger.info("ℹ️  Using model's default chat template")
        else:
            logger.warning("⚠️  No chat template found or specified!")

    if not (tokenizer.pad_token or tokenizer.pad_token_id):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def analyze_single_tokenizer(
    dataset_path: Path,
    analyzer_version: AnalyzerVersion,
    tokenizer_name: str,
    chat_template: str,
    output_dir: Path,
    max_samples: int | None = None,
    # V2-specific (optional)
    assumption_mode: AssumptionMode | None = None,
    prompt_phase: PromptPhase | None = None,
    add_hierarchy: bool = False,
) -> TokenStats:
    """Analyze dataset with a single tokenizer."""
    logger.info(f"🚀 Starting analysis with tokenizer: {tokenizer_name}")

    try:
        with rich_status(description=f"📦 Loading tokenizer..."):
            tokenizer: PreTrainedTokenizer = _configure_tokenizer(
                tokenizer_name=tokenizer_name, chat_template=chat_template
            )

        with rich_status(description=f"📊 Initializing analyzer"):
            analyzer = AnalyzerFactory.create(
                version=analyzer_version,
                tokenizer=tokenizer,
                chat_template=chat_template,
                assumption_mode=assumption_mode,
                prompt_phase=prompt_phase,
                add_hierarchy=add_hierarchy,
            )

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
    analyzer_version: AnalyzerVersion,
    tokenizer_config: list[tuple[str,str]],
    output_dir: Path,
    max_samples: int | None = None,
    **kwargs
):
    """Compare multiple tokenizers."""
    logger.info(f"🔍 Comparing {len(tokenizer_config)} tokenizers")

    results = {}

    for tokenizer_name, chat_template in tokenizer_config:
        tokenizer_output_dir = output_dir / tokenizer_name.replace("/", "_")
        logger.info(f"\n{'='*70}")
        logger.info(f"Analyzing with: {tokenizer_name}")
        logger.info(f"{'='*70}\n")

        try:
            stats = analyze_single_tokenizer(
                dataset_path=dataset_path,
                analyzer_version=analyzer_version,
                tokenizer_name=tokenizer_name,
                chat_template=chat_template,
                output_dir=tokenizer_output_dir,
                max_samples=max_samples,
                **kwargs
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

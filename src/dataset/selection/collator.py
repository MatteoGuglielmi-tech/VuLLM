import sys
import jsonlines
import logging

from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer

from .datatypes import Sample
from .analyzers import BaseSequenceLengthAnalyzer, StatsRecord
from .utilities import (
    rich_status,
    rich_progress,
    rich_rule,
    build_table,
    rich_panel,
    read_and_parse_lines,
)

logger = logging.getLogger(__name__)


def separate_targets(
    jsonl_path: Path, output_dir: Path, save_interval: int = 100, force: bool = False
):
    """Split vulnerable and safe entries into separate files.

    Parameters
    ----------
    jsonl_path : Path
        Input JSONL file path
    output_dir : Path
        Output directory for separated files
    save_interval : int
        Number of samples to batch before writing to disk
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    vul_batch: list[Sample] = []
    safe_batch: list[Sample] = []

    vul_count: int = 0
    safe_count: int = 0

    vulpath: Path = output_dir / "vulnearble.jsonl"
    safepath: Path = output_dir / "safe.jsonl"
    mergepath: Path = output_dir / "complete.jsonl"

    with open(file=jsonl_path, mode="r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    if vulpath.exists() and safepath.exists() and not force:
        logger.warning("Per-target splits already present. Skipping procedure.")
        return
    try:
        with (
            jsonlines.open(file=vulpath, mode="w") as vul_writer,
            jsonlines.open(file=safepath, mode="w") as safe_writer,
            jsonlines.open(file=mergepath, mode="w") as writer,
        ):
            for sample in rich_progress(
                read_and_parse_lines(input_fp=jsonl_path),
                total=total_lines,
                description="🗡️ Separating concerns 🗡️",
            ):
                if sample["target"] == 1 and sample["cwe"]:
                    vul_count += 1
                    vul_batch.append(sample)
                    if len(vul_batch) % save_interval == 0:
                        vul_writer.write_all(vul_batch)
                        writer.write_all(vul_batch)
                        vul_batch.clear()
                elif sample["target"] == 0:
                    sample["cwe"] = []
                    safe_batch.append(sample)
                    safe_count += 1
                    if len(safe_batch) % save_interval == 0:
                        safe_writer.write_all(safe_batch)
                        writer.write_all(safe_batch)
                        safe_batch.clear()
                else:
                    continue

            if vul_batch:
                vul_writer.write_all(vul_batch)
                writer.write_all(safe_batch)
            if safe_batch:
                safe_writer.write_all(safe_batch)
                writer.write_all(safe_batch)

    except (IOError, OSError):
        logger.exception("❌ Failed during vulnerability isolation ❌")
        sys.exit(1)

    except Exception:
        logger.exception("⁉️ Unexpected error during vulnerability isolation ⁉️")
        sys.exit(1)

    else:
        logger.info(
            f"✅ [green]Separation completed: {vul_count} vulnerable, {safe_count} safe samples[/green]"
        )


def _analyze_single_tokenizer(
    dataset_path: Path,
    analyzer_type: type[BaseSequenceLengthAnalyzer],
    tokenizer_name: str,
) -> StatsRecord:
    """Analyze dataset with a single tokenizer."""
    logger.info(f"🚀 Starting analysis with tokenizer: {tokenizer_name}")

    try:
        with rich_status(description=f"Loading tokenizer..."):
            tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name
            )

        with rich_status(description=f"📊 Initializing analyzer"):
            analyzer = analyzer_type(tokenizer=tokenizer)

        stats = analyzer.analyze_dataset(jsonl_path=dataset_path)
        logger.info("✅ Analysis complete!")
        return stats
    except Exception:
        logger.exception(f"❌ Error occured during analysis ❌")
        raise


def analyze_filter_and_save(
    dataset_path: Path,
    analyzer_type: type[BaseSequenceLengthAnalyzer],
    tokenizer_name: str,
    n_samples: int,
    output_dir: Path,
    filename: str,
):
    stats = _analyze_single_tokenizer(
        dataset_path, analyzer_type=analyzer_type, tokenizer_name=tokenizer_name
    )
    debug = stats.get_samples_around_median(N=n_samples)

    table = build_table(debug, columns=["Info", "Value"])
    rich_rule()
    rich_panel(
        table,
        panel_title="Median-based selection info",
        border_style="light_slate_blue",
    )
    rich_rule()

    stats.as_jsonl(output_dir=output_dir, filename=filename)

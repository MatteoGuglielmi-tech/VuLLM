import logging
import jsonlines
import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import PreTrainedTokenizer

from ..utilities import read_and_parse_lines, rich_progress
from ..datatypes import Sample

logger = logging.getLogger(__name__)


@dataclass
class StatsRecord:
    record: dict = field(default_factory=dict)
    filtered_record: dict = field(default_factory=dict)
    index: int = 0
    max_token_length: int = 0
    running_tokens: list[int] = field(default_factory=list)

    def add_sample(self, sample: Sample, total_tokens: int):
        self.record[self.index] = {"sample": sample, "tokens": total_tokens}
        self.running_tokens.append(total_tokens)
        self.index += 1
        if total_tokens > self.max_token_length:
            self.max_token_length = total_tokens

    def sort_by_tokens(self):
        return sorted(self.record.items(), key=lambda item: item[1]["tokens"])

    def clear_running_vars(self):
        self.running_tokens.clear()

    def median(self) -> float:
        return float(np.median(np.array(self.running_tokens)))

    def get_samples_around_median(self, N: int) -> dict:
        """Select N/2 samples before median and (N/2)-1 after median.

        Parameters
        ----------
        N : int
            Total number of samples to select (must be odd for symmetric selection)

        Returns
        -------
        dict
            Dictionary with selected samples around the median
        """
        if not self.record:
            return {}

        median_value = self.median()
        sorted_items = self.sort_by_tokens()

        total_samples = len(sorted_items)

        if N >= total_samples:
            logger.warning(
                "Target number of sample to select greater than dataset size. (N>=total_samples)",
            )
            self.filtered_record = {key: value for key, value in sorted_items}
            return {
                "requested": N,
                "available": total_samples,
                "returned": total_samples,
                "median_value": median_value,
                "95th perc": round(
                    np.percentile(np.array(self.running_tokens), 95).astype(np.float64),
                    2,
                ),
                "99th perc": round(
                    np.percentile(np.array(self.running_tokens), 99).astype(np.float64),
                    2,
                ),
                "max_token_length": -999,
            }

        median_idx: int = min(
            range(len(sorted_items)),
            key=lambda i: abs(sorted_items[i][1]["tokens"] - median_value),
        )
        before: int = N // 2
        after: int = N - before - 1

        start_idx = median_idx - before
        end_idx = median_idx + after + 1

        if start_idx < 0:
            end_idx += abs(start_idx)
            start_idx = 0
        elif end_idx > total_samples:
            overflow = end_idx - total_samples
            start_idx = max(0, start_idx - overflow)
            end_idx = total_samples

        selected_items = sorted_items[start_idx:end_idx]
        self.filtered_record = {key: value for key, value in selected_items}

        stats = {
            "requested": N,
            "returned": len(self.filtered_record),
            "max_token_length": self.max_token_length,
            "95th perc": round(
                np.percentile(np.array(self.running_tokens), 95).astype(np.float64), 2
            ),
            "99th perc": round(
                np.percentile(np.array(self.running_tokens), 99).astype(np.float64), 2
            ),
            "median_value": median_value,
            "median_position": median_idx,
            "range_start": start_idx,
            "range_end": end_idx - 1,
            "token_range": (
                selected_items[0][1]["tokens"],
                selected_items[-1][1]["tokens"],
            ),
        }

        return stats

    def as_jsonl(self, output_dir: Path, filename: str):
        """Save filtered records as JSONL."""
        output_dir.mkdir(parents=True, exist_ok=True)
        outpath = output_dir / filename

        vals = list(self.filtered_record.values())

        with jsonlines.open(file=outpath, mode="w") as writer:
            for data in rich_progress(
                vals, total=len(vals), description="✍️ Writing samples ✍️"
            ):
                writer.write(data["sample"].to_dict)

        logger.info(f"💾 Saved {len(vals)} samples to {outpath} 💾")


class BaseSequenceLengthAnalyzer(ABC):
    """Base class for analyzing token distribution in datasets."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def format_sample(self, sample: Sample) -> tuple[str, str, str]:
        """Format a dataset entry into system, user, and assistant messages.

        Returns
        -------
        tuple[str, str, str]
            (system_content, user_content, assistant_content)
        """
        pass

    @abstractmethod
    def count_tokens_for_sample(self, sample: Sample) -> int:
        """Count tokens for each component of a sample.

        Returns
        -------
        dict[str, int]
            Dictionary with token counts (keys vary by implementation)
        """
        pass

    def _encode_and_count(self, text, add_special_tokens: bool = False) -> int:
        """Helper method for token counting."""
        return len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def apply_template(self, messages: list[dict[str, str]]):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def analyze_dataset(self, jsonl_path: Path) -> StatsRecord:
        """Analyze entire dataset and generate comprehensive report.

        Parameters
        ----------
        jsonl_path, Path
            Path to JSONL dataset file

        Returns
        -------
            TokenStats object with all statistics
        """

        logger.info(f"Starting analysis of {jsonl_path}")

        samples_count: int = 0
        record: StatsRecord = StatsRecord()

        with open(file=jsonl_path, mode="r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        for sample in tqdm(
            read_and_parse_lines(input_fp=jsonl_path),
            total=total_lines,
            desc="Analyzing samples",
        ):
            try:
                token_counts = self.count_tokens_for_sample(sample)
                record.add_sample(sample=sample, total_tokens=token_counts)
                samples_count += 1
            except:
                logger.exception(f"Error processing sample {samples_count}")
                continue

        logger.info(f"✅ [green] Analyzed {samples_count} samples [/green]")

        return record

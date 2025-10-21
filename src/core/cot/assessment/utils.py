import json
import random
import logging

from pathlib import Path
from typing import Any
from collections.abc import Generator

from src.core.cot.loader_config import Loader

logger = logging.getLogger(__name__)

def read_jsonl(fp: str | Path) -> list[dict[str, Any]]:
    with open(file=fp, mode="r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def _stream_random_sample(input_fp: str, sample_size: int) -> tuple[int, Generator]:
    """Efficiently creates a generator for a random sample from a large JSONL file."""

    with Loader(
        desc_msg=f"📖 Counting entries from {input_fp}",
        end_msg="✅ Total number of entries acquired",
        logger=logger,
    ):
        with open(file=input_fp, mode="r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

    if sample_size > total_lines:
        sample_size = total_lines

    sample_indices = set(random.sample(range(total_lines), sample_size))

    def generator():
        with open(file=input_fp, mode="r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in sample_indices:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line {i} during sampling. Skipping.")

    return sample_size, generator()

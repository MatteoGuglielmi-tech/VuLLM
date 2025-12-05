import json
import logging

from collections.abc import Iterator
from pathlib import Path

from pydantic import ValidationError

from ..datatypes import Sample
from ..utilities import require_file, validate_jsonl


logger = logging.getLogger(__name__)


@require_file("input_fp")
@validate_jsonl("input_fp")
def read_and_parse_lines(input_fp: Path) -> Iterator[Sample]:
    """Streamline reading of JSONL file."""

    with open(file=input_fp, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                stripped_line = line.strip()
                if not stripped_line: continue

                yield json.loads(line)
            except (json.JSONDecodeError, KeyError, ValidationError):
                logger.warning(f"Could not parse line {i}. Skipping.")
                continue


@require_file("input_fp")
@validate_jsonl("input_fp")
def count_total_lines(input_fp: Path) -> int:
    """Count total lines in file for progress bar"""

    with open(file=input_fp, mode="r", encoding="utf-8") as f:
        return sum(1 for _ in f)

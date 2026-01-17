"""
OutputParser: Lightweight parser for extracting reasoning and verdict
from raw CoT model outputs.

Designed as a drop-in component for the generation pipeline.
"""

from __future__ import annotations

import logging
# import re

from dataclasses import dataclass, field
from pydantic import ValidationError

from .definitions import ParseErrorType, ParseResult, ParseError
from ..prompts import ExpectedModelResponse

logger = logging.getLogger(__name__)


@dataclass
class ParserStatistics:
    """Tracks parsing statistics across a dataset."""

    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    errors_by_type: dict[ParseErrorType, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.errors_by_type:
            self.errors_by_type = {t: 0 for t in ParseErrorType}

    @property
    def success_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.successful / self.total_processed

    def record_success(self) -> None:
        self.total_processed += 1
        self.successful += 1

    def record_failure(self, error_type: ParseErrorType) -> None:
        self.total_processed += 1
        self.failed += 1
        self.errors_by_type[error_type] += 1

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "PARSER STATISTICS SUMMARY",
            "=" * 60,
            f"Total processed:  {self.total_processed:,}",
            f"Successful:       {self.successful:,} ({self.success_rate:.1%})",
            f"Failed:           {self.failed:,}",
            "",
            "Errors by type:",
        ]

        for error_type, count in self.errors_by_type.items():
            if count > 0:
                lines.append(f"  {error_type.value}: {count:,}")

        lines.append("=" * 60)
        return "\n".join(lines)


# def _clean_json_string(json_str: str) -> str:
#     """
#     Clean common JSON formatting issues from model outputs.
#
#     Handles:
#     - Escaped newlines and quotes
#     - Control characters
#     """
#
#     cleaned = json_str.replace("\\n", " ").replace("\\t", " ")
#     cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
#     return cleaned

class OutputParser:
    """
    Parses raw model outputs into structured samples with reasoning and verdict.

    Attributes
    ----------
    stats : ParserStatistics
        Statistics collected during parsing.
    min_reasoning_length : int
        Minimum required length for reasoning text.

    Examples
    --------
    >>> parser = OutputParser()
    >>> result = parser.parse(raw_response)
    >>> if result.success:
    ...     print(result.sample["reasoning"])
    ...     print(result.sample["verdict"])
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset statistics for a new parsing run."""
        self.stats = ParserStatistics()

    def parse(self, response: str) -> ParseResult:
        """
        Parse a raw model response into reasoning and verdict.

        Parameters
        ----------
        response : str
            Raw text response from the model.

        Returns
        -------
        ParseResult
            Result containing parsed sample or error information.
        """

        # empty input
        if not response or not response.strip():
            error = ParseError(
                error_type=ParseErrorType.EMPTY_RESPONSE,
                message="Empty or whitespace-only response",
            )
            self.stats.record_failure(ParseErrorType.EMPTY_RESPONSE)
            return ParseResult(success=False, error=error)

        # find JSON boundaries
        json_start: int = response.find("{")
        json_end: int = response.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            error = ParseError(
                error_type=ParseErrorType.MISSING_VERDICT,
                message="No JSON object found in response",
                raw_output=response[:200] if len(response) > 200 else response,
            )
            self.stats.record_failure(ParseErrorType.MISSING_VERDICT)
            return ParseResult(success=False, error=error)

        # extract and parse JSON
        json_str: str = response[json_start:json_end]
        # cleaned_json = _clean_json_string(json_str)

        try:
            validated = ExpectedModelResponse.model_validate_json(json_data=json_str)
            # parsed_sample = validated.model_dump(mode="json")
        except ValidationError as e:
            error = ParseError(
                error_type=ParseErrorType.VALIDATION_ERROR,
                message=f"Verdict validation failed: {e}",
                raw_output=json_str,
            )
            self.stats.record_failure(ParseErrorType.INVALID_JSON)
            return ParseResult(success=False, error=error)

        self.stats.record_success()
        return ParseResult(success=True, sample=validated)

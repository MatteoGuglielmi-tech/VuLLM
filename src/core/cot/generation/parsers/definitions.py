from enum import Enum
from typing import TypeAlias
from dataclasses import dataclass
from ..prompts import ExpectedModelResponse

CWEId: TypeAlias = int


class ParseErrorType(Enum):
    """Classification of parsing errors for analytics."""

    MISSING_REASONING = "missing_reasoning"
    MISSING_VERDICT = "missing_verdict"
    INVALID_JSON = "invalid_json"
    VALIDATION_ERROR = "validation_error"
    EMPTY_RESPONSE = "empty_response"


@dataclass
class ParseError:
    """Detailed information about a parsing failure."""

    error_type: ParseErrorType
    message: str
    raw_output: str | None = None

    def __str__(self) -> str:
        return f"[{self.error_type.value}] {self.message}"


@dataclass
class ParseResult:
    """Result of parsing a single response."""

    success: bool
    sample: ExpectedModelResponse | None = None
    error: ParseError | None = None

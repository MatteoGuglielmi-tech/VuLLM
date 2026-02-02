from typing import TypedDict
from enum import StrEnum
from dataclasses import dataclass, fields
from transformers import BatchEncoding
from pydantic import BaseModel, Field, model_validator, field_validator

ChatTmplt = str | list[int] | list[str] | list[list[int]] | BatchEncoding
Message = dict[str, ChatTmplt]
Messages = list[dict[str,str]]


class ReasoningSampleDict(TypedDict):
    project: str
    cwe: list[str]
    target: int
    func: str
    cwe_desc: list[str]
    reasoning: str


@dataclass
class ReasoningSample():
    """Container for reasoning samples"""

    func: str
    target: int
    cwe: list[str]
    cwe_desc: list[str]
    project: str
    reasoning: str

    def to_dict(self) -> ReasoningSampleDict:
        return {
            "func": self.func,
            "target": self.target,
            "cwe": self.cwe,
            "cwe_desc": self.cwe_desc,
            "project": self.project,
            "reasoning": self.reasoning,
        }

    @classmethod
    def required_keys(cls) -> list[str]:
        """
        Get list of required field names dynamically from dataclass fields.

        Returns
        -------
        list[str]
            List of field names
        """
        return [field.name for field in fields(cls)]

    @property
    def has_cwes(self) -> bool:
        return len(self.cwe) > 0

@dataclass
class TokensStats:
    """Container for tokens related metrics per sample."""

    system_tokens: int
    user_tokens: int
    assistant_tokens: int
    total_tokens: int

    @property
    def to_dict(self):
        return self.__dict__

class PromptPhase(StrEnum):
    """Phase of prompt usage."""

    CONSTRAINED_TRAINING = "training" # ass + reminder
    FREE_TRAINING = "training_no_assumptions" # no ass + no reminder
    FULL_CONSTRAINED_INFERENCE = "inference" # full inference -> attacks + assumptions + reminder
    ATTACK_CONSTRAINED_INFERENCE = "inference_attacks_only"  # attacks only, no assumptions no reminder
    FREE_INFERENCE = "inference_barebone" # pure inference -> no attacks, no assumptions, no reminder

class AssumptionMode(StrEnum):
    """Assumption modes for vulnerability analysis."""

    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    NONE = "none"


class VulnInfo(BaseModel):
    """Vulnerability information - nested model."""

    cwe_id: int = Field(..., ge=1, le=9999, description="CWE ID must be 1-9999")
    description: str = Field(..., min_length=1, description="CWE description from MITRE")


class VerdictStruct(BaseModel):
    """Verdict structure - nested model."""

    is_vulnerable: bool = Field(..., description="Required boolean")
    cwe_list: list[int] = Field(default_factory=list, description="List of CWE IDs")


class ExpectedModelResponse(BaseModel):
    """
    Container for parsed JSON response.

    Use model_validate_json() to parse from JSON string.
    Validation errors indicate parse failures.
    """

    reasoning: str = Field(..., min_length=50, description="Required string")
    vulnerabilities: list[VulnInfo] = Field(default_factory=list, description="List of VulnInfo models")
    verdict: VerdictStruct = Field(..., description="Required VerdictStruct model")

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        """Validate reasoning is not just whitespace."""
        if not v.strip():
            raise EmptyReasoningError("Reasoning cannot be empty or whitespace")
        return v.strip()

    @model_validator(mode="after")
    def check_vulnerabilities_match_verdict(self) -> "ExpectedModelResponse":
        """
        Validate entire model after all fields are set.

        This runs AFTER all field validators and AFTER all fields
        have been assigned, so it is safe to access all attributes.
        """
        if self.verdict.is_vulnerable and len(self.vulnerabilities) == 0:
            raise MismatchCWEError(
                "Verdict indicates vulnerability but no vulnerabilities listed. "
                f"Expected CWEs: {self.verdict.cwe_list}"
            )

        verdict_cwes = set(self.verdict.cwe_list)
        vuln_cwes = set(v.cwe_id for v in self.vulnerabilities)

        if verdict_cwes != vuln_cwes:
            raise MismatchCWEError(
                f"CWE mismatch: verdict has {verdict_cwes}, "
                f"but vulnerabilities have {vuln_cwes}"
            )

        return self

    @property
    def is_vulnerable(self) -> bool:
        return self.verdict.is_vulnerable

    @property
    def cwe_list(self) -> list[int]:
        return self.verdict.cwe_list

class EmptyReasoningError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MismatchCWEError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

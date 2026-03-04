#NOTE: Pydantic's validation flow:
"""
response = ExpectedModelResponse.model_validate_json(json_str)

┌─────────────────────────────────────────────────────┐
│ 1. Parse JSON string to dict                        │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 2. Validate ExpectedModelResponse top-level fields  │
│    ✓ reasoning: str? Required? ✅                   │
│    ✓ vulnerabilities: list? Has default? ✅         │
│    ✓ verdict: dict? Required? ✅                    │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 3. Validate nested VerdictStruct                    │
│    ✓ is_vulnerable: bool? Required? ✅              │
│    ✓ cwe_list: list[int]? Has default? ✅           │
│    ✓ Each cwe_list item is int? ✅                  │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 4. Validate each VulnInfo in vulnerabilities list   │
│    For item 0:                                      │
│    ✓ cwe_id: int? Required? 1-9999? ✅              │
│    ✓ description: str? Required? ≥100 chars? ✅     │
│    For item 1:                                      │
│    ✓ ... same checks                                │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 5. Run field validators (@field_validator)          │
│    ✓ reasoning_not_empty                            │
└─────────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────┐
│ 6. Run model validators (@model_validator)          │
│    ✓ validate_consistency                           │
└─────────────────────────────────────────────────────┘
                      ↓
                   ✅ Done!
"""


from typing import Literal, TypedDict, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetExample(TypedDict, total=True):
    func: str
    target: int
    project: str
    reasoning: str
    cwe: list[str]
    cwe_desc: list[str]


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "developer"]
    content: Union[str, list[dict[str, Any]]]


Messages = list[Message]
CweId = str  # CWE-XXX, CWE-YYY
CWEDescription = str


class VulnInfo(BaseModel):
    """Vulnerability information - nested model."""

    cwe_id: int = Field(..., ge=1, le=9999, description="CWE ID must be 1-9999")
    description: str = Field(..., min_length=10, description="At least 10 chars")


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

    reasoning: str = Field(..., description="Required string")
    vulnerabilities: list[VulnInfo] = Field(
        default_factory=list, description="List of VulnInfo models"
    )
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

        # Optional: Check CWE consistency
        verdict_cwes = set(self.verdict.cwe_list)
        vuln_cwes = set(v.cwe_id for v in self.vulnerabilities)

        if verdict_cwes != vuln_cwes:
            raise MismatchCWEError(
                f"CWE mismatch: verdict has {verdict_cwes}, "
                f"but vulnerabilities have {vuln_cwes}"
            )

        return self

class EmptyReasoningError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class MismatchCWEError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)




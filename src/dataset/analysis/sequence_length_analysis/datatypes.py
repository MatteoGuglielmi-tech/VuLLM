from typing import TypedDict
from dataclasses import dataclass, fields
from transformers import BatchEncoding


Message = dict[str, str | list[int] | list[str] | list[list[int]] | BatchEncoding]
Messages = list[dict[str,str]]


class VulnInfo(TypedDict, total=True):
    cwe_id: int
    description: str


class VerdictStruct(TypedDict, total=True):
    is_vulnerable: bool
    cwe_list: list[int]


class ResponseStruct(TypedDict, total=True):
    reasoning: str
    vulnerabilities: list[VulnInfo]
    verdict: VerdictStruct

class ReasoningSampleDict(TypedDict):
    project: str
    cwe: list[str]
    target: int
    func: str
    cwe_desc: list[str]
    reasoning: str


@dataclass
class ReasoningSample:
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
    reasoning_tokens: int
    answer_tokens: int 
    assistant_tokens: int
    total_tokens: int

    @property
    def to_dict(self):
        return self.__dict__


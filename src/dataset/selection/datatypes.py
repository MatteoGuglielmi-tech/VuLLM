from dataclasses import dataclass
from typing import Any

ReasoningSampleDict = dict[str, Any]

@dataclass
class ReasoningSample:
    """Container for reasoning samples"""

    project: str
    cwe: list[str]
    target: int
    func: str
    cwe_desc: list[str]
    reasoning: str

    @property
    def to_dict(self) -> ReasoningSampleDict:
        return self.__dict__

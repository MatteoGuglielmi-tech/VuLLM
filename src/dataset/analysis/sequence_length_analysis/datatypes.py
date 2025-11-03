from dataclasses import dataclass


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
    def to_dict(self):
        return self.__dict__

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


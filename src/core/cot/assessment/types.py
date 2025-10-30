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
    sample_id: str|None = None

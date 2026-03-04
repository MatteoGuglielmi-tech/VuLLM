from typing import NotRequired, TypedDict, TypeAlias

CWEId: TypeAlias = int
UniqueCWEs: TypeAlias = set[int]


class Sample(TypedDict):
    func: str
    target: int
    cwe: list[str]
    cwe_desc: list[str]
    project: str
    reasoning: NotRequired[str]

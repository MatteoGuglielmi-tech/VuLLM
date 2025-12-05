from typing import TypeVar, Generic, Iterator, TypedDict
from dataclasses import dataclass, fields
from datasets import Dataset

T = TypeVar("T")

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


class TestDatasetSchema(TypedDict):
    func: str
    target: int
    cwe: list[str]
    model_prediction: str


class TypedDataset(Generic[T]):
    """
    Generic typed wrapper around HuggingFace Dataset.

    Type Parameters
    ---------------
    T : TypedDict
        Schema defining the dataset columns

    Examples
    --------
    >>> schema = TypedDict('Schema', {'text': str, 'label': int})
    >>> dataset = TypedDataset[schema](raw_dataset)
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> T:
        return self._dataset[idx]  # type: ignore[return-value]

    def __iter__(self) -> Iterator[T]:
        for sample in self._dataset:
            yield sample  # type: ignore[misc]

    @property
    def raw(self) -> Dataset:
        """Access underlying Dataset."""
        return self._dataset


class GenerationError(Exception):
    """Exception raised whenever generation fails.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

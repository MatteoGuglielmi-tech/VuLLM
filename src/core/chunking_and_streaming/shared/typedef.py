from typing import Any, TypeAlias, TypedDict
from datasets import IterableDataset


# <---- chunking.py ---->
class IntermediateChunk(TypedDict):
    text: str # combined text
    tcount: int

class FinalChunk(TypedDict):
    function_signature: str
    label: str
    text: str # The final, combined text for the model
    tcount: int

# <---- dataset.py ---->
StreamedSplit: TypeAlias = dict[str, IterableDataset]
ChatMsg: TypeAlias = list[dict[str, str]]
DatasetEntry: TypeAlias = dict[str,Any]
JsonlData: TypeAlias = list[DatasetEntry]


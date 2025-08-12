from typing import Any, TypeAlias, TypedDict
from tree_sitter import Node
from datasets import IterableDataset


# <---- tree_sitter_parser.py ---->
TSNode: TypeAlias = Node | None
Capture: TypeAlias = list[Node]|None
Captures: TypeAlias = dict[str,list[Node]]

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


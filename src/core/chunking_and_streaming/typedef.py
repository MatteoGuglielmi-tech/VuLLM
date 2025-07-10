import sys
from typing import TypedDict, TypeAlias


from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from tree_sitter import Node


# ====== split_large_chunk() ======
class SubChunk(TypedDict):
    context: str
    text: str
    context_tcount: int
    text_tcount: int
    nodes: list[Node]


# ====== semantic_sliding_window() ======
class Window(TypedDict):
    function_signature: str
    label: str
    lang: str
    parent: str
    text: str
    inner_context: str
    trailing_context: str


# ====== level_based_split() ======
class LBChunk(TypedDict):
    parent: str
    text: str
    inner_context: str
    trailing_context: str


class ASTChunk(TypedDict):
    function_signature: str
    label: str
    lang: str
    parent: str
    text: str
    inner_context: str
    trailing_context: str


# if sys.version_info.minor >= 12:
#     type SubChunks = list[SubChunk]
#     type Windows = list[Window]
#     type LBChunks = list[LBChunk]
#     type ASTChunks = list[ASTChunk]
#     type TSNode = Node | None
#     type StreamingDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset
#     type StreamedSplit = dict[str, StreamingDataset]
# else:
SubChunks = list[SubChunk]
Windows = list[Window]
LBChunks = list[LBChunk]
ASTChunks = list[ASTChunk]
TSNode: TypeAlias = Node | None
StreamingDataset: TypeAlias = (
    DatasetDict | Dataset | IterableDatasetDict | IterableDataset
)

StreamedSplit = dict[str, StreamingDataset]


# type ChatMsg = list[dict[str, str]] # python -V >=3.12
ChatMsg = list[dict[str, str]]

if __name__ == "__main__":
    fixed_metadata: LBChunk = LBChunk(**{k: "" for k in LBChunk.__annotations__.keys()})
    print(fixed_metadata)

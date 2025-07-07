from typing import TypedDict

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from tree_sitter import Node


# ====== split_large_chunk() ======
class SubChunk(TypedDict):
    context: str
    text: str
    context_tcount: int
    text_tcount: int
    nodes: list[Node]


type SubChunks = list[SubChunk]


# ====== semantic_sliding_window() ======
class Window(TypedDict):
    function_signature: str
    label: str
    lang: str
    parent: str
    text: str
    inner_context: str
    trailing_context: str


type Windows = list[Window]


# ====== level_based_split() ======
class LBChunk(TypedDict):
    parent: str
    text: str
    inner_context: str
    trailing_context: str


type LBChunks = list[LBChunk]


class ASTChunk(TypedDict):
    function_signature: str
    label: str
    lang: str
    parent: str
    text: str
    inner_context: str
    trailing_context: str


type ASTChunks = list[ASTChunk]

type TSNode = Node | None

type StreamingDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset
type StreamedSplit = dict[str, StreamingDataset]

if __name__ == "__main__":
    fixed_metadata: LBChunk = LBChunk(**{k: "" for k in LBChunk.__annotations__.keys()})
    print(fixed_metadata)

from typing import TypeAlias
from tree_sitter import Node

# <---- tree_sitter_parser.py ---->
TSNode: TypeAlias = Node | None
Capture: TypeAlias = list[Node]|None
Captures: TypeAlias = dict[str,list[Node]]

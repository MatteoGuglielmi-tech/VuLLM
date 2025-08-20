import logging
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter_extended_c import LANGUAGE as EXT_C_LANG
from tree_sitter import Language, Parser, Tree, Node, Query, QueryCursor

from collections.abc import Generator

from .common_typedef import Captures


logger = logging.getLogger(name=__name__)
SUPPORTED_LANGS = { "c": Language(tsc.language()), "cpp": Language(tscpp.language()), "ext_c": EXT_C_LANG }


class TreeSitterParser:
    def __init__(self, language_name: str = "c") -> None:
        self.language_name: str = language_name
        match(language_name):
            case "c": self.language = SUPPORTED_LANGS[language_name]
            case "cpp": self.language = SUPPORTED_LANGS[language_name]
            case _: self.language = EXT_C_LANG

        self.parser = Parser(self.language)

    def parse(self, code: str|bytes) -> Tree:
        """Parse a codebase.
        Parameters
        ----------
        code: str|bytes
            Represents the string representation of the code to be parsed.
        Returns
        -------
            None
        """

        src = code.encode(encoding="utf-8") if isinstance(code, str) else code
        tree: Tree = self.parser.parse(src)

        return tree

    def traverse_tree(self, node: Node) -> Generator[Node, None, None]:
        """Performs a pre-order (DFS) traversal of the syntax tree from a given node.
        Yields each node in the tree, starting from `node`.

        Yields:
            Node: The next node in the traversal.
        """
        yield node
        for child in node.children:
            yield from self.traverse_tree(node=child)


    def query(self, code:str, query_str:str) -> Captures:
        """Make a query to an AST given by TreeSitter."""
        query:Query = Query(self.language, query_str)
        return QueryCursor(query).captures(self.parse(code).root_node)

    def run_query_on_tree(self, tree: Tree, query_str: str) -> Captures:
        """Runs a query on a pre-parsed tree."""
        query:Query = Query(self.language, query_str)
        return QueryCursor(query).captures(tree.root_node)

    def run_query_on_node(self, node: Node, query_str: str) -> Captures:
        """Runs a query on a pre-parsed tree."""
        query:Query = Query(self.language, query_str)
        return QueryCursor(query).captures(node)

    def has_error_nodes(self, tree: Tree) -> bool:
        return any(node.is_error for node in self.traverse_tree(node=tree.root_node))

    def has_missing_nodes(self, tree: Tree) -> bool:
        return any(node.is_missing for node in self.traverse_tree(node=tree.root_node))

    def is_broken_tree(self, tree:Tree) -> bool:
        return self.has_error_nodes(tree=tree) or self.has_missing_nodes(tree=tree)


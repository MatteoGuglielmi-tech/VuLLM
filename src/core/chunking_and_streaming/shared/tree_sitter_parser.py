from collections.abc import Generator

from tree_sitter import Language, Parser, Tree, Node, Query, QueryCursor
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from .stdout import MY_LOGGER_NAME
from .typedef import TSNode, Captures


import logging
logger = logging.getLogger(MY_LOGGER_NAME)


C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())
SUPPORTED_LANGS = { "c": C_LANGUAGE, "cpp": CPP_LANGUAGE }

class TreeSitterParser:
    def __init__(self, language_name: str = "c") -> None:
        language:Language|None = SUPPORTED_LANGS.get(language_name)

        if language is None:
            language = C_LANGUAGE
            logger.warning(f"Unsupported language: {language_name}\nFalling back to 'C'")

        self.language: Language = language
        self.parser = Parser(self.language)

    def parse(self, code: str|bytes) -> Tree:
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
            yield from self.traverse_tree(child) 

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

    def print_ast(self, node: TSNode, indent=""):
        """Recursively prints the AST to reveal the true node types."""

        assert node is not None
        node_type = node.type
        node_text = node.text.decode("utf-8").strip().split("\n")[0] if node.text else ""
        print(f'{indent}Type: `{node_type}` --- Text: "{node_text}..."')
        for child in node.children:
            self.print_ast(child, indent + "  ")

    def has_error_nodes(self, tree: Tree) -> bool:
        return any(node.is_error for node in self.traverse_tree(tree.root_node))

    def has_missing_nodes(self, tree: Tree) -> bool:
        return any(node.is_missing for node in self.traverse_tree(tree.root_node))

    def is_broken_tree(self, tree:Tree) -> bool:
        return self.has_error_nodes(tree) or self.has_missing_nodes(tree)


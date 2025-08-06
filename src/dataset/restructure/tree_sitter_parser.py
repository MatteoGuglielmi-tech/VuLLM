from collections.abc import Generator

from tree_sitter import Language, Parser, Tree, TreeCursor, Node, Query, QueryCursor
import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp

TSNode = Node | None
Nodes = list[Node]

C_LANGUAGE = Language(tsc.language())
CPP_LANGUAGE = Language(tscpp.language())

c_parser = Parser(C_LANGUAGE)
cpp_parser = Parser(CPP_LANGUAGE)

class TreeSitterParser:
    def __init__(self, language_name: str = "c") -> None:
        self.language_name: str = language_name
        self.language : Language = C_LANGUAGE if language_name == "c" else CPP_LANGUAGE
        self.parser = Parser(self.language)

        self._is_encoded: bool = False
        self._is_parsed: bool = False

    def parse(self, code: str|bytes) -> Tree:
        """Parse a codebase.
        Params:
            code: str|bytes
                Represents the string representation of the code to be parsed.
        Returns:
            None
        """

        src = (
            code.encode(encoding="utf-8") if isinstance(code, str)
            else code
        )

        tree: Tree = self.parser.parse(src)

        return tree

    def traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
        """Performs a DFS traversal of the syntax tree.
        Yields each node in the tree, starting from the root.

        Yields:
            Node: next node in the pre-order traversal.
        """

        cursor: TreeCursor = tree.walk()
        reached_root: bool = False
        while not reached_root:
            assert cursor.node is not None, "Cursor pointed to a null node during traversal"
            yield cursor.node
            # 1. try to move down to the first child.
            if cursor.goto_first_child(): continue
            # 2. no child, try to move to the next sibling.
            if cursor.goto_next_sibling(): continue
            # 3. If there are no children or siblings, move up and look for other branches
            while True:
                if not cursor.goto_parent():
                    reached_root = True
                    break
                if cursor.goto_next_sibling():
                    break

    def query(self, code:str, query_str:str) -> dict[str,list[Node]]:
        """Make a query to an AST given by TreeSitter."""

        function_query:Query = Query(self.language, query_str)
        return QueryCursor(function_query).captures(self.parse(code).root_node)


    def print_ast(self, node: TSNode, indent=""):
        """Recursively prints the AST to reveal the true node types."""

        assert node is not None
        node_type = node.type
        node_text = node.text.decode("utf-8").strip().split("\n")[0] if node.text else ""
        print(f'{indent}Type: `{node_type}` --- Text: "{node_text}..."')
        for child in node.children:
            self.print_ast(child, indent + "  ")

    def has_error_nodes(self, tree: Tree) -> bool:
        return any(node.is_error for node in self.traverse_tree(tree))

    def has_missing_nodes(self, tree: Tree) -> bool:
        return any(node.is_missing for node in self.traverse_tree(tree))

    def is_broken_tree(self, tree:Tree) -> bool:
        return self.has_error_nodes(tree) or self.has_missing_nodes(tree)


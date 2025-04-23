import re
from collections.abc import Generator

from tree_sitter import Language, Node, Parser, Tree, TreeCursor
from tree_sitter_language_pack import (SupportedLanguage, get_language,
                                       get_parser)


class TreeSitter:
    def __init__(self, language_name: SupportedLanguage = "c") -> None:
        self.language: Language = get_language(language_name=language_name)
        self.parser: Parser = get_parser(language_name=language_name)

    def __post_init__(self) -> None:
        self._is_encoded: bool = False
        self._is_parsed: bool = False

    def parse_input(self, code_snippet: str | bytes) -> None:
        self._is_encoded = True
        self._is_parsed = True
        src = (
            code_snippet.encode(encoding="utf-8")
            if isinstance(code_snippet, str)
            else code_snippet
        )

        src = re.sub(pattern=b"\\\\n", repl=b"\n", string=src)

        self.tree: Tree = self.parser.parse(src)
        self.root_node = self.tree.root_node

    def extract_comments(self) -> list[bytes]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        comments: list[bytes] = [
            self._assert_not_none(attribute=cmnt.text, attr_name="cmnt.text")
            for cmnt in self.traverse_tree()
            if cmnt.type == "comment"
        ]

        return comments

    def get_missing_nodes(self) -> list[Node]:
        lom = []
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        for node in self.traverse_tree():
            if node.is_missing:
                # technically speaking, this shouldn't be an option
                assert node is not None
                lom.append(node)

        return lom

    def get_error_nodes(self) -> list[Node]:
        loe = []
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        for node in self.traverse_tree():
            if node.is_error:
                # technically speaking, this shouldn't be an option
                assert node is not None
                loe.append(node)

        return loe

    def extract_directives(self) -> list[bytes]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        directives: list[bytes] = [
            self._assert_not_none(attribute=drctv.text, attr_name="drctv.text")
            for drctv in self.traverse_tree()
            if (
                drctv.type == "preproc_def"
                or drctv.type == "preproc_function_def"
                or drctv.type == "preproc_if"
                or drctv.type == "preproc_ifdef"
                or drctv.type == "preproc_directive"
                # or drctv.type == "preproc_else"
                # or drctv.type == "preproc_elif"
            )
        ]

        return directives

    def query(self, query_str: str) -> dict[str, list[Node]]:
        query = self.tree.language.query(query_str)
        captures = query.captures(self.root_node)

        return captures

    def is_closing_curvy_needed(self) -> bool:
        missing = self.get_missing_nodes()
        function_node: Node = self.root_node.children[0]

        if missing:
            # check for missing "}" based on node position
            if (
                (function_node.end_point == missing[-1].end_point)
                and (missing[-1].start_point == missing[-1].end_point)
                and missing[-1].type == "}"
            ):
                return True

        return False

    def reset_is_parsed(self) -> None:
        self._is_parsed = False

    def set_is_parsed(self) -> None:
        self._is_parsed = True

    def reset_is_encoded(self) -> None:
        self._is_encoded = False

    def set_is_encoded(self) -> None:
        self._is_encoded = True

    def traverse_tree(
        self,
    ) -> Generator[Node, None, None]:
        cursor: TreeCursor = self.tree.walk()
        visited_children = False

        while True:
            if not visited_children:
                # This tells both the linter and type checker cusors.node will never be None
                assert cursor.node is not None, "Node not found (Null)"
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    def _assert_not_none(self, attribute: bytes | None, attr_name: str) -> bytes:
        assert attribute is not None, f"Attribute {attr_name} is None"

        return attribute

    def replace_error_nodes(self, src: str, target: str = "") -> str:

        re_d: dict[int, re.Pattern] = {
            1: re.compile(pattern=r"#\s*else"),
            2: re.compile(pattern=r"#\s*elif"),
            3: re.compile(pattern=r"#\s*endif"),
        }

        error_nodes: list[Node] = self.get_error_nodes()
        # this excludes the "translation_unit" voice
        func_node = self.root_node.child(0)
        if func_node is None:
            return src

        first_node: Node | None = func_node.child(0)
        second_node: Node | None = func_node.child(1)

        if first_node is None:
            return src

        def __is_if():
            return (
                True
                if (
                    (first_node.text == b"#if")
                    or (first_node.text == b"#ifndef")
                    or (first_node.text == b"#ifdef")
                )
                else False
            )

        if error_nodes:
            if __is_if() and first_node and second_node:
                src = (
                    src.replace(
                        b" ".join([first_node.text, second_node.text]).decode("utf-8"),
                        target,
                    )
                    if (first_node.text is not None and second_node.text is not None)
                    else src
                )
            else:
                for v in re_d.values():
                    src = re.sub(pattern=v, repl="", string=src)

        return src


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="rb") as f:
        file_content: bytes = f.read()

    return file_content.decode(encoding="utf-8")


def test():
    ts: TreeSitter = TreeSitter()

    code = read_file_content_as_str(filepath="tmp.c")

    ts.parse_input(code_snippet=code)
    print(ts.is_closing_curvy_needed())

    error_nodes = ts.get_error_nodes()
    print(error_nodes)
    print(ts.replace_error_nodes(code))
    print(ts.get_missing_nodes())

    comments: list[bytes] = ts.extract_comments()
    directives: list[bytes] = ts.extract_directives()

    print(directives)

    for comment in comments:
        str_cmnt = comment.decode(encoding="utf-8")
        code = code.replace(str_cmnt, "")

    print(code)


if __name__ == "__main__":
    test()

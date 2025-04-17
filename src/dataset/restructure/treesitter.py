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

    def extract_comments(self) -> list[bytes]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        comments: list[bytes] = [
            self._assert_not_none(attribute=cmnt.text, attr_name="cmnt.text")
            for cmnt in self.traverse_tree(tree=self.tree)
            if cmnt.type == "comment"
        ]

        return comments

    def check_missing_nodes(self) -> tuple[bool, str]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        for node in self.traverse_tree(tree=self.tree):
            if node.is_missing:
                return (True, f"{node}")

        return (False, "")

    def extract_directives(self) -> list[bytes]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        directives: list[bytes] = [
            self._assert_not_none(attribute=drctv.text, attr_name="drctv.text")
            for drctv in self.traverse_tree(tree=self.tree)
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

    def reset_is_parsed(self) -> None:
        self._is_parsed = False

    def set_is_parsed(self) -> None:
        self._is_parsed = True

    def reset_is_encoded(self) -> None:
        self._is_encoded = False

    def set_is_encoded(self) -> None:
        self._is_encoded = True

    def traverse_tree(self, tree: Tree) -> Generator[Node, None, None]:
        cursor: TreeCursor = tree.walk()
        visited_children = False

        while True:
            if not visited_children:
                # This tells both the linter and type checker cusors.node
                # will never be None
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


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="rb") as f:
        file_content: bytes = f.read()

    return file_content.decode(encoding="utf-8")


def test():
    ts: TreeSitter = TreeSitter()

    code = read_file_content_as_str(filepath="tmp.c")
    print(code)

    ts.parse_input(code_snippet=code)
    ts.check_missing_nodes()
    comments: list[bytes] = ts.extract_comments()
    directives: list[bytes] = ts.extract_directives()

    print(comments)

    print("\n")
    print(directives)

    for comment in comments:
        str_cmnt = comment.decode(encoding="utf-8")
        code = code.replace(str_cmnt, "")

    print(code)


if __name__ == "__main__":
    test()

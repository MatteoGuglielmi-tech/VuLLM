import re
from collections.abc import Generator
from os import error

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

    def replace_error_nodes(self, src: str) -> str:
        to_rm: str = ""

        is_at_beginning: re.Match[str] | None = re.match(pattern=r"\s*#if", string=src)
        if_dir: list[str] = re.findall(pattern=r"#if", string=src)

        end_dir_re: re.Pattern = re.compile(pattern=r"(#end|#else|#el)if")
        end_dir = re.findall(pattern=end_dir_re, string=src)

        # check if there is something to do
        # if there are no conditional directives or
        # if there is a properly formed condition
        if not (if_dir or end_dir) or (if_dir and end_dir):
            return src

        error_nodes: list[Node] = self.get_error_nodes()

        if not error_nodes:
            return src

        if if_dir:
            if is_at_beginning:
                err = error_nodes[0]
                assert err is not None
                assert err.text is not None
                assert err.children is not None
                assert err.children[0].text is not None
                assert err.children[0].next_sibling is not None
                assert err.children[0].next_sibling.text is not None

                to_rm = " ".join(
                    [
                        err.children[0].text.decode(),
                        err.children[0].next_sibling.text.decode(),
                    ]
                )

            for err in error_nodes:
                assert err is not None
                assert err.text is not None
                if (
                    (err.text == b"#if")
                    or (err.text == b"#ifndef")
                    or (err.text == b"#ifdef")
                ):
                    assert err.next_sibling is not None
                    assert err.next_sibling.text is not None

                    to_rm = " ".join(
                        [err.text.decode(), err.next_sibling.text.decode()]
                    )
            src = src.replace(to_rm, "")

        elif end_dir:
            src = re.sub(pattern=end_dir_re, repl="", string=src)

        return src


def read_file_content_as_str(filepath: str) -> str:
    with open(file=filepath, mode="rb") as f:
        file_content: bytes = f.read()

    return file_content.decode(encoding="utf-8")


def test():
    ts: TreeSitter = TreeSitter()

    code = read_file_content_as_str(filepath="tmp.c")

    ts.parse_input(code_snippet=code)
    # print(ts.is_closing_curvy_needed())

    error_nodes = ts.get_error_nodes()
    for err in error_nodes:
        if (err.text == b"#if") or (err.text == b"#ifndef") or (err.text == b"#ifdef"):
            assert err.text is not None
            assert err.next_sibling is not None and err.next_sibling.text
            print(" ".join([err.text.decode(), err.next_sibling.text.decode()]))
    print(ts.replace_error_nodes(code))
    # print(ts.get_missing_nodes())

    # comments: list[bytes] = ts.extract_comments()
    # directives: list[bytes] = ts.extract_directives()

    # print(directives)

    # for comment in comments:
    #     str_cmnt = comment.decode(encoding="utf-8")
    #     code = code.replace(str_cmnt, "")
    #
    # print(code)


if __name__ == "__main__":
    test()

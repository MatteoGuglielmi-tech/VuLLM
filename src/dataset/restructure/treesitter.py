import re
from collections.abc import Generator

from tree_sitter import Language, Node, Parser, Tree, TreeCursor
from tree_sitter_language_pack import (SupportedLanguage, get_language,
                                       get_parser)


class TreeSitter:
    def __init__(self, language_name: SupportedLanguage = "c") -> None:
        self.language_name: SupportedLanguage = language_name
        self.language: Language = get_language(language_name=language_name)
        self.parser: Parser = get_parser(language_name=language_name)
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

    def _check_is_parsed(self) -> None:
        if not self._is_parsed:
            raise Exception("Missing call to parse function! Abort...")

    def _check_is_encoded(self) -> None:
        if not self._is_encoded:
            raise Exception("Missing call to parse function! Abort...")

    def _check_is_encoded_and_parsed(self) -> None:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

    def extract_comments(self) -> list[bytes]:
        self._check_is_encoded_and_parsed()

        comments: list[bytes] = [
            self._assert_not_none(attribute=cmnt.text, attr_name="cmnt.text")
            for cmnt in self.traverse_tree()
            if cmnt.type == "comment"
        ]

        return comments

    def get_missing_nodes(self) -> list[Node]:
        lom = []
        self._check_is_encoded_and_parsed()

        for node in self.traverse_tree():
            if node.is_missing:
                # technically speaking, this shouldn't be an option
                assert node is not None
                lom.append(node)

        return lom

    def get_error_nodes(self) -> list[Node]:
        loe = []
        self._check_is_encoded_and_parsed()

        for node in self.traverse_tree():
            if node.is_error:
                # technically speaking, this shouldn't be an option
                assert node is not None
                loe.append(node)

        return loe

    def extract_directives(self) -> list[bytes]:
        self._check_is_encoded_and_parsed()

        directives: list[bytes] = [
            self._assert_not_none(attribute=drctv.text, attr_name="drctv.text")
            for drctv in self.traverse_tree()
            if (
                drctv.type == "preproc_def"
                or drctv.type == "preproc_function_def"
                or drctv.type == "preproc_if"
                or drctv.type == "preproc_ifdef"
                or drctv.type == "preproc_directive"
                or drctv.type == "preproc_else"
                or drctv.type == "preproc_elif"
            )
        ]

        return directives

    def query(self, query_str: str) -> dict[str, list[Node]]:
        self._check_is_encoded_and_parsed()

        query = self.tree.language.query(query_str)
        captures = query.captures(self.root_node)

        return captures

    def is_closing_curvy_needed(self) -> bool:
        self._check_is_encoded_and_parsed()

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

    def _setBit(self, int_type: int, offset: int):
        mask = 1 << offset
        return int_type | mask

    def _clearBit(self, int_type: int, offset: int):
        mask = ~(offset)
        return int_type & mask

    def remove_if_condition(self, src: str) -> str:
        is_at_beginning: re.Match[str] | None = re.match(
            pattern=r"\s*#\s*if", string=src
        )
        if_dir: list[str] = re.findall(pattern=r"\s*#\s*if", string=src)

        if not if_dir:
            return src

        flags: int = 0
        count: int = 1
        entry_to_rm: list[str] = []

        if_comp_str: str = ""
        bin_exp: str = ""

        self.parse_input(code_snippet=src)
        print(f"Language used for parsing: {self.language_name}")

        if if_dir:
            for n in self.traverse_tree():
                assert n.text is not None
                match n.type:
                    case "#if":
                        if_comp_str = ""
                        flags |= self._setBit(int_type=flags, offset=0)
                        if_comp_str = n.text.decode() + " "

                    case "binary_expression":
                        if (flags & 0x1):
                            bin_exp = n.text.decode(encoding="utf-8")
                            flags |= self._setBit(int_type=flags, offset=count)
                            count += 1
                        else:
                            flags = self._clearBit(int_type=flags, offset=0xFF)

                    case "identifier":
                        if flags == 0x3:
                            if_comp_str += re.findall(
                                pattern=rf"{n.text.decode()}\s*", string=src
                            )[0]
                            flags |= self._setBit(int_type=flags, offset=count)
                            count += 1
                        else:
                            flags = self._clearBit(int_type=flags, offset=0xFF)

                    case "<" | ">" | "==" | ">=" | "<=":
                        if flags == 0x7:
                            if_comp_str += re.findall(
                                pattern=rf"{n.text.decode()}\s*", string=bin_exp
                            )[0]
                            flags |= self._setBit(int_type=flags, offset=count)
                            count += 1
                        else:
                            flags = self._clearBit(int_type=flags, offset=0xFF)

                    # an ERROR node is create when #if condition is
                    # at the beginning and only in case of cpp parser
                    case "ERROR":
                        if is_at_beginning and self.language_name == "cpp"  and flags == 0xF:
                            flags |= self._setBit(int_type=flags, offset=count)
                            count += 1
                            continue
                        else:
                            flags = self._clearBit(int_type=flags, offset=0xFF)

                    case "number_literal":
                        if (flags == 0xF) or (flags== 0x1F):
                            if_comp_str += re.findall(
                                pattern=rf"{n.text.decode()}\s*", string=src
                            )[0]

                            entry_to_rm.append(if_comp_str)
                            flags = self._clearBit(int_type=flags, offset=0xFF)
                        else:
                            flags = self._clearBit(int_type=flags, offset=0xFF)

                    case _:
                        if_comp_str = ""
                        flags = self._clearBit(int_type=flags, offset=0xFF)

        for s in entry_to_rm:
            src = src.replace(s, "")

        return src

    def replace_error_nodes(self, src: str) -> str:
        to_rm: str = ""
        idx: int = 0

        is_at_beginning: re.Match[str] | None = re.match(
            pattern=r"\s*#\s*if", string=src
        )
        if_dir: list[str] = re.findall(pattern=r"\s*#\s*if", string=src)
        else_dir_re: re.Pattern = re.compile(pattern=r"\s*#\s*else")
        else_dir: list[str] = re.findall(pattern=else_dir_re, string=src)

        end_dir_re: re.Pattern = re.compile(pattern=r"#\s*(?:end|el)if")
        end_dir = re.findall(pattern=end_dir_re, string=src)

        # check if there is something to do
        if not (if_dir or end_dir or else_dir) or (len(if_dir) == len(end_dir)):
            return src

        self.parse_input(code_snippet=src)
        error_nodes: list[Node] = self.get_error_nodes()

        if if_dir and error_nodes:
            while True:
                assert error_nodes[idx] is not None
                assert error_nodes[idx].text is not None
                if is_at_beginning:
                    err = error_nodes[0]
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

                    src = src.replace(to_rm, "")
                    is_at_beginning = re.match(pattern=r"\s*#if", string=src)

                    self.parse_input(code_snippet=src)
                    error_nodes = self.get_error_nodes()
                    if not error_nodes:
                        break
                else:
                    if self.language_name == "cpp":
                        if (
                            (error_nodes[idx].children[0].text == b"#if")
                            or (error_nodes[idx].children[0].text == b"#ifndef")
                            or (error_nodes[idx].children[0].text == b"#ifdef")
                        ):

                            assert error_nodes[idx].children[0].text is not None
                            assert error_nodes[idx].children[1].text is not None

                            to_rm = " ".join(
                                [
                                    error_nodes[idx].children[0].text.decode(),
                                    error_nodes[idx].children[1].text.decode(),
                                ]
                            )
                    else:
                        if (
                            (error_nodes[idx].text == b"#if")
                            or (error_nodes[idx].text == b"#ifndef")
                            or (error_nodes[idx].text == b"#ifdef")
                        ):
                            assert error_nodes[idx].next_sibling is not None
                            assert error_nodes[idx].next_sibling.text is not None

                            to_rm = " ".join(
                                [
                                    error_nodes[idx].text.decode(),
                                    error_nodes[idx].next_sibling.text.decode(),
                                ]
                            )
                    src = src.replace(to_rm, "")

                    if idx == len(error_nodes) - 1:
                        break

                    idx += 1
        elif end_dir:
            src = re.sub(pattern=end_dir_re, repl="", string=src)
            src = re.sub(pattern=else_dir_re, repl="", string=src)

        # else:
        elif if_dir and self.is_endif_missing():
            if is_at_beginning:
                err = self.root_node.children[0]
                assert err.children[0].text is not None
                assert err.child_by_field_name("name") is not None

                to_rm = " ".join(
                    [
                        err.children[0].text.decode(),
                        err.child_by_field_name("name").text.decode(),
                    ]
                )
                src = src.replace(to_rm, "")

            else:
                if self.language_name == "cpp":
                    err = self.root_node.children[1]

                    assert err.children[0].text is not None
                    assert err.child_by_field_name("name") is not None
                    to_rm = " ".join(
                        [
                            err.children[0].text.decode(),
                            err.child_by_field_name("name").text.decode(),
                        ]
                    )

                src = src.replace(to_rm, "")

        return src

    def is_valid_namespace(self, proto: str) -> bool:
        # append dummy parenthesis for proper parsing
        if self.language_name == "cpp":
            proto = proto[:-1] + "{}"
            self.parse_input(code_snippet=proto)
            nodes: list[Node] = self.root_node.children
            first_node: Node = nodes[0]

            if first_node.type == "namespace_definition":
                return True

        return False

    def is_valid_function(self, proto: str) -> bool:
        # append dummy parenthesis for proper parsing
        proto = proto[:-1] + "{}"
        self.parse_input(code_snippet=proto)
        nodes: list[Node] = self.root_node.children
        first_node: Node = nodes[0]

        if first_node.type == "function_definition":
            return True

        # in cpp not issues of this type
        if self.language_name == "c":
            # in case of C language, a function prototype without primitive_type
            # leads to either:
            # 1. an "ERROR" node -> go to children and verify if it's recognized as "macro_type_specifier" or "call_expression".
            # 2. "expression_statement" node -> get first node of the dummy proto and see if "function_definition" node is there

            # int because is the default type
            proto = "int " + proto
            self.parse_input(code_snippet=proto)
            nodes: list[Node] = self.root_node.children

            if (
                first_node.type == "ERROR"
                and (
                    first_node.children[0].type == "macro_type_specifier"
                    # call no params
                    or first_node.children[0].type == "call_expression"
                )
                # call with params
                or first_node.type == "expression_statement"
            ):
                # update first node based on "dummy" func tree
                first_node: Node = nodes[0]
                if first_node.type == "function_definition":
                    return True

        return False

    def is_valid_template(self, proto: str) -> bool:
        if self.language_name == "cpp":
            self.parse_input(code_snippet=proto)
            nodes: list[Node] = self.root_node.children
            first_node: Node = nodes[0]
            if first_node.type == "template_declaration":
                return True

        return False

    def is_endif_missing(self) -> bool:
        missing_nodes: list[Node] = self.get_missing_nodes()
        for mn in missing_nodes:
            assert mn is not None
            if mn.type == "#endif":
                return True

        return False


# exported vars
c_ts: TreeSitter = TreeSitter(language_name="c")
cpp_ts: TreeSitter = TreeSitter(language_name="cpp")

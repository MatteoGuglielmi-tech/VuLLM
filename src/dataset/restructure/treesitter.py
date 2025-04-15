import re
from collections.abc import Generator
from typing import cast

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
            # assert cmnt.text is not None
            # cmnt.text
            for cmnt in self.traverse_tree(tree=self.tree)
            if cmnt.type == "comment"
        ]

        return comments

    def extract_directives(self) -> list[bytes]:
        if not (self._is_parsed or self._is_encoded):
            raise Exception("Missing call to parse function! Abort...")

        directives: list[bytes] = [
            self._assert_not_none(attribute=drctv.text, attr_name="drctv.text")
            # cast(bytes, drctv.text)
            for drctv in self.traverse_tree(tree=self.tree)
            if (
                drctv.type == "preproc_def"
                or drctv.type == "preproc_if"
                or drctv.type == "preproc_ifdef"
                or drctv.type == "preproc_directive"
                or drctv.type == "preproc_else"
                or drctv.type == "preproc_elif"
                or drctv.type == "preproc_function_def"
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


def test():
    ts: TreeSitter = TreeSitter()

    code = 'static int read_and_execute(bool interactive)\n{\n#if defined(_WIN32)\n String tmpbuf;\n String buffer;\n#endif\n\n /*\n line can be allocated by:\n - batch_readline. Use my_free()\n - my_win_console_readline. Do not free, see tmpbuf.\n - readline. Use free()\n */\n char *line= NULL;\n char in_string=0;\n ulong line_number=0;\n bool ml_comment= 0; \n COMMANDS *com;\n ulong line_length= 0;\n status.exit_status=1;\n\n real_binary_mode= !interactive && opt_binary_mode;\n for (;;)\n {\n /* Reset as SIGINT has already got handled. */\n sigint_received= 0;\n\n if (!interactive)\n {\n /*\n batch_readline can return 0 on EOF or error.\n In that case, we need to double check that we have a valid\n line before actually setting line_length to read_length.\n */\n line= batch_readline(status.line_buff, real_binary_mode);\n if (line) \n {\n line_length= status.line_buff->read_length;\n\n /*\n ASCII 0x00 is not allowed appearing in queries if it is not in binary\n mode.\n */\n if (!real_binary_mode && strlen(line) != line_length)\n {\n status.exit_status= 1;\n String msg;\n msg.append("ASCII \'\\\\0\' appeared in the statement, but this is not "\n "allowed unless option --binary-mode is enabled and mysql is "\n "run in non-interactive mode. Set --binary-mode to 1 if ASCII "\n "\'\\\\0\' is expected. Query: \'");\n msg.append(glob_buffer);\n msg.append(line);\n msg.append("\'.");\n put_info(msg.c_ptr(), INFO_ERROR);\n break;\n }\n\n /*\n Skip UTF8 Byte Order Marker (BOM) 0xEFBBBF.\n Editors like "notepad" put this marker in\n the very beginning of a text file when\n you save the file using "Unicode UTF-8" format.\n */\n if (!line_number &&\n (uchar) line[0] == 0xEF &&\n (uchar) line[1] == 0xBB &&\n (uchar) line[2] == 0xBF)\n {\n line+= 3;\n // decrease the line length accordingly to the 3 bytes chopped\n line_length -=3;\n }\n }\n line_number++;\n if (!glob_buffer.length())\n status.query_start_line=line_number;\n }\n else\n {\n char *prompt= (char*) (ml_comment ? " /*> " :\n glob_buffer.is_empty() ? construct_prompt() :\n !in_string ? " -> " :\n in_string == \'\\\'\' ?\n " \'> " : (in_string == \'`\' ?\n " `> " :\n " \\"> "));\n if (opt_outfile && glob_buffer.is_empty())\n fflush(OUTFILE);\n\n#if defined(_WIN32)\n size_t nread;\n tee_fputs(prompt, stdout);\n if (!tmpbuf.is_alloced())\n tmpbuf.alloc(65535);\n tmpbuf.length(0);\n buffer.length(0);\n line= my_win_console_readline(charset_info,\n (char *) tmpbuf.ptr(),\n tmpbuf.alloced_length(),\n &nread);\n if (line && (nread == 0))\n {\n tee_puts("^C", stdout);\n reset_prompt(&in_string, &ml_comment);\n continue;\n }\n else if (*line == 0x1A) /* (Ctrl + Z) */\n break;\n#else\n if (opt_outfile)\n fputs(prompt, OUTFILE);\n /*\n free the previous entered line.\n */\n if (line)\n free(line);\n line= readline(prompt);\n\n if (sigint_received)\n {\n sigint_received= 0;\n tee_puts("^C", stdout);\n reset_prompt(&in_string, &ml_comment);\n continue;\n }\n#endif /* defined(_WIN32) */\n /*\n When Ctrl+d or Ctrl+z is pressed, the line may be NULL on some OS\n which may cause coredump.\n */\n if (opt_outfile && line)\n fprintf(OUTFILE, "%s\\n", line);\n\n line_length= line ? strlen(line) : 0;\n }\n // End of file or system error\n if (!line)\n {\n if (status.line_buff && status.line_buff->error)\n status.exit_status= 1;\n else\n status.exit_status= 0;\n break;\n }\n\n /*\n Check if line is a mysql command line\n (We want to allow help, print and clear anywhere at line start\n */\n if ((named_cmds || glob_buffer.is_empty())\n && !ml_comment && !in_string && (com= find_command(line)))\n {\n if ((*com->func)(&glob_buffer,line) > 0)\n {\n // lets log the exit/quit command.\n if (interactive && status.add_to_history && com->cmd_char == \'q\')\n add_filtered_history(line);\n break;\n }\n if (glob_buffer.is_empty()) // If buffer was emptied\n in_string=0;\n if (interactive && status.add_to_history)\n add_filtered_history(line);\n continue;\n }\n if (add_line(glob_buffer, line, line_length, &in_string, &ml_comment,\n status.line_buff ? status.line_buff->truncated : 0))\n break;\n }\n /* if in batch mode, send last query even if it doesn\'t end with \\g or go */\n\n if (!interactive && !status.exit_status)\n {\n remove_cntrl(glob_buffer);\n if (!glob_buffer.is_empty())\n {\n status.exit_status=1;\n if (com_go(&glob_buffer,line) <= 0)\n status.exit_status=0;\n }\n }\n\n#if defined(_WIN32)\n buffer.free();\n tmpbuf.free();\n#else\n if (interactive)\n /*\n free the last entered line.\n */\n free(line);\n#endif\n\n /*\n If the function is called by \'source\' command, it will return to interactive\n mode, so real_binary_mode should be FALSE. Otherwise, it will exit the\n program, it is safe to set real_binary_mode to FALSE.\n */\n real_binary_mode= FALSE;\n return status.exit_status;\n}}'

    ts.parse_input(code_snippet=code)
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

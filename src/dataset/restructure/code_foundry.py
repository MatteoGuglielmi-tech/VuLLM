from __future__ import annotations
from tree_sitter import Language, Parser, Query, QueryCursor


class CodeFoundry:

    def __init__(self, ts_lang: Language):
        self.ts_lang: Language = ts_lang
        self.ts_parser: Parser = Parser(language=ts_lang)

    def fix_dangling_directives(self, code: str) -> str:
        """Finds dangling directives. Deletes lone #endif directives, and for
        dangling #elif/#else blocks, it removes the directives and wraps the
        orphaned code in a new scope with '{}' to preserve it.
        """

        tree = self.ts_parser.parse(bytes(code, encoding="utf-8"))
        code_bytes = bytearray(code, encoding="utf-8")

        query_string = """
        [ (preproc_if) (preproc_ifdef) ] @opener
        [ (preproc_elif) (preproc_else) (preproc_call directive: (_) @elif (#eq? @elif "#elif"))
          (preproc_call directive: (_) @else (#eq? @else "#else")) ] @intermediate
        [ (preproc_call directive: (_) @endif (#eq? @endif "#endif")) ("#endif") ] @closer
        """
        query = Query(self.ts_lang, query_string)
        captures = QueryCursor(query).captures(tree.root_node)
        all_directives = sorted(
            [(n, c) for c, n_list in captures.items() for n in n_list],
            key=lambda x: x[0].start_byte,
        )

        # identify the indices of all dangling directives.
        dangling_indices = set()
        opener_stack = []
        for i, (_, name) in enumerate(all_directives):
            if name == "opener": opener_stack.append(i)
            elif name == "intermediate":
                if not opener_stack: dangling_indices.add(i)
            elif name == "closer":
                if not opener_stack: dangling_indices.add(i)
                else: opener_stack.pop()

        # plan the replacements.
        # replacements -> list of (start_byte, end_byte, new_text_bytes)
        replacements: list[tuple[int, int, bytes]] = []
        new_block_ranges = []
        processed_indices = set()
        for i in sorted(list(dangling_indices)):
            if i in processed_indices: continue

            start_node, name = all_directives[i]
            processed_indices.add(i)

            def get_bounds(node):
                end_byte = node.end_byte
                if end_byte < len(code_bytes) and code_bytes[end_byte] == ord("\n"):
                    end_byte += 1
                return node.start_byte, end_byte

            if name == "closer":  # lone dangling #endif
                replacements.append((*get_bounds(start_node), b""))
            elif name == "intermediate":  # dangling #else or #elif
                nesting_level, end_node = 0, None
                for j in range(i + 1, len(all_directives)):
                    if j in dangling_indices:
                        inner_node, inner_name = all_directives[j]
                        if inner_name == "opener":
                            nesting_level += 1
                        elif inner_name == "closer":
                            if nesting_level == 0:
                                end_node = inner_node
                                processed_indices.update(range(i, j + 1))
                                break
                            nesting_level -= 1

                if end_node:
                    content = code_bytes[start_node.end_byte:end_node.start_byte]
                    new_text = b"{\n" + content.strip() + b"\n}"
                    start_byte, end_byte = (start_node.start_byte, get_bounds(end_node)[1])
                    replacements.append((start_byte, end_byte, new_text))
                    new_block_ranges.append((start_byte, start_byte + len(new_text)))
                else:
                    # (FALLBACK): No matching #endif found. Delete the lone directive.
                    replacements.append((*get_bounds(start_node), b""))

        return code_bytes.decode(encoding="utf-8")


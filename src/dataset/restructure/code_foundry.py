from __future__ import annotations
from tree_sitter import Language, Parser, Query, QueryCursor


class CodeFoundry:

    def __init__(self, ts_lang: Language):
        self.ts_lang: Language = ts_lang
        self.ts_parser: Parser = Parser(language=ts_lang)
        self.directive_query: Query = self._create_directive_query()

    def _create_directive_query(self) -> Query:
        query_string = """
        [ (preproc_if) (preproc_ifdef) ] @opener
        [ (preproc_elif) (preproc_else) (preproc_call directive: (_) @elif (#eq? @elif "#elif"))
          (preproc_call directive: (_) @else (#eq? @else "#else")) ] @intermediate
        [ (preproc_call directive: (_) @endif (#eq? @endif "#endif")) ("#endif") ] @closer
        """

        return Query(self.ts_lang, query_string)

    def _create_safety_check_query(self) -> Query:
        query_string = "(ERROR) @broken_block"

        return Query(self.ts_lang, query_string)

    def fix_dangling_directives(self, code: str) -> str:
        """
        Finds dangling directives. Deletes lone #endif directives, and for
        dangling #elif/#else blocks, it removes the directives and wraps the
        orphaned code in a new scope with '{}' to preserve it.
        """

        tree = self.ts_parser.parse(bytes(code, encoding="utf-8"))
        code_bytes = bytearray(code, encoding="utf-8")

        captures = QueryCursor(self.directive_query).captures(tree.root_node)
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

        if not dangling_indices: return code

        replacements: list[tuple[int, int, bytes]] = []
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
                block_directives_indices = []
                for j in range(i + 1, len(all_directives)):
                    if j in dangling_indices:
                        inner_node, inner_name = all_directives[j]
                        if inner_name == "opener":
                            nesting_level += 1
                        elif inner_name == "closer":
                            if nesting_level == 0:
                                end_node = inner_node
                                # processed_indices.update(range(i, j + 1))
                                block_directives_indices = list(range(i, j+1))
                                break
                            nesting_level -= 1

                if end_node:
                    # content = code_bytes[start_node.end_byte:end_node.start_byte]
                    # new_text = b"{\n" + content.strip() + b"\n}"
                    # new_text = content.strip()
                    new_text = b""
                    start_byte, end_byte = (start_node.start_byte, get_bounds(end_node)[1])
                    replacements.append((start_byte, end_byte, new_text))
                    processed_indices.update(block_directives_indices)
                else:
                    # (FALLBACK): No matching #endif found. Delete the lone directive.
                    replacements.append((*get_bounds(start_node), b""))

        offset = 0
        for start, end, new_text in sorted(replacements, key=lambda r: r[0]):
            adj_start, adj_end = start + offset, end + offset
            code_bytes[adj_start:adj_end] = new_text
            offset += len(new_text) - (end - start)

        return code_bytes.decode(encoding="utf-8")

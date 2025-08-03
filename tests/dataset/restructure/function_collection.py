from tree_sitter import Language, Parser, Node, Query, QueryCursor
from collections.abc import Generator
from tree_sitter_c import language as c_language

class TreeSitter:
    def __init__(self) -> None:
        C_LANGUAGE = Language(c_language())
        self.language : Language = C_LANGUAGE
        self.parser = Parser(self.language)

    def parse_input(self, code_snippet: str):
        self.tree = self.parser.parse(bytes(code_snippet, "utf8"))
        self.root_node = self.tree.root_node

    def traverse_tree(self) -> Generator[Node, None, None]:
        if not self.tree: return
        cursor = self.tree.walk()
        reached_root = False
        while not reached_root:
            assert cursor.node is not None
            yield cursor.node
            if cursor.goto_first_child(): continue
            if cursor.goto_next_sibling(): continue
            while True:
                if not cursor.goto_parent():
                    reached_root = True
                    break
                if cursor.goto_next_sibling():
                    break

def _remove_dangling_directives_isolate_preserve(code: str, ts:TreeSitter) -> str:
        """
        Finds dangling directives. Deletes lone #endif directives, and for
        dangling #elif/#else blocks, it removes the directives and wraps the
        orphaned code in a new scope with '{}' to preserve it.
        """

        ts.parse_input(code)
        code_bytes = bytearray(code, "utf8")

        # The query remains the same as it correctly identifies all directive types.
        query_string = """
        [ (preproc_if) (preproc_ifdef) ] @opener
        [ (preproc_elif) (preproc_else) (preproc_call directive: (_) @elif (#eq? @elif "#elif")) (preproc_call directive: (_) @else (#eq? @else "#else")) ] @intermediate
        [ (preproc_call directive: (_) @endif (#eq? @endif "#endif")) ("#endif") ] @closer
        """
        query = Query(ts.language, query_string)
        captures = QueryCursor(query).captures(ts.root_node)

        all_directives = sorted([(n, c) for c, n_list in captures.items() for n in n_list], key=lambda x: x[0].start_byte)

        # 1. identify the indices of all dangling directives.
        dangling_indices = set()
        opener_stack = []
        for i, (_, capture_name) in enumerate(all_directives):
            if capture_name == "opener":
                opener_stack.append(i)
            elif capture_name == "intermediate":
                if not opener_stack:
                    dangling_indices.add(i)
            elif capture_name == "closer":
                if not opener_stack:
                    dangling_indices.add(i)
                else:
                    opener_stack.pop()

        # 2. plan the replacements.
        replacements: list[tuple[int,int,bytes]] = []  # list of (start_byte, end_byte, new_text_bytes)
        processed_indices = set()
        for i in sorted(list(dangling_indices)):
            if i in processed_indices:
                continue

            start_node, capture_name = all_directives[i]

            def get_bounds(node):
                end_byte = node.end_byte
                if end_byte < len(code_bytes) and code_bytes[end_byte] == ord('\n'):
                    end_byte += 1
                return node.start_byte, end_byte

            if capture_name == "closer": # lone dangling #endif
                replacements.append((*get_bounds(start_node), b''))


            elif capture_name == "intermediate":  # dangling #else or #elif
                nesting_level = 0
                end_node = None
                for j in range(i + 1, len(all_directives)):
                    inner_node, inner_name = all_directives[j]
                    if j in dangling_indices:
                        inner_node, inner_name = all_directives[j]
                        processed_indices.add(j)
                        if inner_name == "opener":
                            nesting_level += 1
                        elif inner_name == "closer":
                            if nesting_level == 0:
                                end_node = inner_node
                                break
                            nesting_level -= 1

                if end_node:
                    content = code_bytes[start_node.end_byte:end_node.start_byte]
                    new_text = b'{\n' + content.strip() + b'\n}'
                    replacements.append((start_node.start_byte, get_bounds(end_node)[1], new_text))
                else:
                    # (FALLBACK): No matching #endif found. Delete the lone directive.
                    replacements.append((*get_bounds(start_node), b''))

        for start, end, new_text in sorted(replacements, key=lambda r: r[0], reverse=True):
            code_bytes[start:end] = new_text

        return code_bytes.decode("utf8")


if __name__ == "__main__":
    sample_code = """
#if A
  int x = 1;
#endif
#else
  int x = 2;
  vulnerable_call();
#endif
"""
    sample_code = """
void func() {
  #else // Dangling intermediate
    int y = 2;
  #endif
}"""
    sample_code = """
#endif // Dangling
void func() {
#if A
  int x = 1;
#else
  int y = 2;
#endif
#elif C // Dangling
  int z = 3;
}

    """
    sample_code = """
void func() {
#if A
    #if B
        int x = 1;
    #endif
    #endif // This one is correct
#else
    int y = 2;
    #endif // This one is dangling
#endif // This one is also dangling
}

    """
    tsc = TreeSitter()
    # tsc.parse_input(sample_code)
    # visited_nodes = [(n.type, n.text) for n in tsc.traverse_tree()]
    # print("Visited node metadata (type, text):\n[")
    # for node_tuple in visited_nodes:
    #     print(f"\t{node_tuple}")
    # print("]")

    code_bytes = _remove_dangling_directives_isolate_preserve(sample_code, tsc)
    print(code_bytes)



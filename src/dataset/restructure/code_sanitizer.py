import os
import re
from dataclasses import dataclass
from tree_sitter import Node, Tree, Query, QueryCursor

from .shared.tree_sitter_parser import TreeSitterParser

TSNode = Node | None
Nodes = list[Node]
Match = re.Match[str]|None


@dataclass
class CodeSanitizer:
    """Encapsulates the end-to-end process of cleaning, repairing, and
    refactoring source code snippets from a JSON dataset."""

    def __post_init__(self) -> None:
        """Initializes variables and necessary folders."""

        self.processed_data: dict[int, dict] = {}
        self.last_used_language: str | None = None

        os.makedirs("./misc", exist_ok=True)
        self.tmp_c_file = "./misc/tmp.c"
        self.tmp_cpp_file = "./misc/tmp.cpp"

    def remove_comments(self, code: str, tsp: TreeSitterParser) -> str:
        """Parses the code and removes all comment nodes.

        Params:
            code: str
                The soruce code comments needs to be removed from.
            lang_name: str
                Language of the code to be processed.
        Returns:
            str
                source code cleaned up from comments
        """

        tree: Tree = tsp.parse(code)
        comments: list[Node] = [node for node in tsp.traverse_tree(tree=tree) if node.type == "comment"]

        # Iterate backwards to avoid index shifting issues during replacement (in-place replacement)
        for comment_node in sorted(comments, key=lambda c: c.start_byte, reverse=True):
            code = code[: comment_node.start_byte] + code[comment_node.end_byte :]

        return code.lstrip()

    def _validate_and_extract_body(self, code: str, tsp: TreeSitterParser) -> str | None:
        """
        Validates the snippet contains a desirable compound statement and
        extracts the code from that point onward, discarding leading garbage.

        Params:
            code: str
                source code to be modified.
            parser: TreeSitter
                class implementing Tree-Sitter based utilities.

        Returns:
            str|None:
                the cleaned code string if valid, otherwise None.
        """

        tree = tsp.parse(code)

        # This query specifically targets compound statements we want to keep.
        # - function_definition: Standard C/C++ functions.
        # - template_declaration: C++ templates wrapping functions or classes.
        # - namespace_definition: C++ namespaces.
        # - class_specifier: C++ classes (which contain methods).
        # - compound_statement: A standalone { ... } block. This is our key
        #   for catching old K&R C functions, where the type is implicit and
        #   Tree-sitter might see it as a declaration followed by a compound statement.
        if tsp.language_name == "cpp":
            query_str = """
            [
              (function_definition)
              (template_declaration)
              (namespace_definition)
              (compound_statement)
            ] @target
            """
        else:
            # C query is restricted to C constructs
            query_str = "[ (function_definition) (compound_statement) ] @target"

        query: Query = Query(tsp.language, query_str)
        query_cursor: QueryCursor = QueryCursor(query)
        captures: dict[str, list[Node]] = query_cursor.captures(tree.root_node)
        target_nodes: list[Node]|None = captures.get("target")
        if not target_nodes:
            return None

        # find the earliest valid construct
        first_node: Node = min(target_nodes, key=lambda n: n.start_byte)

        # <---- handling for K&R C functions ---->
        # If we found a standalone compound_statement, check if its parent is
        # the root or an ERROR at the root. This confirms it's not a nested block.
        if first_node.type == "compound_statement":
            parent: Node|None = first_node.parent
            if parent and parent.type not in ["translation_unit", "ERROR"]:
                return None

        return code[first_node.start_byte :]

    def _preprocess_directives(self, code: str, tsp: TreeSitterParser) -> str:
        """Recursively simplifies preprocessor directives like #if 1 and #if 0.

        Params:
            code: str
                source code to be modified.
        Returns:
            str: the restructured code
        """

        tree = tsp.parse(code)

        # find all possible #if blocks and iterate from innermost to outermost
        # query: Query = parser.language.query("(preproc_if) @if")
        query: Query = Query(tsp.language, "(preproc_if) @if")
        query_cursor: QueryCursor = QueryCursor(query)
        all_if_nodes: list[Node] = sorted(
            query_cursor.captures(tree.root_node).get('if', []),
            key=lambda n: n.start_byte, reverse=True
        )

        for node_to_replace in all_if_nodes:
            condition_node:TSNode = node_to_replace.child_by_field_name('condition')
            # if literal not 1 or 0
            if not (condition_node and condition_node.type == 'number_literal'):
                continue

            assert condition_node.text is not None
            condition_text:str = condition_node.text.decode()
            replacement_text:str|None = None

            # if 1
            if condition_text == '1':
                # the consequence is all nodes between the condition and the alternative/endif
                consequence_nodes: Nodes = []
                next_sibling: TSNode = condition_node.next_sibling
                while next_sibling:
                    if next_sibling.type in ['preproc_else', '#endif']:
                        break
                    consequence_nodes.append(next_sibling)
                    next_sibling = next_sibling.next_sibling

                if consequence_nodes:
                    # reconstruct the code from the collected nodes
                    start:int = consequence_nodes[0].start_byte
                    end:int = consequence_nodes[-1].end_byte
                    replacement_text = code[start:end]
                else:
                    replacement_text = ""

            # if 0
            elif condition_text == '0':
                alternative_node:TSNode = node_to_replace.child_by_field_name('alternative')
                if alternative_node and alternative_node.type == "preproc_else":
                    if alternative_node.children:
                        content_node = alternative_node.children[-1]
                        replacement_text = content_node.text.decode("utf-8") if content_node.text else ""
                    else:
                        replacement_text = ""
                else:
                    replacement_text = ""

            if replacement_text is not None:
                new_code = (
                    code[:node_to_replace.start_byte] +
                    replacement_text +
                    code[node_to_replace.end_byte:]
                )
                # On the first successful replacement, recurse immediately
                return self._preprocess_directives(code=new_code, tsp=tsp)

        return code

    def _balance_directives(self, code: str, tsp: TreeSitterParser) -> str:
        """Balances preprocessor directives using a text-based classification
        of fundamental `preproc_directive` nodes.

        Params:
            code: str
                source code to be modified.
        Returns:
            str: the restructured code
        """

        all_directives: list[tuple[Node, str]] = []
        tree = tsp.parse(code)

        for node in tsp.traverse_tree(tree=tree):
            # ignore zero-width "phantom" nodes
            if node.start_byte == node.end_byte:
                continue

            node_text: bytes|None = node.text
            if node.type in {"#if", "#ifdef", "#ifndef"}:
                all_directives.append((node, 'opener'))
            elif node.type in {"#endif", "preproc_directive"} and node_text:
                if node_text.startswith(b'#endif'):
                    all_directives.append((node, 'closer'))

        all_directives.sort(key=lambda d: d[0].start_byte)
        balance = sum(1 if name == 'opener' else -1 for _, name in all_directives)

        return code + ("\n#endif" * balance) if balance > 0 else code

    def add_missing_braces(self, code: str) -> str:
        open_braces = code.count('{')
        close_braces = code.count('}')
        missing_braces = open_braces - close_braces

        if missing_braces > 0:
            return code + '\n' + '}' * missing_braces

        return code

    def add_missing_return_types(self, code: str, tsp: TreeSitterParser) -> str:
        """Finds and fixes C functions that are missing an explicit
        return type by adding the default 'int'.
        """

        tree: Tree = tsp.parse(code=code)
        root_node:Node = tree.root_node

        if root_node.child_count > 0 and root_node.children[0].type == "function_definition":
            return code

        # A function with a missing return type is often parsed as two top-level
        # nodes: the header (as an ERROR or expression_statement) and the body.
        if root_node.child_count >= 2:
            first_child = root_node.children[0]
            second_child = root_node.children[1]

            # Check for the two known broken patterns:
            # 1. (ERROR) followed by (compound_statement)
            # 2. (expression_statement) followed by (compound_statement)
            is_broken_pattern = (
                first_child.type in ["ERROR", "expression_statement"] and
                second_child.type == "compound_statement"
            )

            if is_broken_pattern:
                return "int " + code

        return code

    def _find_identifier_in_declarator(self, node: Node) -> Node|None:
        """Recursively finds the base identifier node within a declarator."""

        if node.type == "identifier": return node
        declarator_child = node.child_by_field_name("declarator")
        if declarator_child:
            return self._find_identifier_in_declarator(declarator_child)

        return None

    def _kr_style_to_ansi(self, code: str, tsp: TreeSitterParser) -> str:
        """Transforms a K&R C-style function into a modern ANSI C function using
        a robust, two-stage Tree-sitter approach.
        """

        # --- Stage 1: Normalization Pass ---
        # check for the "broken" AST pattern of a K&R function
        broken_kr_query_str = """
            (translation_unit
              (expression_statement) @header
              (declaration)+ @param_decls .
              (compound_statement) @body
            )
        """
        if tsp.query(code=code, query_str=broken_kr_query_str):
            code = "int " + code.strip()

        # --- Stage 2: Conversion Pass ---
        # all K&R functions should have a valid `function_definition` node.
        # This query finds a function_definition with K&R-style parameter declarations.
        kr_func_query_str = """
            (function_definition
              (storage_class_specifier)* @specifiers
              type: (primitive_type) @ret_type
              declarator: (function_declarator
                parameters: (parameter_list) @param_names)
              (declaration)+ @param_decls
              body: (compound_statement) @body
            ) @kr_function
        """

        captures:dict[str,list[Node]] = tsp.query(code=code, query_str=kr_func_query_str)

        param_names_nodes:list[Node]|None = captures.get("param_names", [])
        param_decl_nodes:list[Node]|None = captures.get("param_decls", [])

        if not (param_names_nodes and param_decl_nodes): return code

        param_names_node:Node = param_names_nodes[0]

        # --- Reconstruct the Parameter List ---
        param_names:list[str] = [
            p.text.decode('utf-8') for p in param_names_node.children 
            if p.type == 'identifier' and p.text
        ]

        param_decl_map: dict[str, str] = {}
        for decl_node in param_decl_nodes:
            type_node = decl_node.child_by_field_name("type")
            if not type_node or not type_node.text: continue

            type_text = type_node.text.decode('utf-8')

            for declarator_node in decl_node.children_by_field_name("declarator"):
                if not declarator_node.text: continue

                full_declarator_text = declarator_node.text.decode('utf-8')
                identifier_node = self._find_identifier_in_declarator(declarator_node)

                if identifier_node and identifier_node.text:
                    var_name = identifier_node.text.decode('utf-8')
                    param_decl_map[var_name] = f"{type_text} {full_declarator_text}"

        new_params = [param_decl_map.get(name, f"int {name}") for name in param_names]

        # --- Plan and Apply Edits ---
        edits = []
        new_param_list_text = f"({', '.join(new_params)})".encode('utf-8')
        edits.append((param_names_node.start_byte, param_names_node.end_byte, new_param_list_text))

        start_byte_to_remove = param_names_node.end_byte
        end_byte_to_remove = param_decl_nodes[-1].end_byte
        edits.append((start_byte_to_remove, end_byte_to_remove, b''))

        code_bytes = bytearray(code, "utf-8")
        for start, end, text in sorted(edits, key=lambda x: x[0], reverse=True):
            code_bytes[start:end] = text

        return code_bytes.decode("utf-8")

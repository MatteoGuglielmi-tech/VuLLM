import subprocess
import logging
import tree_sitter

from dataclasses import dataclass
from common.common_typedef import Captures, TSNode
from tree_sitter import Node, QueryError, Tree, Query, QueryCursor

from ...common.tree_sitter_parser import TreeSitterParser

logger = logging.getLogger(__name__)


@dataclass
class CodeSanitizer:
    """Encapsulates the end-to-end process of cleaning, repairing, and
    refactoring source code snippets from a JSON dataset."""

    def __post_init__(self) -> None:
        """Initializes variables and necessary folders."""

        self.processed_data: dict[int, dict] = {}
        self.last_used_language: str | None = None

    def remove_non_ascii(self, code: str) -> str:
        """Removes all non-ASCII characters from a string."""
        return code.encode("ascii", "ignore").decode("ascii")

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

        comments: list[Node] = [
            node
            for node in tsp.traverse_tree(node=tree.root_node)
            if node.type == "comment"
        ]

        # Iterate backwards to avoid index shifting issues during replacement (in-place replacement)
        for comment_node in sorted(comments, key=lambda c: c.start_byte, reverse=True):
            code = code[: comment_node.start_byte] + code[comment_node.end_byte :]

        return code.lstrip()

    def validate_and_extract_body(self, code: str, tsp: TreeSitterParser) -> str:
        """
        Validates the snippet contains a desirable compound statement and
        extracts the code from that point onward, discarding leading garbage.

        Params
        ------
        code: str
            source code to be modified.
        parser: TreeSitter
            class implementing Tree-Sitter based utilities.

        Returns
        -------
        str:
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
            query_str = """[ 
                (function_definition)
                (compound_statement)
                (zend_vm_handler)
            ] @target"""

        captures: Captures = tsp.run_query_on_node(node=tree.root_node, query_str=query_str)
        target_nodes: list[Node] = captures.get("target", [])
        if not target_nodes: return ""

        # find the earliest valid construct
        first_node: Node = min(target_nodes, key=lambda n: n.start_byte)

        # <---- handling for K&R C functions ---->
        # If we found a standalone compound_statement, check if its parent is
        # the root or an ERROR at the root. This confirms it's not a nested block.
        if first_node.type == "compound_statement":
            parent: TSNode = first_node.parent
            if parent and parent.type not in ["translation_unit", "ERROR"]:
                return ""

        return code[first_node.start_byte:]

    def balance_directives(self, code: str, tsp: TreeSitterParser) -> str:
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

        for node in tsp.traverse_tree(node=tree.root_node):
            # ignore zero-width "phantom" nodes
            if node.start_byte == node.end_byte:
                continue

            node_text: bytes | None = node.text
            if node.type in {"#if", "#ifdef", "#ifndef"}:
                all_directives.append((node, "opener"))
            elif node.type in {"#endif", "preproc_directive"} and node_text:
                if node_text.startswith(b"#endif"):
                    all_directives.append((node, "closer"))

        all_directives.sort(key=lambda d: d[0].start_byte)
        balance = sum(1 if name == "opener" else -1 for _, name in all_directives)

        return code + ("\n#endif" * balance) if balance > 0 else code

    def add_missing_braces(self, code: str, tsp: TreeSitterParser) -> str:
        """
        Appends missing closing braces to a code snippet using a syntax-aware
        parser (tree-sitter) to avoid incorrectly counting braces within
        strings or comments.

        Params:
            code: The code snippet to check and fix.
            language: The tree-sitter language grammar to use (e.g., C or C++).

        Returns:
            The code snippet with missing braces appended.
        """

        tree: Tree = tsp.parse(code=code)
        query_string: str = """
            "{" @open
            "}" @close
        """
        captures: Captures = tsp.run_query_on_tree(tree=tree, query_str=query_string)
        openers = captures.get("open", [])
        closers = captures.get("close", [])

        def _filer_ghost_nodes(array: list[Node]):
            return [
                n
                for n in array
                if n.start_byte != n.end_byte and not (n.is_missing or n.is_error)
            ]

        open_braces: int = len(_filer_ghost_nodes(openers))
        close_braces: int = len(_filer_ghost_nodes(closers))
        balance = open_braces - close_braces
        if balance > 0:
            return code.rstrip() + "\n" + "}" * balance

        return code

    def add_missing_return_types(self, code: str, tsp: TreeSitterParser) -> str:
        """Finds and fixes C functions that are missing an explicit
        return type by adding the default 'int'.
        """

        is_broken_pattern: bool = False

        tree: Tree = tsp.parse(code=code)
        root_node: Node = tree.root_node

        if (
            root_node.child_count > 0
            and root_node.children[0].type == "function_definition"
        ):
            return code

        if root_node.child_count >= 2:
            # parsed as two top-level
            first_child = root_node.children[0]
            second_child = root_node.children[1]

            # Check for the two known broken patterns:
            # 1. (ERROR) followed by (compound_statement)
            # 2. (expression_statement) followed by (compound_statement)
            is_broken_pattern = (
                first_child.type in ["ERROR", "expression_statement"]
                and second_child.type == "compound_statement"
            )
        # case in which no type specifier and the function is parsed as macro_type_specifier
        if root_node and (
            (
                root_node.children[0].type == "declaration"
                and root_node.children[0].named_children[0].type
                == "macro_type_specifier"
            )
            or (root_node.children[0].type == "macro_type_specifier")
        ):
            is_broken_pattern = True

        if is_broken_pattern:
            return "int " + code

        return code

    def _find_identifier_in_declarator(self, node: Node) -> Node | None:
        """Recursively finds the base identifier node within a declarator."""

        if node.type == "identifier":
            return node
        declarator_child = node.child_by_field_name("declarator")
        if declarator_child:
            return self._find_identifier_in_declarator(declarator_child)

        return None

    def is_php_zend(self, code: str, tsp: TreeSitterParser) -> bool:
        tree: Tree = tsp.parse(code=code)
        root_node: Node = tree.root_node

        query_php_zend = """
        (translation_unit
            (zend_vm_handler) @zend
        )
        """
        try:
            captures: Captures = tsp.run_query_on_node(node=root_node, query_str=query_php_zend)
            if captures: return True  # applying this fix to the PHP Zend virtual machine would break the code
        except tree_sitter.QueryError as e: raise QueryError(f" ❌ Grammar is broken!\n {e} ")
        except Exception as e: raise e

        return False

    def kr_style_to_ansi(self, code: str, tsp: TreeSitterParser) -> str:
        """Transforms a K&R C-style function into a modern ANSI C function using
        a robust, two-stage Tree-sitter approach.
        """

        tree: Tree = tsp.parse(code=code)
        root_node: Node = tree.root_node


        # --- Stage 1: Normalization Pass ---
        # check for the "broken" AST pattern of a K&R function
        broken_kr_query_str = """
            (translation_unit
              (expression_statement) @header
              (declaration)+ @param_decls .
              (compound_statement) @body
            )
        """
        if tsp.run_query_on_node(node=root_node, query_str=broken_kr_query_str):
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

        captures: Captures = tsp.run_query_on_node(node=root_node, query_str=kr_func_query_str)

        param_names_nodes: list[Node] | None = captures.get("param_names", [])
        param_decl_nodes: list[Node] | None = captures.get("param_decls", [])

        if not (param_names_nodes and param_decl_nodes): return code

        param_names_node: Node = param_names_nodes[0]

        # --- Reconstruct the Parameter List ---
        param_names: list[str] = [
            p.text.decode("utf-8")
            for p in param_names_node.children
            if p.type == "identifier" and p.text
        ]

        param_decl_map: dict[str, str] = {}
        for decl_node in param_decl_nodes:
            type_node = decl_node.child_by_field_name("type")
            if not type_node or not type_node.text: continue

            type_text = type_node.text.decode("utf-8")

            for declarator_node in decl_node.children_by_field_name("declarator"):
                if not declarator_node.text: continue

                full_declarator_text = declarator_node.text.decode("utf-8")
                identifier_node = self._find_identifier_in_declarator(declarator_node)

                if identifier_node and identifier_node.text:
                    var_name = identifier_node.text.decode("utf-8")
                    param_decl_map[var_name] = f"{type_text} {full_declarator_text}"

        new_params = [param_decl_map.get(name, f"int {name}") for name in param_names]

        # --- Plan and Apply Edits ---
        edits = []
        new_param_list_text = f"({', '.join(new_params)})".encode("utf-8")
        edits.append((param_names_node.start_byte, param_names_node.end_byte, new_param_list_text))

        start_byte_to_remove = param_names_node.end_byte
        end_byte_to_remove = param_decl_nodes[-1].end_byte
        edits.append((start_byte_to_remove, end_byte_to_remove, b""))

        code_bytes = bytearray(code, "utf-8")
        for start, end, text in sorted(edits, key=lambda x: x[0], reverse=True):
            code_bytes[start:end] = text

        return code_bytes.decode("utf-8")

    def preprocess_code_gcc_e(self, code: str, gcc_flags: list | None = None) -> str:
        """Preprocesses a C code string using the external GCC preprocessor.

        This method invokes the `gcc -E -P -` command to run the C
        preprocessor on the input code string. It reads from stdin and
        captures the preprocessed output from stdout.

        Parameters
        ----------
        code : str
            A string containing the C source code to be preprocessed.
        gcc_flags : list of str, optional
            A list of optional flags to pass to the GCC preprocessor,
            e.g., `['-DDEBUG=1']` to define a macro. If None, no extra
            flags are used.

        Returns
        -------
        str
            The preprocessed C code as a string.

        Raises
        ------
        FileNotFoundError
            If the 'gcc' command is not found. Ensure GCC is installed and
            accessible in the system's PATH.
        subprocess.SubprocessError
            If the preprocessing command fails (returns a non-zero exit code),
            which can occur due to errors in the C code.
        """

        if gcc_flags is None:
            gcc_flags = []

        # Command to execute: gcc -E -P -
        # -E: run the preprocessor only.
        # -P: inhibit generation of linemarkers
        # -:  read from standard input.
        command: list[str] = ["gcc", "-E", "-P", "-"] + gcc_flags

        try:
            process = subprocess.run(
                command, input=code, text=True, capture_output=True, check=True
            )
            return process.stdout
        except FileNotFoundError:
            raise FileNotFoundError(" ❌ 'gcc' not found. Ensure it is installed and in Path")
        except subprocess.CalledProcessError as e:
            raise subprocess.SubprocessError(f" ❌ Error during preprocessing:\n{e.stderr}")

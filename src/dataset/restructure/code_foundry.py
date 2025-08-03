from __future__ import annotations
import os

from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree
from dataclasses import dataclass, field


# --- Data Structures ---
@dataclass
class Symbol:
    """
    Attributes:
        name: str -> stores the name of the symbol (used for lookups)
        node: Node -> stores direct reference to tree-sitter node object
    """

    name: str
    node: Node


@dataclass
class Scope:
    """
    Represents a single lexical scope in the code, typically enclosed in {}
    Attributes:
        node: Node -> stores tree-sitter node which defines the boundary
        symbols: dict[str,Symbol] -> symbol table for this scope
        children: list[Scope] -> list holding all nested scopes
        parent: Scope | None -> reference to parent state that contains this one
    """

    node: Node
    symbols: dict[str, Symbol] = field(default_factory=dict)
    children: list[Scope] = field(default_factory=list)
    parent: Scope | None = None


class SymbolTableManager:
    def __init__(self, root_node: Node):
        self.root_scope = Scope(node=root_node)
        self._scope_map: dict[int, Scope] = {root_node.id: self.root_scope}


class CodeFoundry:

    def __init__(self, ts_lang: Language):
        self.ts_lang: Language = ts_lang
        self.ts_parser: Parser = Parser(language=ts_lang)

    # --- Pass 1: Structural Fix ---
    def _fix_dangling_directives(self, code: str) -> tuple[str, list[tuple[int, int]]]:
        """
        Finds dangling directives. Deletes lone #endif directives, and for
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

        # 1. identify the indices of all dangling directives.
        dangling_indices = set()
        opener_stack = []
        for i, (_, name) in enumerate(all_directives):
            if name == "opener":
                opener_stack.append(i)
            elif name == "intermediate":
                if not opener_stack:
                    dangling_indices.add(i)
            elif name == "closer":
                if not opener_stack:
                    dangling_indices.add(i)
                else:
                    opener_stack.pop()

        # 2. plan the replacements.
        # replacements -> list of (start_byte, end_byte, new_text_bytes)
        replacements: list[tuple[int, int, bytes]] = []
        new_block_ranges = []
        processed_indices = set()
        for i in sorted(list(dangling_indices)):
            if i in processed_indices:
                continue

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

        sorted_replacements = sorted(replacements, key=lambda r: r[0])
        new_block_map = {r[0]: r for r in new_block_ranges}
        final_block_ranges = []
        offset = 0

        for start, end, new_text in sorted_replacements:
            adj_start = start + offset
            adj_end = end + offset
            if start in new_block_map:
                final_block_ranges.append((adj_start, adj_start + len(new_text)))
            code_bytes[adj_start:adj_end] = new_text
            offset += len(new_text) - (end - start)

        return code_bytes.decode(encoding="utf-8"), final_block_ranges

    # --- Pass 2: Symbol Table Builder ---
    def _build_symbol_table(self, tree: Tree) -> SymbolTableManager:
        manager = SymbolTableManager(tree.root_node)
        self._traverse_and_build(tree.root_node, manager.root_scope, manager)
        return manager

    def _traverse_and_build(self, node: Node, parent_scope: Scope, manager: SymbolTableManager):
        # --- Function Definition: Creates a scope for its body ---
        if node.type == "function_definition":
            body_node = node.child_by_field_name("body")
            if body_node and body_node.type == "compound_statement":
                func_scope = Scope(node=body_node, parent=parent_scope)
                parent_scope.children.append(func_scope)
                manager._scope_map[body_node.id] = func_scope

                # Add parameters to the new function scope
                declarator = node.child_by_field_name("declarator")
                if declarator:
                    params = declarator.child_by_field_name("parameters")
                    if params:
                        for param in params.children:
                            if param.type == "parameter_declaration":
                                id_node = self._find_identifier_in_declarator(param)
                                if id_node and id_node.text:
                                    var_name = id_node.text.decode(encoding="utf-8")
                                    func_scope.symbols[var_name] = Symbol(name=var_name, node=id_node)

                # Recurse into the function's body
                for child in body_node.children:
                    self._traverse_and_build(node=child, parent_scope=func_scope, manager=manager)
            return

        # --- For Statement: Creates a scope for its body ---
        if node.type == "for_statement":
            body_node = node.child_by_field_name("body")
            if body_node and body_node.type == "compound_statement":
                # The body node defines the scope
                for_scope = Scope(node=body_node, parent=parent_scope)
                parent_scope.children.append(for_scope)
                manager._scope_map[body_node.id] = for_scope

                # The initializer variable belongs in this new scope
                initializer = node.child_by_field_name("initializer")
                if initializer and initializer.type == "declaration":
                    for declarator in initializer.children_by_field_name("declarator"):
                        id_node = self._find_identifier_in_declarator(declarator)
                        if id_node and id_node.text:
                            var_name = id_node.text.decode(encoding="utf-8")
                            for_scope.symbols[var_name] = Symbol(name=var_name, node=id_node)

                # Recurse into the for-loop's body
                for child in body_node.children:
                    self._traverse_and_build(node=child, parent_scope=for_scope, manager=manager)
            return

        # --- Compound Statement: Creates a new scope ---
        if node.type == "compound_statement":
            new_scope = Scope(node=node, parent=parent_scope)
            parent_scope.children.append(new_scope)
            manager._scope_map[node.id] = new_scope
            # Recurse into the block's children with the new scope
            for child in node.children:
                self._traverse_and_build(node=child, parent_scope=new_scope, manager=manager)
            return

        # --- Declaration: Adds a symbol to the CURRENT scope ---
        if node.type == "declaration":
            for declarator in node.children_by_field_name("declarator"):
                id_node = self._find_identifier_in_declarator(declarator)
                if id_node and id_node.text:
                    var_name = id_node.text.decode(encoding="utf-8")
                    # Add to the parent scope it was given
                    if var_name not in parent_scope.symbols:
                        parent_scope.symbols[var_name] = Symbol(name=var_name, node=id_node)
            return

        # --- Fallback for other nodes (if, while, etc.) ---
        # These nodes don't create scopes themselves, so they just pass the context down.
        for child in node.children:
            self._traverse_and_build(node=child, parent_scope=parent_scope, manager=manager)


    def _find_identifier_in_declarator(self, node: Node) -> Node | None:
        """Recursively searches within a declarator node (which can be nested,
        e.g., for pointers) to find the base identifier node.
        """

        if node.type == "identifier":
            return node

        # Recursive step: search in named child fields first, which is more efficient.
        for field_name in ["declarator", "identifier"]:
            child = node.child_by_field_name(field_name)
            if child:
                found = self._find_identifier_in_declarator(child)
                if found:
                    return found

        # Fallback: if no named fields match, search all children.
        for child in node.children:
            found = self._find_identifier_in_declarator(child)
            if found:
                return found

        return None

    # --- Pass 3: Scope Refinement ---
    def _scope_or_descendants_have_conflict(self, scope_to_check: Scope, initial_parent_scope: Scope | None) -> bool:
        """Recursively checks if the given scope or any of its descendants
        declare a symbol that conflicts with an ancestor scope.
        """

        for var_name in scope_to_check.symbols:
            ancestor = initial_parent_scope
            while ancestor:
                if var_name in ancestor.symbols:
                    return True
                ancestor = ancestor.parent

        for child_scope in scope_to_check.children:
            if self._scope_or_descendants_have_conflict(child_scope, initial_parent_scope):
                return True

        return False

    def _refine_scopes(self, code: str, tree: Tree, blocks_to_check: list[tuple[int, int]], table_manager: SymbolTableManager) -> str:
        root_node = tree.root_node
        code_bytes = bytearray(code, encoding="utf-8")
        removals: list[tuple[int, int, bytes]] = []

        for start_byte, end_byte in blocks_to_check:
            block_node = root_node.descendant_for_byte_range(start_byte, end_byte-1)
            if not block_node or block_node.type != "compound_statement":
                continue

            parent_node = block_node.parent
            if not parent_node or parent_node.type != "compound_statement":
                continue

            # 1. Check for symbol conflicts.
            parent_scope: Scope|None = None
            cursor = block_node.parent
            while cursor:
                if cursor.id in table_manager._scope_map:
                    parent_scope = table_manager._scope_map[cursor.id]
                    break
                cursor = cursor.parent

            if not parent_scope: continue

            block_scope: Scope|None = table_manager._scope_map.get(block_node.id)
            if not block_scope: continue

            has_conflict: bool = False
            for var_name in block_scope.symbols:
                scope_cursor: Scope|None = parent_scope
                while scope_cursor:
                    if var_name in scope_cursor.symbols:
                        has_conflict = True
                        break
                    scope_cursor = scope_cursor.parent
                if has_conflict: break

            if has_conflict: continue

            # 2. Check for syntactic necessity.
            #    A block is syntactically necessary if its parent requires braces for 
            #    a multi-statement body (e.g., if, for, while).
            parent_node = block_node.parent
            is_syntactically_required = False
            if parent_node:
                if parent_node.type == "function_definition":
                    is_syntactically_required = True

                # The direct body of an if/for/while requires braces if it
                # contains more than one statement.
                if parent_node.type in ["if_statement", "for_statement", "while_statement", "do_statement"]:
                    # named_children conveniently ignores `{` and `}`
                    if len(block_node.named_children) > 1:
                        is_syntactically_required = True

            if is_syntactically_required:
                continue

            # 3. If the block can be unwrapped, plan to remove only its braces.
            # This is safer than replacing the entire block's content.
            open_brace = None
            close_brace = None
            for child in block_node.children:
                if child.type == '{':
                    open_brace = child
                elif child.type == '}':
                    close_brace = child

            if open_brace and close_brace:
                # Plan to remove the opening brace
                removals.append((open_brace.start_byte, open_brace.end_byte, b''))
                # Plan to remove the closing brace
                removals.append((close_brace.start_byte, close_brace.end_byte, b''))

        for start, end, new_text in sorted(removals, key=lambda r: r[0], reverse=True):
            code_bytes[start:end] = new_text

        return code_bytes.decode(encoding="utf-8")

    # --- Main Entry Point ---
    def run_multi_pass_fix(self, code: str) -> str:

        fixed_code, _ = self._fix_dangling_directives(code)
        iter = 1
        while True:
            tree = self.ts_parser.parse(bytes(fixed_code, encoding="utf-8"))
            symbol_manager = self._build_symbol_table(tree=tree)
            query_string: str = """[
                (compound_statement (compound_statement) @block_to_check)
                (for_statement body: (compound_statement) @block_to_check)
                (if_statement consequence: (compound_statement) @block_to_check)
                (if_statement alternative: (else_clause (compound_statement) @block_to_check))
                (while_statement body: (compound_statement) @block_to_check)
                (do_statement body: (compound_statement) @block_to_check)
                (switch_statement body: (compound_statement) @block_to_check)
                ]"""

            query = Query(self.ts_lang, query_string)
            captures = QueryCursor(query).captures(tree.root_node).get("block_to_check", [])
            blocks_to_check = [(node.start_byte, node.end_byte) for node in captures]

            refined_code = self._refine_scopes(
                code=fixed_code,
                tree=tree,
                blocks_to_check=blocks_to_check,
                table_manager=symbol_manager,
            )

            iter +=1

            if refined_code == fixed_code: break
            fixed_code = refined_code

        # remove empty lines and lines of only spaces
        fixed_code = os.linesep.join([line for line in fixed_code.splitlines() if line and line.strip() != ""])

        return fixed_code


import functools
from typing import Any

from transformers.tokenization_utils import PreTrainedTokenizer
from tree_sitter import Node, Tree

from .shared.decorators import ensure_jsonl_extension, validate_argument_value
from .shared.utils import save_to_jsonl
from .shared.typedef import *
from ...common.tree_sitter_parser import TreeSitterParser
from ...common.common_typedef import TSNode, Capture, Captures


BLOCK_TYPES: set[str] = {
    "declaration", "if_statement",
    "for_statement", "while_statement",
    "do_statement", "switch_statement",
    "return_statement", "call_expression",
    "expression_statement", "preproc_def",
    "preproc_function_def", "preproc_if",
    "preproc_ifdef", "preproc_directive",
    "preproc_else", "preproc_elif",
}

# <---- Constants for context adjustment ---->
# number of lines to preserve as context during line-wise fallback
STMNTS_OVERLAP: int = 2
# number of lines to
CONTEXT_SIZE: int = 3
# inner_context/trailing_context should not be more than 1.5x text length
CONTEXT_TEXT_RATIO = 1.5
#  25% of available_tokens for trailing context
CONTEXT_PROPORTION_FOR_TRAILING = 0.25
# If text is shorter than this (in tokens), add trailing context
MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT = 50
# <---- Constants for context adjustment ---->


def _get_function_signature(func_def_node: Node, source_code: str) -> str:
    """Extracts the function signature from a Tree-sitter function_definition node.

    This function combines the function's return type and declarator to reconstruct
    the signature.

    Parameters
    ----------
    node : Node
        The Tree-sitter node corresponding to a 'function_definition'.
    code : str
        The full source code as a string, from which the signature text is
        extracted.

    Returns
    -------
    str
        The reconstructed function signature string. Returns '<unknown>' if the
        function's type or declarator cannot be found.
    """

    if func_def_node.type != 'function_definition':
        return "<not a function>"

    primitive_type: TSNode = func_def_node.child_by_field_name("type")
    declarator_node: TSNode = func_def_node.child_by_field_name("declarator")

    if not primitive_type or not declarator_node:
        return "<unknown signature>"

    code_str = source_code.decode("utf-8", "replace") if isinstance(source_code, bytes) else source_code

    return (code_str[primitive_type.start_byte:primitive_type.end_byte] + " "
        + code_str[declarator_node.start_byte:declarator_node.end_byte])

def get_node_structured_text(node: Node, code: bytes) -> str:
    """Extracts the text of a tree-sitter node while preserving its indentation.

    Parameters
    ----------
    node : Node
        The Tree-sitter node whose text needs to be extracted.
    code : bytes
        The full source code byte string containing the node.

    Returns
    -------
    str
        The extracted text of the node, with its original indentation preserved.
    """

    if node.start_point[0] == node.end_point[0]:
        # Simple, fast path for single-line nodes
        return code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # For multi-line nodes, find the beginning of the first line
    first_line_start_byte = code.rfind(b'\n', 0, node.start_byte) + 1

    # Slice from the start of the first line to the precise end of the node
    indented_text_bytes = code[first_line_start_byte:node.end_byte]

    return indented_text_bytes.decode("utf-8", errors="replace")


def get_func_statements(body_node: TSNode, tsp: TreeSitterParser) -> list[Node]:
    """Collects top-level statements from a function's body.

    This function uses a Tree-sitter query to find all statements and
    preprocessor directives that are direct children of a function's
    `compound_statement` node. It filters out any `ERROR` nodes and ensures
    only immediate child statements are returned, sorted by their position in
    the source code.

    Parameters
    ----------
    body_node : Node
        The Tree-sitter node representing the function's body, expected to be
        of type `compound_statement`.
    tsp : TreeSitterParser
        An instance of a Tree-sitter parser configured for the relevant
        language.

    Returns
    -------
    list of Node
        A sorted list of Tree-sitter nodes, where each node represents a
        top-level statement or preprocessor directive within the function body.
        Returns an empty list if `body_node` is invalid or no statements are
        found.
    """

    nodes: list[Node] = []

    # TODO: remove PREPROCESSOR DIRECTIVES and add custom top level macros

    # Query for statements and preprocessor directives that can appear directly inside a compound_statement
    query_string: str = """
    (declaration) @decl
    (expression_statement) @expr_stmt
    (if_statement) @if_stmt
    (while_statement) @while_statement
    (for_statement) @for_statement
    (do_statement) @do_statement
    (switch_statement) @switch_stmt
    (return_statement) @ret_stmt
    (goto_statement) @goto_stmt
    (labeled_statement) @label_stmt
    (break_statement) @break_statement
    (continue_statement) @cont_stmt
    (compound_statement) @compound_stmt_nested ; Nested blocks within the function body


    ; --- PREPROCESSOR DIRECTIVES ---
    (preproc_def) @preproc_def
    (preproc_function_def) @preproc_func_def
    (preproc_if) @preproc_if
    (preproc_ifdef) @preproc_ifdef
    """

    if body_node and body_node.type == "compound_statement":
        captures: Captures = tsp.run_query_on_node(node=body_node, query_str=query_string)

        all_captured_nodes: list[Node] = []
        for nodes_list in captures.values():
            all_captured_nodes.extend(nodes_list)

        for node in all_captured_nodes:
            # exclude ERROR nodes, as they often indicate parsing issues
            # hopefully not the case thanks to heavy pre-processing
            if node.type == "ERROR":
                continue
            if node.parent == body_node:
                nodes.append(node)

        nodes.sort(key=lambda n: n.start_byte)  # sort statements by their start byte

    return nodes


@functools.lru_cache(maxsize=1024)
def get_tcount(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """Computes the number of tokens for a given text.
    It excludes special tokens and uses memoization of results.

    Parameters
    ----------
    text : str
        The input string for which to compute the token count.
    tokenizer : PreTrainedTokenizer
        The tokenizer instance to use for encoding the text.

    Returns
    -------
    int
        The number of tokens in the input text.
    """

    return len(tokenizer.encode(text=text, add_special_tokens=False))


# ========================
# AST Utilities
# ========================
def get_enclosing_stmnt_header(node: Node, source_code: bytes, root_node: Node) -> str:
    """Finds the header of the closest enclosing statement for a given node.

    This function traverses the Abstract Syntax Tree (AST) upwards from a given node
    to find the closest logical parent statemen.
    It then extracts the code for that statement's header (e.g., `if (condition)`).

    Parameters
    ----------
    node : Node
        The starting Tree-sitter node to begin the upward traversal from.
    source_code : bytes
        The complete source code of the file as a byte string.
    root_node : Node
        The root node of the entire AST for the source code.

    Returns
    -------
    str
        The header text of the enclosing statement. This could be:
        - The condition of a loop or `if` statement (e.g., "if (a > 10)").
        - "__FUNCTION_SIGNATURE_PARENT__" if the immediate parent is a function.
        - "Global Scope" if the node is a top-level global statement.
        - An empty string if no relevant parent statement is found.
    """

    current_ancestor: Node | None = node.parent
    while current_ancestor and current_ancestor != root_node:
        # check if the current ancestor is a function definition
        if current_ancestor.type == "function_definition":
            # if the immediate parent is the function definition, return a special marker.
            return "__FUNCTION_SIGNATURE_PARENT__"

        elif current_ancestor.type in ["if_statement", "for_statement", "while_statement", "do_statement", "switch_statement"]:
            compound_child: Node | None = next((c for c in current_ancestor.children if c.type == "compound_statement"), None)
            if compound_child:
                parent_text: str = (source_code[current_ancestor.start_byte : compound_child.start_byte]
                    .decode(encoding="utf-8", errors="ignore")
                    .strip())
                if parent_text.endswith("{"):
                    parent_text = parent_text[:-1].strip()
                return parent_text
            else:
                # if no compound statement child , just return node text
                return get_node_structured_text(node=current_ancestor, code=source_code).strip()

        # move up to the parent
        current_ancestor = current_ancestor.parent

    # if we reach the translation_unit (root_node) or current_ancestor becomes None
    if current_ancestor and current_ancestor.type == "translation_unit":
        # should ideally not be hit for statements inside a function
        return "Global Scope"

    return ""


def _find_parent_function(start_node: Node) -> Node | None:
    """Traverses up the AST from start_node to find the enclosing function_definition."""

    current_node: TSNode = start_node
    while current_node:
        if current_node.type == 'function_definition':
            return current_node
        current_node = current_node.parent

    return None


def _get_inner_ast_context(
    target_node: Node,
    source_code: bytes,
    max_token_budget: int,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Builds the 'inner' context: function signature, enclosing blocks, and preceding siblings."""

    context_pieces: list[tuple[str, int]] = []
    remaining_budget: int = max_token_budget

    # get function signature
    parent_func_node = _find_parent_function(start_node=target_node)
    if parent_func_node:
        signature_text: str = _get_function_signature(parent_func_node, source_code.decode(encoding="utf-8")) + " {"
        signature_tokens: int = get_tcount(text=signature_text, tokenizer=tokenizer)
        if signature_tokens <= remaining_budget:
            context_pieces.append((signature_text, parent_func_node.start_byte))
            remaining_budget -= signature_tokens

    # add enclosing block headers (e.g., 'if (x > 0)', 'for (...)')
    ancestor: TSNode = target_node.parent
    while ancestor and ancestor != parent_func_node:
        if ancestor.type in ["if_statement", "for_statement", "while_statement", "switch_statement"]:
            body_node: TSNode = ancestor.child_by_field_name("body")
            if body_node:
                # Get the text from the start of the statement up to its body
                header_text: str = source_code[ancestor.start_byte:body_node.start_byte].decode(encoding="utf-8", errors="replace").strip()
                header_tokens: int = get_tcount(text=header_text, tokenizer=tokenizer)
                if header_tokens <= remaining_budget:
                    context_pieces.append((header_text, ancestor.start_byte))
                    remaining_budget -= header_tokens
        ancestor = ancestor.parent

    # add preceding sibling statements
    if target_node.parent:
        try:
            siblings: list[Node] = target_node.parent.children
            target_idx: int = siblings.index(target_node)
            # iterate backwards through siblings
            for i in range(target_idx - 1, -1, -1):
                sibling: Node = siblings[i]
                if sibling.type in ["comment", "{", "}"]:
                    continue

                sibling_text: str = get_node_structured_text(node=sibling, code=source_code)
                sibling_tokens: int = get_tcount(text=sibling_text, tokenizer=tokenizer)
                if sibling_tokens <= remaining_budget:
                    context_pieces.append((sibling_text, sibling.start_byte))
                    remaining_budget -= sibling_tokens
                else:
                    break # stop if budget is exceeded
        except ValueError:
            pass # target_node not found in siblings, skip

    # sort by start position and join
    context_pieces.sort(key=lambda x: x[1])
    return "\n".join(piece[0] for piece in context_pieces)

def _get_trailing_ast_context(
    target_node: Node,
    source_code: bytes,
    max_token_budget: int,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Builds the 'trailing' context: subsequent siblings and the closing brace."""

    context_pieces: list[str] = []
    remaining_budget: int = max_token_budget

    # add subsequent sibling statements
    if target_node.parent:
        try:
            siblings: list[Node] = target_node.parent.children
            target_idx: int = siblings.index(target_node)
            # iterate forwards through siblings
            for i in range(target_idx + 1, len(siblings)):
                sibling: Node = siblings[i]
                if sibling.type in ["comment", "{", "}"]:
                    continue

                sibling_text: str = get_node_structured_text(node=sibling, code=source_code)
                sibling_tokens: int = get_tcount(text=sibling_text, tokenizer=tokenizer)
                if sibling_tokens <= remaining_budget:
                    context_pieces.append(sibling_text)
                    remaining_budget -= sibling_tokens
                else:
                    break # stop if budget is exceeded
        except ValueError:
            pass # target_node not found in siblings, skip

    # add the closing brace of the parent block, if it's a compound statement
    if target_node.parent and target_node.parent.type == "compound_statement":
        closing_brace_node = target_node.children[-1]
        if closing_brace_node and closing_brace_node.type == "}":
            brace_text: str = "}"
            brace_tokens: int = get_tcount(text=brace_text, tokenizer=tokenizer)
            if brace_tokens <= remaining_budget:
                context_pieces.append(brace_text)

    return "\n".join(context_pieces)



def get_node_ast_context(
    target_node: Node,
    source_code: bytes,
    max_token_budget: int,
    context_type: str,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Extracts AST-based context for a target node within a token budget.

    This function acts as a dispatcher, calling the appropriate helper to
    build either 'inner' (preceding) or 'trailing' (succeeding) context.

    Parameters
    ----------
    target_node : Node
        The AST node for which to find the context.
    source_code : bytes
        The complete source code as a byte string.
    max_token_budget : int
        The maximum number of tokens allowed for the context string.
    context_type : str
        The type of context to retrieve ('inner' or 'trailing').
    tokenizer : PreTrainedTokenizer
        The tokenizer used to count tokens for budget adherence.

    Returns
    -------
    str
        A string containing the formatted code context.

    Raises
    ------
    ValueError
        If an invalid `context_type` is provided.

    Important Note:
        both `_get_inner_ast_context` and `_get_trailing_ast_context` captures only the 
        headers of surrounding statement nodes. This is desired since giving the LLM just 
        the headers is not only sufficient but often better than providing the full, nested context.

        Furthermore, other reasons are: token efficiency, focus on structure (most of the times the most 
        important information are the control flow conditions) and reduced noise.
    """

    if context_type == "inner":
        return _get_inner_ast_context(target_node=target_node, source_code=source_code,
            max_token_budget=max_token_budget, tokenizer=tokenizer,)
    elif context_type == "trailing":
        return _get_trailing_ast_context(target_node=target_node, source_code=source_code,
            max_token_budget=max_token_budget, tokenizer=tokenizer)
    else:
        raise ValueError(f"Invalid context_type: '{context_type}'. Must be 'inner' or 'trailing'.")


# ========================
# CHUNK EXTRACTION
# ========================
def split_by_lines(
    units_for_splitting: list[dict[str, Any]],
    max_tokens: int,
    tokenizer: PreTrainedTokenizer,
    overlap_stmts: int = STMNTS_OVERLAP,
) -> list[IntermediateChunk]:

    def _finalize_and_append_group(group_texts: list[str], context_text: str) -> None:
        """Joins texts, calculates final token counts, and appends the group."""
        final_text = "\n".join(group_texts)
        final_combined_text = context_text + "\n" + final_text if context_text else final_text
        subgroups.append({
            "text": final_combined_text,
            "tcount": get_tcount(text=final_combined_text, tokenizer=tokenizer)
        })

    subgroups: list[IntermediateChunk] = []
    current_group_texts: list[str] = []
    current_group_text_tokens: int = 0
    current_group_context_text: str = ""
    current_group_context_tokens: int = 0

    for idx, unit in enumerate(units_for_splitting):
        stmt_text: str = unit.get("text", "")
        stmt_tokens: int = unit.get("tokens", 0)

        # determine the context for a potential new chunk starting here.
        context_for_new_chunk_text: str = ""
        if idx > 0:
            start_idx = max(0, idx - overlap_stmts)
            nodes_for_context = units_for_splitting[start_idx:idx]
            context_for_new_chunk_text = "\n".join([u.get("text", "") for u in nodes_for_context])

        context_for_new_chunk_tokens = get_tcount(text=context_for_new_chunk_text, tokenizer=tokenizer)

        # are we starting a brand new group?
        if not current_group_texts:
            # check if a single statement is too large even by itself.
            if (stmt_tokens + context_for_new_chunk_tokens) > max_tokens:
                # finalize this oversized statement as its own group.
                _finalize_and_append_group([stmt_text], context_for_new_chunk_text)
                continue

            current_group_texts.append(stmt_text)
            current_group_text_tokens = stmt_tokens
            current_group_context_text = context_for_new_chunk_text
            current_group_context_tokens = context_for_new_chunk_tokens

        # are we adding to an existing group?
        else:
            # check if adding the new statement exceeds the budget for the *current* group.
            if (current_group_context_tokens + current_group_text_tokens + stmt_tokens) > max_tokens:
                _finalize_and_append_group(group_texts=current_group_texts, context_text=current_group_context_text)

                # start a new group with the current statement.
                current_group_texts = [stmt_text]
                current_group_text_tokens = stmt_tokens
                current_group_context_text = context_for_new_chunk_text
                current_group_context_tokens = context_for_new_chunk_tokens

            # otherwise, the statement fits. add it to the current group
            else:
                current_group_texts.append(stmt_text)
                current_group_text_tokens += stmt_tokens

    # finalize any remaining statements in the last group.
    if current_group_texts:
        _finalize_and_append_group(group_texts=current_group_texts, context_text=current_group_context_text)

    return subgroups


def _split_by_ast(
    text_nodes: list[Node],
    tokenizer: PreTrainedTokenizer,
    available_tokens: int,
    original_full_tree: Tree,
    original_full_code_bytes: bytes,
) -> list[IntermediateChunk]:

    def _finalize_and_append_ast_chunk(nodes: list[Node], parent_text: str, inner_context: str) -> str:
        """Finalizes a chunk and appends it to the main list."""

        if not nodes: return ""

        subchunk_text:str = "".join([get_node_structured_text(node=n, code=original_full_code_bytes) for n in nodes])
        subchunk_text_tcount:int = get_tcount(text=subchunk_text, tokenizer=tokenizer)

        # determine trailing context for the chunk being finalized.
        parent_tokens:int = get_tcount(text=parent_text, tokenizer=tokenizer)
        effective_content_budget:int = max(0, available_tokens-parent_tokens)
        max_trailing_tokens:int = int(effective_content_budget * CONTEXT_PROPORTION_FOR_TRAILING)

        trailing_context_str = get_node_ast_context(
            target_node=nodes[-1],
            source_code=original_full_code_bytes,
            max_token_budget=max_trailing_tokens,
            context_type="trailing",
            tokenizer=tokenizer,
        )

        # inner_context trimming logic
        final_inner_context = inner_context
        inner_context_tcount = get_tcount(text=final_inner_context, tokenizer=tokenizer)
        cap_inner_context = subchunk_text_tcount*CONTEXT_TEXT_RATIO
        # check if the inner_context is disproportionately large compared to the chunk's main text
        if subchunk_text_tcount > 0 and (inner_context_tcount > cap_inner_context):
            trimmed_lines = final_inner_context.splitlines(keepends=True)
            while (get_tcount(text="".join(trimmed_lines), tokenizer=tokenizer) > int(cap_inner_context)) \
                and len(trimmed_lines) > 1:
                trimmed_lines.pop(0)
            final_inner_context = "".join(trimmed_lines)

        full_text = final_inner_context + "\n" + subchunk_text + "\n" + trailing_context_str
        new_chunk: IntermediateChunk = {
            "text": full_text,
            "tcount": get_tcount(text=full_text, tokenizer=tokenizer)
        }
        subchunks.append(new_chunk)

        # return the trailing context for the next chunk.
        return trailing_context_str


    # --- Main function logic ---
    subchunks: list[IntermediateChunk] = []

    if not text_nodes: return []
    current_subchunk_nodes: list[Node] = []
    current_subchunk_text_tokens: int = 0
    current_subchunk_parent_text: str = ""
    next_subchunk_inner_context: str = ""

    for stmt_node in text_nodes:
        if not current_subchunk_nodes: # if first statement of the group, simply start the first chunk and continue.
            current_subchunk_nodes = [stmt_node]
            stmt_text = get_node_structured_text(node=stmt_node, code=original_full_code_bytes)
            current_subchunk_text_tokens = get_tcount(text=stmt_text, tokenizer=tokenizer)
            raw_parent_text = get_enclosing_stmnt_header(
                node=stmt_node,
                source_code=original_full_code_bytes,
                root_node=original_full_tree.root_node
            )
            current_subchunk_parent_text = "" if raw_parent_text == "__FUNCTION_SIGNATURE_PARENT__" else raw_parent_text
            continue

        stmt_text = get_node_structured_text(node=stmt_node, code=original_full_code_bytes)
        stmt_tokens = get_tcount(text=stmt_text, tokenizer=tokenizer)

        # determine parent for the potential new subchunk
        raw_parent_text = get_enclosing_stmnt_header(
            node=stmt_node,
            source_code=original_full_code_bytes,
            root_node=original_full_tree.root_node
        )
        candidate_parent_text = "" if raw_parent_text == "__FUNCTION_SIGNATURE_PARENT__" else raw_parent_text
        candidate_parent_tokens = get_tcount(text=candidate_parent_text, tokenizer=tokenizer)
        # budget for content is what's left after the parent text is accounted for.
        effective_content_budget = max(0, (available_tokens-candidate_parent_tokens))
        # estimate the size if we add this statement to the current group.
        estimated_total_tokens = (
            get_tcount(text=next_subchunk_inner_context, tokenizer=tokenizer)
            + current_subchunk_text_tokens
            + stmt_tokens
        )
        # decide whether to start a new chunk.
        should_start_new_chunk = (
            not current_subchunk_nodes or
            estimated_total_tokens > effective_content_budget or
            candidate_parent_text != current_subchunk_parent_text
        )
        if should_start_new_chunk:
            # finalize the previous chunk before starting a new one.
            trailing_context = _finalize_and_append_ast_chunk(
                nodes=current_subchunk_nodes,
                parent_text=current_subchunk_parent_text,
                inner_context=next_subchunk_inner_context,
            )

            # start the new chunk with the current statement.
            current_subchunk_nodes = [stmt_node]
            current_subchunk_text_tokens = stmt_tokens
            current_subchunk_parent_text = candidate_parent_text
            next_subchunk_inner_context = trailing_context
        else:
            # add the statement to the current chunk.
            current_subchunk_nodes.append(stmt_node)
            current_subchunk_text_tokens += stmt_tokens

    # finalize the last chunk being built.
    _finalize_and_append_ast_chunk(
        nodes=current_subchunk_nodes,
        parent_text=current_subchunk_parent_text,
        inner_context=next_subchunk_inner_context,
    )

    return subchunks

@validate_argument_value(arg_name="label", allowed_values=["0", "1"])
@validate_argument_value(arg_name="trimming_technique", allowed_values=["ast", "line"])
def generate_code_chunks(
    code: str,
    tsp: TreeSitterParser,
    label: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    overlap_stmts: int = STMNTS_OVERLAP,
    trimming_technique: str = "ast"
) -> list[FinalChunk]:

    # parse the entire code
    tree: Tree = tsp.parse(code=code)
    code_bytes: bytes = code.encode(encoding="utf-8")
    captures: Captures = tsp.run_query_on_node(node=tree.root_node, query_str="(function_definition) @func")
    func_nodes: Capture = captures.get("func")
    func_node: TSNode = func_nodes[0] if func_nodes else None

    current_function_signature: str = ""
    func_body_text: str = code
    body_node: TSNode = None
    raw_chunks: list[IntermediateChunk] = []
 
    if func_node:
        current_function_signature = _get_function_signature(func_def_node=func_node, source_code=code)
        body_node = func_node.child_by_field_name("body")
        if body_node and len(body_node.children) > 2: # len(children) > 2 ensures that there are some statements inside the body
            # the first child is '{', the last is '}'. I want everything in between.
            inner_statements = body_node.children[1:-1]
            start_byte = inner_statements[0].start_byte
            end_byte = inner_statements[-1].end_byte
            func_body_text = code_bytes[start_byte:end_byte].decode(encoding="utf-8", errors="replace")
        else:
            func_body_text = ""
    else:
        if trimming_technique == "ast":
            raise ValueError("AST-based chunking requires a function definition.")

    signature_tokens: int = get_tcount(text=current_function_signature, tokenizer=tokenizer)
    available_tokens_for_content: int = max(0, max_tokens-signature_tokens)

    is_tree_valid = not tsp.is_broken_tree(tree) if tree else False
    use_ast_splitting = (trimming_technique == "ast" and is_tree_valid and func_node)

    if use_ast_splitting:
        nodes_for_splitting: list[Node] = get_func_statements(body_node, tsp) if body_node else []
        if nodes_for_splitting:
            raw_chunks = _split_by_ast(
                text_nodes=nodes_for_splitting,
                tokenizer=tokenizer,
                available_tokens=available_tokens_for_content,
                original_full_tree=tree,
                original_full_code_bytes=code_bytes,
            )
    else:  # fallback into line-based chunking
        lines = func_body_text.splitlines(keepends=True)
        units = [{"text": line, "tokens": get_tcount(line, tokenizer)} for line in lines if line.strip()]
        if units:
            raw_chunks = split_by_lines(
                units_for_splitting=units,
                max_tokens=available_tokens_for_content,
                tokenizer=tokenizer,
                overlap_stmts=overlap_stmts,
            )

    # add common metadata
    final_chunks = []
    for chunk in raw_chunks:
        final_chunk = {
            "function_signature": current_function_signature,
            "label": label,
            "text": chunk.get("text", ""),
            "tcount": chunk.get("tcount", 0)
        }
        final_chunks.append(final_chunk)

    return final_chunks

# ========================
# OUTPUT FILES
# ========================
@ensure_jsonl_extension
def save_chunks_to_jsonl(chunks: list[FinalChunk], filepath: str) -> None:
    """Saves a collection of code chunks to a JSONL file.

    Parameters
    ----------
    chunks : Windows
        A collection of code chunks, where each chunk is a dictionary-like
        object ready for JSON serialization.
    filepath : str
        The path to the output file, which is expected to have a '.jsonl'
        extension.

    Raises
    ------
    IOError
        If an error occurs during file writing.
    Exception
        For any other unexpected errors during the saving process.
    """

    try:
        save_to_jsonl(dataset=chunks, filepath=filepath)
        print(f"📂 Successfully saved {filepath} as JSONL 📂")
    except IOError as e:
        raise IOError(f"Error saving JSONL file {filepath}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while saving JSONL to {filepath}: {e}")

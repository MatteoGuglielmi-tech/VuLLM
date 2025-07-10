import csv
import json
import os
from collections.abc import Generator
from typing import Any, Callable, Optional, Union

from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from tree_sitter import Language, Node, Parser, Query, Tree, TreeCursor
from tree_sitter_language_pack import (SupportedLanguage, get_language,
                                       get_parser)

from .decorators import (safeguard_label_values, safeguard_trimming_type,
                        validate_chunking_fn_params,
                        validate_filepath_extension)
from .typedef import *
from .utils import UNUSED

# ========================
# CONFIG
# ========================
BLOCK_TYPES: set[str] = {
    "declaration",
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "return_statement",
    "call_expression",
    "expression_statement",
    "preproc_def",
    "preproc_function_def",
    "preproc_if",
    "preproc_ifdef",
    "preproc_directive",
    "preproc_else",
    "preproc_elif",
}

# <---- Constants for context adjustment ---->
# number of lines to preserve as context during line-wise fallback
STMNTS_OVERLAP: int = 2
# number of lines to
CONTEXT_SIZE: int = 3
# how many tokens to leave free for special chars added by tokenizer
ROOM_FOR_RESERVED_TOKEN: int = 32  # in tokens
# Maximum character length for inner_context
MAX_INNER_CONTEXT_CHARS = 100
# inner_context/trailing_context should not be more than 1.5x text length
CONTEXT_TEXT_RATIO = 1.5
#  25% of available_tokens for inner context
CONTEXT_PROPORTION_FOR_INNER = 0.25
#  25% of available_tokens for trailing context
CONTEXT_PROPORTION_FOR_TRAILING = 0.25
# If text is shorter than this (in tokens), add trailing context
MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT = 50
# <---- Constants for context adjustment ---->


def extract_function_signature(node: Node, code: str) -> str:
    primitive_type: Node | None = node.child_by_field_name("type")
    declarator: Node | None = node.child_by_field_name("declarator")
    # uncomment next line to put signature in one line
    # signature: str = " ".join(
    #     list(
    #         map(
    #             lambda x: x.strip(),
    #             code[declarator.start_byte : declarator.end_byte].strip().splitlines(),
    #         )
    #     )
    # )

    return (
        code[primitive_type.start_byte : primitive_type.end_byte]
        + " "
        + code[declarator.start_byte : declarator.end_byte]
        # + signature
        if declarator and primitive_type
        else "<unknown>"
    )


def is_statement_node(node: Node) -> bool:
    return node.type.endswith("_statement")


def extract_statements_from_node(
    node: Node, code: str, lang: SupportedLanguage = "c"
) -> list[str]:
    statements = []
    node_code: str = code[node.start_byte : node.end_byte]
    seen = set()

    parser: Parser = get_parser(language_name=lang)
    subtree: Tree = parser.parse(node_code.encode(encoding="utf-8"))

    for subnode in traverse_tree(subtree):
        if subnode in seen:
            continue

        if subnode.type in BLOCK_TYPES or subnode.type.endswith("_statement"):
            stmt_text: str = code[subnode.start_byte : subnode.end_byte].strip()
            # ensure block is wrapped
            if subnode.type.endswith("_statement"):
                if not stmt_text.startswith("{"):
                    stmt_text = "{\n" + stmt_text
                if not stmt_text.endswith("}"):
                    stmt_text += "\n}"

            statements.append(stmt_text)
            seen.update(subnode.children)  # avoid nested duplicates

    return statements


def get_node_text_preserving_indentation(node: Node, code: str) -> str:
    lines: list[str] = code.splitlines(keepends=True)
    start_line, _ = node.start_point
    end_line, end_col = node.end_point

    statement_lines = lines[start_line : end_line + 1]
    # Trim last line to correct column
    statement_lines[-1] = statement_lines[-1][:end_col]
    statement_lines = list(filter(None, statement_lines))

    # Join the lines (rstrip to remove final characters)
    statement_text = "".join(statement_lines).rstrip()

    return statement_text


# def collect_top_level_statements(root_node: Node) -> list[Node]:
#     return [node for node in root_node.children if node.type in BLOCK_TYPES]


def collect_function_body_statements(body_node: Node) -> list[Node]:
    """Collects top-level statements and preprocessor directives directly
    within a function's compound_statement body."""

    nodes: list[Node] = []

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
    (preproc_directive) @preproc_directive ; Preprocessor directives inside function body
    (compound_statement) @compound_stmt_nested ; Nested blocks within the function body
    """

    # C_LANGUAGE will never be null at this point based on function calls order
    # this is here to make linter happy
    if (
        TREE_SITTER_LOADED
        and C_LANGUAGE
        and body_node
        and body_node.type == "compound_statement"
    ):
        query: Query = C_LANGUAGE.query(query_string)
        captures: dict[str, list[Node]] = query.captures(body_node)

        all_captured_nodes: list[Node] = []
        for nodes_list in captures.values():
            all_captured_nodes.extend(nodes_list)

        for node in all_captured_nodes:
            # IMPORTANT: Exclude ERROR nodes, as they often indicate parsing issues
            # and can lead to incorrect parent/context determination.
            if node.type == "ERROR":
                continue

            if node.parent == body_node:
                nodes.append(node)

        # sort statements by their start byte
        nodes.sort(key=lambda n: n.start_byte)

    return nodes


def collect_top_level_statements(root_node: TSNode) -> list[Node]:
    """Collects top-level statements within a compound_statement node.

    This is for direct children of the function body.
    """
    nodes: list[Node] = []
    # Query for various top-level constructs, including preprocessor directives
    # (function_definition) @func
    query_string = """
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
    (preproc_directive) @preproc_directive ; Covers all preprocessor directives like #if, #define, #include etc.
    (type_definition) @typedef         ; Covers typedefs at global scope
    (compound_statement) @compound_stmt ; Consider nested blocks as single statements for top-level purposes
    """

    # C_LANGUAGE will never be null at this point based on function calls order
    # this is here to make linter happy
    if C_LANGUAGE and root_node:
        # query for statements
        query: Query = C_LANGUAGE.query(query_string)
        captures: dict[str, list[Node]] = query.captures(root_node)

        all_captured_nodes: list[Node] = []
        for nodes_list in captures.values():
            all_captured_nodes.extend(nodes_list)

        for node in all_captured_nodes:
            # IMPORTANT: Exclude ERROR nodes, as they often indicate parsing issues
            # and can lead to incorrect parent/context determination.
            if node.type == "ERROR":
                continue

            if node.parent == root_node:
                nodes.append(node)

        # sort statements by their start byte
        nodes.sort(key=lambda n: n.start_byte)

    return nodes


def compute_token_count(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """Computes the number of tokens for a given text using the tokenizer."""

    return len(tokenizer.encode(text=text, add_special_tokens=False))


def get_indentation_level(line: str) -> int:
    """Returns the number of leading spaces for a line."""
    return len(line) - len(line.lstrip(" "))


def level_based_split_line_based(
    chunk_metadata: SubChunk,
    tokenizer: PreTrainedTokenizer,
    available_tokens: int,
    partial_context_lines: int,
) -> LBChunks:
    """
    Implements level-based splitting with a progressive approach:
    1. Check if full content (context+text) fits.
    2. Intelligent shrinking of context if needed, prioritizing text content.
    3. Anatomical statement splitting, prioritizing new content over inner context,
       with dynamic inner/trailing context adjustment.

    This line-based version does NOT include function_signature, label, or lang
    in its returned chunks, as these are added by the semantic_sliding_window wrapper.
    """

    subchunks: LBChunks = []

    original_context: str = chunk_metadata["context"]
    text: str = chunk_metadata["text"]
    text_lines: list[str] = text.splitlines(keepends=True)

    # Combine original context and text lines for easier indexing for inner/trailing context
    full_code_str: str = original_context + "\n" + text if original_context else text
    combined_lines: list[str] = full_code_str.splitlines(keepends=True)

    # calculate token limits for contexts
    max_trailing_context_tokens: int = int(
        available_tokens * CONTEXT_PROPORTION_FOR_TRAILING
    )

    # <---- start step 1 ---->

    # check if full content (original context + text) fits inside the usable window
    content: str = "".join(combined_lines).strip()
    content_tcount: int = compute_token_count(text=content, tokenizer=tokenizer)

    single_chunk_data: LBChunk = {
        "parent": "",
        "text": text,
        "inner_context": original_context,
        "trailing_context": "",
    }
    # single_chunk_data: LBChunk = {
    #     "parent": "",
    #     "text": content,
    #     "inner_context": "",
    #     "trailing_context": "",
    # }

    if content_tcount <= available_tokens:
        return [single_chunk_data]

    # <---- end step 1 ---->

    # <---- start step 2 ---->
    # check if shrinking the context may help in fitting the reduced full content.
    if chunk_metadata["text_tcount"] > available_tokens:
        # text itself is too large, move to anatomical splitting
        pass
    else:
        original_context_line_count: int = (
            len(original_context.splitlines(keepends=True)) if original_context else 0
        )
        start_line: int = max(0, original_context_line_count - partial_context_lines)
        reduced_context_lines: list[str] = combined_lines[
            start_line:
        ]  # original_context_line_count
        # combine reduced context with the main text part
        reduced_full_text_combined: str = "".join(
            reduced_context_lines + text_lines
        ).strip()
        reduced_full_text_combined_tcount: int = compute_token_count(
            text=reduced_full_text_combined, tokenizer=tokenizer
        )

        # if it fits after shrinking original context, return as a single chunk
        if reduced_full_text_combined_tcount <= available_tokens:
            reduced_single_chunk_data: LBChunk = {
                "parent": "",
                "text": text,
                "inner_context": "".join(reduced_context_lines).strip(),
                "trailing_context": "",
            }
            return [reduced_single_chunk_data]

        # text_lines: list[str] = combined_lines[original_context_line_count:]
        #
        # reduced_full_text_combined: str = "".join(
        #     reduced_context_lines + text_lines
        # ).strip()
        #
        # reduced_full_text_combined_tcount: int = compute_token_count(
        #     text=reduced_full_text_combined, tokenizer=tokenizer
        # )
        #
        # if reduced_full_text_combined_tcount <= available_tokens:
        #     reduced_single_chunk_data: LBChunk = {
        #         "parent": "",
        #         "text": reduced_full_text_combined,
        #         "inner_context": "",
        #         "trailing_context": "",
        #     }
        #
        #     return [reduced_single_chunk_data]

    # <---- end step 2 ---->

    # <---- start step 3 ---->
    # 3. Perform anatomical statement splitting with dynamic context adjustment
    text_lines_only: list[str] = text.splitlines(keepends=True)
    # text_lines_only: list[str] = combined_lines[original_context_line_count:]

    parent_statement: str = ""
    # token_parent_stmnt: int = 0

    current_subchunk_text_lines: list[str] = []
    current_subchunk_start_idx_in_text_lines_only: int = -1
    current_subchunk_text_tokens: int = 0

    # this variable will hold the inner_context for the next subchunk to be formed.
    # it starts empty, and gets populated by the trailing_context of the previous subchunk.
    next_subchunk_inner_context: str = original_context
    # next_subchunk_inner_context: str = ""

    for idx, stmt_line in enumerate(text_lines_only):
        stmt_tokens: int = compute_token_count(text=stmt_line, tokenizer=tokenizer)

        # determine the inner context that would be used if this statement starts a new subchunk
        potential_inner_context_for_new_subchunk: str = next_subchunk_inner_context
        potential_inner_context_for_new_subchunk_tokens: int = compute_token_count(
            text=potential_inner_context_for_new_subchunk, tokenizer=tokenizer
        )

        estimated_current_subchunk_content_tokens: int = (
            # token_parent_stmnt
            +potential_inner_context_for_new_subchunk_tokens
            + current_subchunk_text_tokens
            + stmt_tokens
        )

        # decision point: does this statement fit in the current subchunk, or does it start a new one?
        if (
            not current_subchunk_text_lines  # start new subchunk
            or estimated_current_subchunk_content_tokens > available_tokens
        ):
            # if there's an existing subchunk, finalize it before starting a new one
            if current_subchunk_text_lines:
                subchunk_text: str = "".join(current_subchunk_text_lines)
                subchunk_text_tcount: int = compute_token_count(
                    text=subchunk_text, tokenizer=tokenizer
                )

                trailing_context_str: str = ""
                trailing_lines: list[str] = []
                # add trailing context if the text is short
                # if subchunk_text_tcount < MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT:
                #     trailing_lines = []
                #     subchunk_end_idx_in_text_lines_only: int = (
                #         current_subchunk_start_idx_in_text_lines_only
                #         + len(current_subchunk_text_lines)
                #         - 1
                #     )
                #
                #     current_subchunk_indent: int = (
                #         get_indentation_level(line=current_subchunk_text_lines[0])
                #         if current_subchunk_text_lines
                #         else 0
                #     )
                #
                #     for j in range(
                #         subchunk_end_idx_in_text_lines_only + 1, len(text_lines_only)
                #     ):
                #         line_to_add: str = text_lines_only[j]
                #         stripped_line_to_add: str = line_to_add.strip()
                #
                #         if stripped_line_to_add == "}":
                #             break
                #         if (
                #             get_indentation_level(line=line_to_add)
                #             < current_subchunk_indent
                #             and stripped_line_to_add != "{"
                #         ):
                #             break
                #
                #         trailing_lines.append(line_to_add)
                #         # Check token count for trailing context
                #         if (
                #             compute_token_count(
                #                 text="".join(trailing_lines), tokenizer=tokenizer
                #             )
                #             > max_trailing_context_tokens
                #         ):
                #             break
                #     trailing_context_str = "".join(trailing_lines)

                # Iterate over lines *after* the current subchunk within the `text_lines_only`
                for j in range(idx, len(text_lines_only)):
                    line_to_add: str = text_lines_only[j]

                    if (
                        compute_token_count(
                            text="".join(trailing_lines) + line_to_add,
                            tokenizer=tokenizer,
                        )
                        > max_trailing_context_tokens
                    ):
                        break
                    trailing_lines.append(line_to_add)

                trailing_context_str = "".join(trailing_lines).strip()

                # finalize inner context for the subchunk being added
                final_inner_context_str: str = next_subchunk_inner_context
                # trim inner_context if it's too large relative to text
                final_inner_context_tcount: int = compute_token_count(
                    text=final_inner_context_str, tokenizer=tokenizer
                )
                if (
                    subchunk_text_tcount > 0
                    and final_inner_context_tcount
                    > subchunk_text_tcount * CONTEXT_TEXT_RATIO
                ):
                    target_len: int = int(subchunk_text_tcount * CONTEXT_TEXT_RATIO)
                    trimmed_str_lines = [
                        str(line)
                        for line in final_inner_context_str.splitlines(keepends=True)
                    ]
                    while (
                        compute_token_count(
                            text="".join(trimmed_str_lines).strip(), tokenizer=tokenizer
                        )
                        > target_len
                        and len(trimmed_str_lines) > 1
                    ):
                        trimmed_str_lines = trimmed_str_lines[1:]
                    final_inner_context_str = "".join(trimmed_str_lines)

                subchunks.append(
                    {
                        "parent": parent_statement,
                        "text": subchunk_text,
                        "inner_context": final_inner_context_str,
                        "trailing_context": trailing_context_str,
                    }
                )

                # after appending, update next_subchunk_inner_context for the *next* iteration's new subchunk
                next_subchunk_inner_context = trailing_context_str

            # Start a new subchunk with the current statement
            current_subchunk_text_lines = [stmt_line]
            current_subchunk_start_idx_in_text_lines_only = idx
            current_subchunk_text_tokens = stmt_tokens
            # For a newly started subchunk, its inner_context is empty.
            next_subchunk_inner_context = ""

        else:
            # Current statement fits into the existing subchunk
            current_subchunk_text_lines.append(stmt_line)
            current_subchunk_text_tokens += stmt_tokens

    # After the loop, append any remaining subchunk that was being built.
    if current_subchunk_text_lines:
        subchunk_text = "".join(current_subchunk_text_lines)
        subchunk_text_tcount = compute_token_count(subchunk_text, tokenizer)

        trailing_context_str = ""
        if subchunk_text_tcount < MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT:
            trailing_lines = []
            # subchunk_end_idx_in_text_lines_only = (
            #     current_subchunk_start_idx_in_text_lines_only
            #     + len(current_subchunk_text_lines)
            #     - 1
            # )
            # current_subchunk_indent = (
            #     get_indentation_level(line=current_subchunk_text_lines[0])
            #     if current_subchunk_text_lines
            #     else 0
            # )

            # for j in range(
            #     subchunk_end_idx_in_text_lines_only + 1, len(text_lines_only)
            # ):
            #     line_to_add: str = text_lines_only[j]
            #     stripped_line_to_add: str = line_to_add.strip()
            #
            #     if stripped_line_to_add == "}":
            #         break
            #     if (
            #         get_indentation_level(line=line_to_add) < current_subchunk_indent
            #         and stripped_line_to_add != "{"
            #     ):
            #         break
            #
            #     trailing_lines.append(line_to_add)
            #     if (
            #         compute_token_count(
            #             text="".join(trailing_lines), tokenizer=tokenizer
            #         )
            #         > max_trailing_context_tokens
            #     ):
            #         break
            # trailing_context_str = "".join(trailing_lines)

            # subchunk_end_idx_in_text_lines_only = len(text_lines_only) # End of text_lines_only
            start_for_trailing_context = (
                current_subchunk_start_idx_in_text_lines_only
                + len(current_subchunk_text_lines)
            )

            for j in range(
                start_for_trailing_context, len(text_lines_only)
            ):  # MODIFIED: Start from idx+1 for lines *after* the last processed
                line_to_add: str = text_lines_only[j]

                if (
                    compute_token_count(
                        text="".join(trailing_lines) + line_to_add, tokenizer=tokenizer
                    )
                    > max_trailing_context_tokens
                ):
                    break
                trailing_lines.append(line_to_add)
            trailing_context_str = "".join(
                trailing_lines
            ).strip()  # MODIFIED: Strip trailing context

        final_inner_context_str: str = next_subchunk_inner_context
        final_inner_context_tcount: int = compute_token_count(
            text=final_inner_context_str, tokenizer=tokenizer
        )

        if (
            subchunk_text_tcount > 0
            and final_inner_context_tcount > subchunk_text_tcount * CONTEXT_TEXT_RATIO
        ):
            target_len: int = int(subchunk_text_tcount * CONTEXT_TEXT_RATIO)
            trimmed_str_lines: list[str] = [
                str(line) for line in final_inner_context_str.splitlines(keepends=True)
            ]

            while (
                compute_token_count(
                    text="".join(trimmed_str_lines).strip(), tokenizer=tokenizer
                )
                > target_len
                and len(trimmed_str_lines) > 1
            ):
                trimmed_str_lines = trimmed_str_lines[1:]
            final_inner_context_str = "".join(trimmed_str_lines)

        subchunks.append(
            {
                "parent": parent_statement,
                "text": subchunk_text,
                "inner_context": final_inner_context_str,
                "trailing_context": trailing_context_str,
            }
        )

    return subchunks


@validate_chunking_fn_params
def semantic_sliding_window(
    chunk: SubChunk,
    tokenizer: PreTrainedTokenizer,
    available_tokens: int,
    function_signature: str,
    label: str,
    lang: SupportedLanguage,
    fn: Callable[..., list[Union[LBChunk, ASTChunk]]],
    _callable_args: Optional[dict[str, Any]] = None,
    **fn_kwargs,
) -> ASTChunks:
    """Applies a semantic sliding window (level-based splitting) to a given
    chunk using a dynamically provided chunking function.

    Args:
        chunk (Split): The initial chunk data containing 'context', 'text', 'nodes'.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for token counting.
        available_tokens (int): The maximum number of tokens allowed per subchunk.
        function_signature (str): The signature of the function containing the code.
        label (str): A label associated with the chunk.
        lang (SupportedLanguage): The programming language of the code, used for final ASTChunk metadata.
        fn (Callable): The chunking function to use (e.g., level_based_split_line_based,
                        level_based_split_with_dynamic_context). This function is expected to
                        return a list of dictionaries compatible with LBChunk or ASTChunk structure.
        _callable_args (Optional[Dict[str, Any]]): Internal argument, prepared by the decorator,
                                                    containing all arguments for 'fn'.
        **fn_kwargs (Any): Additional keyword arguments originally passed to semantic_sliding_window.

    Returns:
        ASTChunks: A list of ASTChunk objects, each enriched with function_signature, label, and lang.

    Raises:
        ValueError: If essential parameters required by the dynamic chunking function are missing
                    (handled by the decorator).
    """

    # these variables are used inside the decorator
    UNUSED(var=chunk)
    UNUSED(var=tokenizer)
    UNUSED(var=available_tokens)
    UNUSED(var=fn_kwargs)
    # chunk_metadata preparation and parameter validation are handled by the decorator.
    if _callable_args is None:
        # safeguard for direct calls without decorator.
        raise RuntimeError(
            "semantic_sliding_window called without decorator-prepared arguments."
        )

    # call the dynamic chunking function
    raw_subchunks: list[LBChunk | ASTChunk] = fn(**_callable_args)

    # convert raw_subchunks to ASTChunks by adding common metadata
    # to be type agnostic
    final_subchunks: ASTChunks = []
    for subchunk_dict in raw_subchunks:
        final_subchunk: ASTChunk = {
            "function_signature": function_signature,
            "label": label,
            "lang": lang,
            "parent": subchunk_dict.get("parent", ""),
            "text": subchunk_dict.get("text", ""),
            "inner_context": subchunk_dict.get("inner_context", ""),
            "trailing_context": subchunk_dict.get("trailing_context", ""),
        }
        final_subchunks.append(final_subchunk)

    # add closing parenthesis to final chunk
    # final_subchunks[-1]["text"] += "\n}"
    return final_subchunks


# ========================
# AST Utilities
# ========================
def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor: TreeCursor = tree.walk()
    visited_children: bool = False

    while True:
        if not visited_children:
            assert cursor.node is not None, "Node not found (Null)"
            yield cursor.node

            # if there are no more children nodes to explore
            # in this branch, tag it as visited
            if not cursor.goto_first_child():
                visited_children = True

        # if there are no more child nodes to explore
        # but there is the sibling at the same level,
        # go there and tag it as unvisited
        elif cursor.goto_next_sibling():
            visited_children = False

        # jump to parent node, at some point
        # all tree will be visited and the exit
        # condition is verified when the cursor node is the root
        elif not cursor.goto_parent():
            break


def node_is_within(child: Node, parent: Node) -> bool:
    # this check is based on byte ranges
    return (child.start_byte >= parent.start_byte) and (
        child.end_byte <= parent.end_byte
    )


def get_node_text(node: TSNode, source_code: bytes) -> str:
    """Extracts the text of a tree-sitter node."""
    assert node is not None

    return source_code[node.start_byte : node.end_byte].decode(
        encoding="utf-8", errors="ignore"
    )


# <---- Tree-sitter Setup ---->
_parser: Optional[Parser] = None
C_LANGUAGE: Optional[Language] = None
TREE_SITTER_LOADED = False

try:
    try:
        # get language to make queries
        C_LANGUAGE = get_language(language_name="c")
        # get parser
        _parser = get_parser(language_name="c")
        # flag as module loaded, this indicated both language and parser has been setup correctly
        TREE_SITTER_LOADED = True
    except Exception as e:
        print(
            f"Warning: Could not load Tree-sitter C language via tree_sitter_language_pack: {e}. "
            "Please ensure 'tree-sitter-language-pack' is installed. "
            "AST-based context will not be available."
        )
        TREE_SITTER_LOADED = False
except ImportError:
    print(
        "Warning: 'tree_sitter' or 'tree_sitter_language_pack' not installed. "
        "Please install them (`pip install tree-sitter tree-sitter-language-pack`) for AST-based context. "
        "Falling back to basic text parsing for context, which is less precise."
    )
    TREE_SITTER_LOADED = False

# <---- Tree-sitter Setup ---->


def get_immediate_parent_statement_text(
    node: Node, source_code: bytes, root_node: Node
) -> str:
    """Finds the text of the closest logical parent statement (e.g., if, for,
    while, function) for a given node by traversing up the AST. `source_code`
    and `root_node` must refer to the entire file's content and AST root.

    Returns:
        str: The text of the parent statement, or "__FUNCTION_SIGNATURE_PARENT__"
             if the immediate parent is the function signature, or "Global Scope"
             if it's a top-level global statement, or an empty string if no relevant parent is found.
    """

    current_ancestor: Node | None = node.parent
    while current_ancestor and current_ancestor != root_node:
        # check if the current ancestor is a function definition
        if current_ancestor.type == "function_definition":
            # if the immediate parent is the function definition, return a special marker.
            return "__FUNCTION_SIGNATURE_PARENT__"

        elif current_ancestor.type in [
            "if_statement",
            "for_statement",
            "while_statement",
            "do_statement",
            "switch_statement",
        ]:
            compound_child: Node | None = next(
                (
                    c
                    for c in current_ancestor.children
                    if c.type == "compound_statement"
                ),
                None,
            )
            if compound_child:
                parent_text: str = (
                    source_code[current_ancestor.start_byte : compound_child.start_byte]
                    .decode(encoding="utf-8", errors="ignore")
                    .strip()
                )
                if parent_text.endswith("{"):
                    parent_text = parent_text[:-1].strip()
                return parent_text
            else:
                # if no compound statement child , just return node text
                return get_node_text(
                    node=current_ancestor, source_code=source_code
                ).strip()

        # move up to the parent
        current_ancestor = current_ancestor.parent

    # if we reach the translation_unit (root_node) or current_ancestor becomes None
    if current_ancestor and current_ancestor.type == "translation_unit":
        # should ideally not be hit for statements inside a function
        return "Global Scope"

    return ""


def get_ast_context(
    target_node: Node,
    root_node: Node,
    source_code: bytes,
    max_token_budget: int,
    context_type: str,
    tokenizer: PreTrainedTokenizer,
) -> str:
    """Extracts AST-based context (inner or trailing) for a target node,
    strictly adhering to a max_token_budget."""

    target_node_idx: int
    sibling: TSNode
    context_content_nodes: list[Node] = []
    current_tokens: int = 0
    siblings: list[TSNode] = []
    final_context_parts: list[str] = []

    if context_type == "inner":
        # inner context: find relevant preceding nodes in the AST hierarchy.

        # 1. Get parent function signature
        func_parent_node: TSNode = target_node.parent
        while func_parent_node and func_parent_node.type not in [
            "function_definition",
            "translation_unit",
        ]:
            func_parent_node = func_parent_node.parent

        if func_parent_node and func_parent_node.type == "function_definition":
            # extract function signature (return type + declarator)
            declarator_node: TSNode = func_parent_node.child_by_field_name("declarator")
            type_node: TSNode = func_parent_node.child_by_field_name("type")

            sig_parts: list[str] = []
            if type_node:
                sig_parts.append(get_node_text(node=type_node, source_code=source_code))
            if declarator_node:
                sig_parts.append(
                    get_node_text(node=declarator_node, source_code=source_code)
                )

            func_sig_text: str = " ".join(sig_parts).strip() + " {"
            func_sig_tokens: int = compute_token_count(
                text=func_sig_text, tokenizer=tokenizer
            )

            if func_sig_tokens <= max_token_budget:
                # add the whole function def node for context
                context_content_nodes.append(func_parent_node)
                current_tokens += func_sig_tokens
            else:
                # function signature alone exceeds budget, no context possible
                return ""

        # 2. Traverse up to find enclosing compound statements (blocks) and their opening lines
        ancestor: TSNode = target_node.parent
        block_openers_nodes: list[Node] = []

        while ancestor and ancestor != root_node:
            if ancestor.type == "compound_statement" and ancestor.children:
                owner_stmt: TSNode = ancestor.parent
                if owner_stmt and owner_stmt.type in [
                    "if_statement",
                    "for_statement",
                    "while_statement",
                    "do_statement",
                    "switch_statement",
                ]:
                    # get text of the owner statement up to the block start
                    block_openers_nodes.insert(0, owner_stmt)

            ancestor = ancestor.parent

        # Add block openers to context, respecting budget
        for opener_node in block_openers_nodes:
            node_text: str = get_node_text(node=opener_node, source_code=source_code)
            node_tokens: int = compute_token_count(text=node_text, tokenizer=tokenizer)
            if (current_tokens + node_tokens) <= max_token_budget:
                context_content_nodes.append(opener_node)
                current_tokens += node_tokens
            else:
                break

        # 3. Add preceding siblings within the same block, if space allows
        if target_node.parent:
            siblings = list(target_node.parent.children)
            target_node_idx = siblings.index(target_node)

            for i in range(target_node_idx - 1, -1, -1):
                sibling: TSNode = siblings[i]

                assert sibling is not None
                if sibling.type in ["comment", "{", "}"]:
                    continue

                sibling_text: str = get_node_text(node=sibling, source_code=source_code)
                sibling_tokens: int = compute_token_count(
                    text=sibling_text, tokenizer=tokenizer
                )

                if (current_tokens + sibling_tokens) <= max_token_budget:
                    context_content_nodes.insert(
                        len(context_content_nodes) - len(block_openers_nodes), sibling
                    )
                    current_tokens += sibling_tokens
                else:
                    break

        # reconstruct the context string from the collected nodes, ensuring correct order
        # robust solution: sort by start_byte
        unique_nodes = sorted(
            list(set(context_content_nodes)), key=lambda n: n.start_byte
        )

        final_context_parts = []
        for node in unique_nodes:
            node_text: str = get_node_text(node=node, source_code=source_code)
            if (
                compute_token_count(
                    text=("".join(final_context_parts) + node_text), tokenizer=tokenizer
                )
                <= max_token_budget
            ):
                final_context_parts.append(node_text)
            else:
                # budget exceeded during final assembly
                break

        return "".join(final_context_parts).strip()

    elif context_type == "trailing":
        # trailing context: find relevant subsequent nodes within the same block.
        if target_node.parent and target_node.parent.type == "compound_statement":
            siblings = list(target_node.parent.children)
            target_node_idx = siblings.index(target_node)

            for i in range(target_node_idx + 1, len(siblings)):
                sibling = siblings[i]

                assert sibling is not None
                # skip comments, based on pre-processing no comments should be found
                if sibling.type in ["comment"]:
                    continue

                sibling_text = get_node_text(node=sibling, source_code=source_code)
                sibling_tokens = compute_token_count(
                    text=sibling_text, tokenizer=tokenizer
                )

                if (current_tokens + sibling_tokens) <= max_token_budget:
                    context_content_nodes.append(sibling)
                    current_tokens += sibling_tokens
                else:
                    break

        if target_node.parent and target_node.parent.type == "compound_statement":
            # find the last child of the compound statement, which is usually '}'
            last_child: Node = target_node.parent.children[-1]
            if last_child.type == "}" and last_child.start_byte > target_node.end_byte:
                brace_text: str = get_node_text(
                    node=last_child, source_code=source_code
                )
                brace_tokens: int = compute_token_count(
                    text=brace_text, tokenizer=tokenizer
                )
                if current_tokens + brace_tokens <= max_token_budget:
                    context_content_nodes.append(last_child)
                    current_tokens += brace_tokens

    # reconstruct the trailing context from collected nodes
    final_context_parts = []
    for node in context_content_nodes:
        node_text = get_node_text(node=node, source_code=source_code)
        if (
            compute_token_count(
                text=("".join(final_context_parts) + node_text), tokenizer=tokenizer
            )
            <= max_token_budget
        ):
            final_context_parts.append(node_text)
        else:
            break

    return "".join(final_context_parts).strip()


# ========================
# CHUNK EXTRACTION
# ========================
def split_large_chunk(
    # code_bytes: bytes,
    units_for_splitting: list[dict[str, Any]],
    # statement_nodes: list[Node],
    max_tokens: int,
    tokenizer: PreTrainedTokenizer,
    overlap_stmts: int = STMNTS_OVERLAP,
) -> SubChunks:
    """Splits a large chunk (e.g., a function body) into smaller sub-chunks
    based on statement nodes, ensuring each sub-chunk fits within max_tokens
    and includes a specified overlap of statements for context.

    The `max_tokens` here is the budget for (parent + inner + text + trailing).
    """

    subgroups: SubChunks = []
    current_group_texts: list[str] = []
    current_group_tokens: int = 0
    current_group_nodes: list[Node] = []

    running_overlap_context_texts: list[str] = []
    context_text: str = ""

    # for idx, stmt in enumerate(statement_nodes):
    for idx, unit in enumerate(units_for_splitting):
        stmt_text: str = unit.get("text", "")  # MODIFIED: Get text from unit
        stmt_tokens: int = unit.get("tokens", 0)  # MODIFIED: Get tokens from unit
        stmt_node: Optional[Node] = unit.get("node")  # NEW: Get node if available
        # stmt_text: str = get_node_text_preserving_indentation(
        #     node=stmt, code=code_bytes.decode(encoding="utf-8")
        # )
        # stmt_tokens: int = compute_token_count(text=stmt_text, tokenizer=tokenizer)

        context_for_current_group_text: str = ""
        if idx >= overlap_stmts:
            # nodes_for_context: list[Node] = statement_nodes[idx - overlap_stmts : idx]
            nodes_for_context_units: list[dict[str, Any]] = units_for_splitting[
                idx - overlap_stmts : idx
            ]  # MODIFIED

            context_for_current_group_text: str = "\n".join(
                [
                    # get_node_text_preserving_indentation(
                    #     node=cn, code=code_bytes.decode(encoding="utf-8")
                    # )
                    # for cn in nodes_for_context
                    u.get("text", "")  # MODIFIED: Get text from unit
                    for u in nodes_for_context_units
                ]
            )

        context_for_current_group_tokens: int = compute_token_count(
            text=context_for_current_group_text, tokenizer=tokenizer
        )

        if not current_group_texts:
            if (stmt_tokens + context_for_current_group_tokens) > max_tokens:
                subgroups.append(
                    {
                        "context": remove_empty_lines(
                            text=context_for_current_group_text
                        ),
                        "text": remove_empty_lines(text=stmt_text),
                        "context_tcount": context_for_current_group_tokens,
                        "text_tcount": stmt_tokens,
                        # store node if present
                        "nodes": ([stmt_node] if stmt_node else []),
                        # "nodes": [stmt],
                    }
                )
                (
                    current_group_texts,
                    current_group_nodes,
                    running_overlap_context_texts,
                ) = []
                current_group_tokens = 0
                continue
            else:
                current_group_texts.append(stmt_text)
                current_group_tokens += stmt_tokens
                if stmt_node:
                    # append node if present
                    current_group_nodes.append(stmt_node)

                # current_group_nodes.append(stmt)
                running_overlap_context_texts = [stmt_text]
                context_text = context_for_current_group_text
        else:
            if (current_group_tokens + stmt_tokens) > max_tokens:
                subgroups.append(
                    {
                        "context": remove_empty_lines(text=context_text),
                        "text": remove_empty_lines(text="\n".join(current_group_texts)),
                        "context_tcount": compute_token_count(
                            text=context_text, tokenizer=tokenizer
                        ),
                        "text_tcount": current_group_tokens,
                        "nodes": current_group_nodes,
                    }
                )
                current_group_texts = [stmt_text]
                current_group_tokens = stmt_tokens
                current_group_nodes = [stmt_node] if stmt_node else []
                # current_group_nodes = [stmt]
                running_overlap_context_texts = [stmt_text]
                context_text = context_for_current_group_text
            else:
                current_group_texts.append(stmt_text)
                current_group_tokens += stmt_tokens
                if stmt_node:
                    current_group_nodes.append(stmt_node)
                # current_group_nodes.append(stmt)
                running_overlap_context_texts.append(stmt_text)
                if len(running_overlap_context_texts) > overlap_stmts:
                    running_overlap_context_texts.pop(0)

    if current_group_texts:
        subgroups.append(
            {
                "context": remove_empty_lines(text=context_text),
                "text": remove_empty_lines(text="\n".join(current_group_texts)),
                "context_tcount": compute_token_count(
                    text=context_text, tokenizer=tokenizer
                ),
                "text_tcount": current_group_tokens,
                "nodes": current_group_nodes,
            }
        )

    return subgroups


def remove_empty_lines(text: str) -> str:
    return os.linesep.join([s for s in text.splitlines() if s])


def diff_two_strings(s1, s2):
    import difflib

    s = difflib.SequenceMatcher(None, s1, s2, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            print(
                "{:7}   a[{}:{}] --> b[{}:{}] 0{!r:>8} --> {!r}".format(
                    tag, i1, i2, j1, j2, s1[i1:i2], s2[j1:j2]
                )
            )


def level_based_split_ast_based(
    chunk_metadata: SubChunk,
    tokenizer: PreTrainedTokenizer,
    available_tokens: int,
    original_full_tree: Tree,
    original_full_code_bytes: bytes,
) -> ASTChunks:
    """Implements level-based splitting with an AST-based context management
    for C code.

    This version relies on tree-sitter to identify logical code blocks
    and statements, and strictly enforces the available_tokens limit for
    the complete chunk.
    """

    subchunks: ASTChunks = []
    trimmed_str_lines: list[str] = []

    fixed_fields: ASTChunk = {
        "function_signature": "",
        "label": "",
        "lang": "",
        "parent": "",
        "text": "",
        "inner_context": "",
        "trailing_context": "",
    }

    original_context_str: str = chunk_metadata["context"]
    text_str: str = chunk_metadata["text"]
    text_nodes_info_from_split_large_chunk: list[Node] = chunk_metadata.get("nodes", [])

    original_file_root_node: Node = original_full_tree.root_node

    # calc token limits for contexts
    max_trailing_context_tokens_overall_proportion: float = (
        CONTEXT_PROPORTION_FOR_TRAILING
    )

    full_code_str_with_newline: str = (
        original_context_str + "\n" + text_str if original_context_str else text_str
    )

    # <---- Handle case where the entire 'text_str' fits (Step 1) ---->
    parent_text_for_single_chunk: str = ""
    inner_context_for_single_chunk: str = ""

    if text_nodes_info_from_split_large_chunk:
        first_node_of_chunk: Node = text_nodes_info_from_split_large_chunk[0]
        raw_parent_text: str = get_immediate_parent_statement_text(
            node=first_node_of_chunk,
            source_code=original_full_code_bytes,
            root_node=original_file_root_node,
        )

        # 'parent' field: empty if function signature, else raw parent
        if raw_parent_text == "__FUNCTION_SIGNATURE_PARENT__":
            parent_text_for_single_chunk = ""
        else:
            parent_text_for_single_chunk = raw_parent_text

    # verify token lenght for possible single chunk
    # inner_context not taken into account bc 0
    total_tokens_for_single_chunk: int = compute_token_count(
        text=parent_text_for_single_chunk, tokenizer=tokenizer
    ) + compute_token_count(text=full_code_str_with_newline, tokenizer=tokenizer)

    if total_tokens_for_single_chunk <= available_tokens:
        fixed_fields["parent"] = parent_text_for_single_chunk
        fixed_fields["text"] = full_code_str_with_newline
        fixed_fields["inner_context"] = inner_context_for_single_chunk
        return [fixed_fields]

    # <---- End Step 1 ---->

    # <---- Proceed to AST-based splitting (Step 2) ---->
    text_nodes_info: list[Node] = text_nodes_info_from_split_large_chunk

    if not text_nodes_info:
        # fallback if no specific statements found
        parent_text_for_fallback: str = ""
        inner_context_for_fallback: str = ""

        if chunk_metadata["text_tcount"] <= available_tokens:
            if text_nodes_info_from_split_large_chunk:
                first_node_of_chunk: Node = text_nodes_info_from_split_large_chunk[0]
                raw_parent_text: str = get_immediate_parent_statement_text(
                    node=first_node_of_chunk,
                    source_code=original_full_code_bytes,
                    root_node=original_file_root_node,
                )

                if raw_parent_text == "__FUNCTION_SIGNATURE_PARENT__":
                    parent_text_for_fallback = ""
                else:
                    parent_text_for_fallback = raw_parent_text

            fixed_fields["parent"] = parent_text_for_fallback
            fixed_fields["text"] = text_str
            fixed_fields["inner_context"] = inner_context_for_fallback
            return [fixed_fields]

        else:
            return []

    current_subchunk_nodes: list[Node] = []
    current_subchunk_text_tokens: int = 0
    current_subchunk_parent_text: str = ""

    max_trailing_context_for_this_subchunk: int = 0

    # this variable will hold the inner_context for the next subchunk to be formed.
    # it starts empty, and gets populated by the trailing_context of the previous subchunk.
    next_subchunk_inner_context: str = ""

    for _, stmt_node in enumerate(text_nodes_info):
        stmt_text: str = get_node_text(
            node=stmt_node, source_code=original_full_code_bytes
        )
        stmt_tokens: int = compute_token_count(text=stmt_text, tokenizer=tokenizer)

        # determine parent for the potential new subchunk
        candidate_parent_text_raw: str = get_immediate_parent_statement_text(
            node=stmt_node,
            source_code=original_full_code_bytes,
            root_node=original_file_root_node,
        )
        candidate_parent_text_for_group: str = ""

        if candidate_parent_text_raw == "__FUNCTION_SIGNATURE_PARENT__":
            candidate_parent_text_for_group = ""
        else:
            candidate_parent_text_for_group = candidate_parent_text_raw

        candidate_parent_tokens = compute_token_count(
            text=candidate_parent_text_for_group, tokenizer=tokenizer
        )

        effective_content_budget: int = max(
            0, available_tokens - candidate_parent_tokens
        )

        max_trailing_context_for_this_subchunk: int = int(
            effective_content_budget * max_trailing_context_tokens_overall_proportion
        )

        # determine the inner context that would be used if this statement starts a new subchunk
        potential_inner_context_for_new_subchunk: str = next_subchunk_inner_context

        estimated_current_subchunk_content_tokens: int = (
            compute_token_count(
                text=potential_inner_context_for_new_subchunk, tokenizer=tokenizer
            )
            + current_subchunk_text_tokens
            + stmt_tokens
        )

        # decision point: does this statement fit in the current subchunk, or does it start a new one?
        # A new subchunk is started if:
        # 1. current_subchunk_nodes is empty (always start a new subchunk with the first statement)
        # 2. adding the current statement would exceed the effective content budget for the current subchunk.
        # 3. the parent of the current statement changes from the parent of the current subchunk being built.
        if (
            not current_subchunk_nodes
            or estimated_current_subchunk_content_tokens > effective_content_budget
            or candidate_parent_text_for_group != current_subchunk_parent_text
        ):

            # if there's an existing subchunk, finalize it before starting a new one
            if current_subchunk_nodes:
                subchunk_text: str = "".join(
                    [
                        get_node_text(node=n, source_code=original_full_code_bytes)
                        for n in current_subchunk_nodes
                    ]
                )
                subchunk_text_tcount: int = compute_token_count(
                    text=subchunk_text, tokenizer=tokenizer
                )

                # determine trailing context for the just completed subchunk
                trailing_context_str: str = ""
                # add trailing context if the text is short or it's a nested split
                if (
                    subchunk_text_tcount < MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT
                    or current_subchunk_parent_text != ""
                ):
                    last_node_of_current_subchunk: Node = current_subchunk_nodes[-1]
                    trailing_context_str = get_ast_context(
                        target_node=last_node_of_current_subchunk,
                        root_node=original_file_root_node,
                        source_code=original_full_code_bytes,
                        max_token_budget=max_trailing_context_for_this_subchunk,
                        context_type="trailing",
                        tokenizer=tokenizer,
                    )

                # finalize inner context for the subchunk being added
                final_inner_context_str: str = next_subchunk_inner_context
                # trim inner_context if it's too large relative to text
                final_inner_context_tcount: int = compute_token_count(
                    text=final_inner_context_str, tokenizer=tokenizer
                )

                trimmed_str_lines = []

                if (
                    subchunk_text_tcount > 0
                    and final_inner_context_tcount
                    > subchunk_text_tcount * CONTEXT_TEXT_RATIO
                ):
                    target_len_tokens: int = int(
                        subchunk_text_tcount * CONTEXT_TEXT_RATIO
                    )
                    # this super ugly thing to avoid Pyright linting error
                    trimmed_str_lines = [
                        str(line)
                        for line in final_inner_context_str.splitlines(keepends=True)
                    ]
                    while (
                        compute_token_count(
                            text="".join(trimmed_str_lines).strip(), tokenizer=tokenizer
                        )
                        > target_len_tokens
                        and len(trimmed_str_lines) > 1
                    ):
                        trimmed_str_lines = trimmed_str_lines[1:]
                    final_inner_context_str = "".join(trimmed_str_lines)

                fixed_fields["parent"] = current_subchunk_parent_text
                fixed_fields["text"] = subchunk_text
                fixed_fields["inner_context"] = final_inner_context_str
                fixed_fields["trailing_context"] = trailing_context_str
                subchunks.append(fixed_fields)

                # update next_subchunk_inner_context
                next_subchunk_inner_context = trailing_context_str

            # ttart a new subchunk with the current statement
            current_subchunk_nodes = [stmt_node]
            current_subchunk_text_tokens = stmt_tokens
            current_subchunk_parent_text = candidate_parent_text_for_group

            next_subchunk_inner_context = ""

        else:
            # current statement fits into the existing subchunk
            current_subchunk_nodes.append(stmt_node)
            current_subchunk_text_tokens += stmt_tokens

    # after the loop, append any remaining subchunk that was being built.
    if current_subchunk_nodes:
        subchunk_text = "".join(
            [
                get_node_text(node=n, source_code=original_full_code_bytes)
                for n in current_subchunk_nodes
            ]
        )
        subchunk_text_tcount = compute_token_count(
            text=subchunk_text, tokenizer=tokenizer
        )

        trailing_context_str = ""
        if (
            subchunk_text_tcount < MIN_TEXT_TOKENS_FOR_TRAILING_CONTEXT
            or current_subchunk_parent_text != ""
        ):
            last_node_of_current_subchunk: Node = current_subchunk_nodes[-1]
            trailing_context_str = get_ast_context(
                target_node=last_node_of_current_subchunk,
                root_node=original_file_root_node,
                source_code=original_full_code_bytes,
                max_token_budget=max_trailing_context_for_this_subchunk,
                context_type="trailing",
                tokenizer=tokenizer,
            )

        final_inner_context_str = next_subchunk_inner_context
        final_inner_context_tcount = compute_token_count(
            text=final_inner_context_str, tokenizer=tokenizer
        )

        if (
            subchunk_text_tcount > 0
            and final_inner_context_tcount > subchunk_text_tcount * CONTEXT_TEXT_RATIO
        ):
            target_len_tokens = int(subchunk_text_tcount * CONTEXT_TEXT_RATIO)
            trimmed_str_lines = [
                str(line) for line in final_inner_context_str.splitlines(keepends=True)
            ]
            while (
                compute_token_count("".join(trimmed_str_lines).strip(), tokenizer)
                > target_len_tokens
                and len(trimmed_str_lines) > 1
            ):
                trimmed_str_lines = trimmed_str_lines[1:]
            final_inner_context_str = "".join(trimmed_str_lines)

        fixed_fields["parent"] = current_subchunk_parent_text
        fixed_fields["text"] = subchunk_text
        fixed_fields["inner_context"] = final_inner_context_str
        fixed_fields["trailing_context"] = trailing_context_str
        subchunks.append(fixed_fields)

    return subchunks


@safeguard_label_values
@safeguard_trimming_type
def extract_structured_chunks_with_context(
    code: str,
    label: str,
    tokenizer: PreTrainedTokenizer,
    max_tokens: int,
    overlap_stmts: int = STMNTS_OVERLAP,
    context_to_preserve: int = CONTEXT_SIZE,
    lang: SupportedLanguage = "c",
    trimming_technique: str = "ast",  # "ast" or "line"
) -> Windows:
    """Extracts structured chunks with context from a given code snippet. This
    function acts as the orchestrator for different chunking strategies.

    Assumes each `code` input contains exactly one valid function definition.
    """

    if not TREE_SITTER_LOADED:
        # fallback if Tree-sitter is not loaded, force line-based
        if trimming_technique == "ast":
            print(
                "Warning: Tree-sitter not loaded, falling back to line-based chunking."
            )
        trimming_technique = "line"

    if not _parser or not C_LANGUAGE:
        raise ValueError("Not all Tree-sitter components have been properly set.")

    # parse the entire code
    full_code_bytes: bytes = code.encode(encoding="utf-8")
    full_tree: Tree = _parser.parse(full_code_bytes)

    # find single function definition (guaranteed by pre-processing)
    func_node: Optional[Node] = None
    for node in traverse_tree(tree=full_tree):
        if node.type == "function_definition":
            func_node = node
            break

    # if func_node is still None, it means the pre-processing guarantee was violated
    # or there's an issue with parsing. Raise an error
    current_function_signature_tokens: int = 0
    units_for_splitting: list[dict[str, Any]] = []
    if func_node is None:
        if trimming_technique == "ast":
            raise ValueError(
                "AST-based chunking expected a function definition but none was found. Pre-processing guarantee violated?"
            )
        # for line-based, we treat the entire code as the initial "function"
        current_function_signature: str = ""
        # For line-based fallback, if no function node, treat entire code as lines
        if trimming_technique == "line":  # NEW
            lines = code.splitlines(keepends=True)  # NEW
            units_for_splitting = [
                {"text": line, "tokens": compute_token_count(line, tokenizer)}
                for line in lines
                if line.strip()
            ]
        # # If no function node is found, the entire code is treated as the "function text"
        # # and top_level_statement_nodes will be collected from the root.
        # # In this specific scenario (single function input), this path might indicate an issue.
        # top_level_nodes_for_splitting: list[Node] = []

    else:
        assert func_node.text is not None
        current_function_signature = extract_function_signature(
            node=func_node, code=code
        )
        current_function_signature_tokens = compute_token_count(
            text=current_function_signature, tokenizer=tokenizer
        )

        # collect top-level statements from the function's body
        func_body_node: TSNode = func_node.child_by_field_name("body")
        func_body_text: str = get_node_text(
            node=func_body_node, source_code=full_code_bytes
        )

        # MODIFIED
        # clean from initial { and \n
        # func_body_text = func_body_text.replace("{", "", 1).replace("\n", "", 1)
        if func_body_node and func_body_node.type == "compound_statement":
            # find the actual content between the first '{' and last '}'
            first_brace_idx: int = func_body_text.find("{")
            last_brace_idx: int = func_body_text.rfind("}")
            if (
                first_brace_idx != -1
                and last_brace_idx != -1
                and last_brace_idx > first_brace_idx
            ):
                func_body_text = func_body_text[
                    first_brace_idx + 1 : last_brace_idx
                ].strip()
            else:
                func_body_text = func_body_text.strip()  # fallback if braces not found
        else:
            func_body_text = (
                func_body_text.strip()
            )  # if not a compound statement, take all text
        # MODIFIED END

        # in case the whole code fits inside available_token window, just return
        if compute_token_count(text=code, tokenizer=tokenizer) <= max_tokens:
            return [
                ASTChunk(
                    function_signature=current_function_signature,
                    label=label,
                    lang=lang,
                    parent="",
                    text=func_body_text,
                    inner_context="",
                    trailing_context="",
                )
            ]

        if func_body_node is None or func_body_node.type != "compound_statement":
            if trimming_technique == "ast":
                raise ValueError(
                    "AST-based chunking expected a compound statement for function body but none was found."
                )
            # if body is malformed and line-based, fall back to splitting entire function text by lines
            if trimming_technique == "line":
                lines: list[str] = func_body_text.splitlines(keepends=True)
                units_for_splitting = [
                    {
                        "text": line,
                        "tokens": compute_token_count(text=line, tokenizer=tokenizer),
                    }
                    for line in lines
                    if line.strip()
                ]
            # # fallback for line-based if body is malformed, treat entire function text as statements
            # top_level_nodes_for_splitting = []
        else:
            if trimming_technique == "ast":
                # For AST-based, collect top-level statements from the function's body
                ast_nodes_for_splitting: list[Node] = (
                    collect_function_body_statements(body_node=func_body_node)
                    if C_LANGUAGE
                    else []
                )
                units_for_splitting = [
                    {
                        "node": n,
                        "text": get_node_text_preserving_indentation(node=n, code=code),
                        "tokens": compute_token_count(
                            text=get_node_text_preserving_indentation(
                                node=n, code=code
                            ),
                            tokenizer=tokenizer,
                        ),
                    }
                    for n in ast_nodes_for_splitting
                ]
            elif trimming_technique == "line":
                # for line-based, prepare units from function body lines
                # lines = func_body_text.strip().splitlines(keepends=True)
                lines = func_body_text.splitlines(keepends=True)

                units_for_splitting = [
                    {
                        "text": line,
                        "tokens": compute_token_count(text=line, tokenizer=tokenizer),
                    }
                    for line in lines
                    if line.strip()
                ]

        #     # for AST-based, collect top-level statements from the function's body
        #     top_level_nodes_for_splitting: list[Node] = (
        #         collect_function_body_statements(body_node=func_body_node)
        #         if C_LANGUAGE
        #         else []
        #     )
        # if trimming_technique == "line":
        #     # pass the entire function body text to semantic_sliding_window.
        #     intermediate_splits: SubChunks = [
        #         {
        #             "context": "",
        #             "text": func_body_text,
        #             "context_tcount": 0,
        #             "text_tcount": compute_token_count(
        #                 text=func_body_text,
        #                 tokenizer=tokenizer,
        #             ),
        #             "nodes": [],
        #         }
        #     ]
        #     all_processed_chunks: Windows = []
        #     # directly process with semantic_sliding_window for line-based
        #     subchunks: ASTChunks = semantic_sliding_window(
        #         chunk=intermediate_splits[0],
        #         tokenizer=tokenizer,
        #         available_tokens=max_tokens,
        #         function_signature=current_function_signature,
        #         label=label,
        #         lang=lang,
        #         fn=level_based_split_line_based,
        #         partial_context_lines=context_to_preserve,
        #     )
        #     all_processed_chunks.extend(subchunks)
        #
        #     return all_processed_chunks
        #
    all_processed_chunks: Windows = []

    # Calculate available tokens for the *content* of the chunk, excluding the signature
    available_tokens: int = max(0, max_tokens - current_function_signature_tokens)

    # Step 1: Use split_large_chunk to break down the full function text (or entire code)
    # into manageable segments, with overlap.
    intermediate_splits: SubChunks = split_large_chunk(
        # code_bytes=full_code_bytes,
        # statement_nodes=top_level_nodes_for_splitting,
        units_for_splitting=units_for_splitting,
        max_tokens=available_tokens,
        tokenizer=tokenizer,
        overlap_stmts=overlap_stmts,
    )

    # Step 2: Iterate over these intermediate splits and apply semantic_sliding_window
    for intermediate_split_chunk in intermediate_splits:
        fn_to_use: Callable[..., Union[LBChunks, ASTChunks]]
        fn_specific_kwargs: dict[str, Any] = {}

        if trimming_technique == "ast":
            fn_to_use = level_based_split_ast_based
            # tree-sitter components are guaranteed to be available here due to initial check
            fn_specific_kwargs = {
                "original_full_tree": full_tree,
                "original_full_code_bytes": full_code_bytes,
            }
        else:  # type_chunking == "line"
            fn_to_use = level_based_split_line_based
            fn_specific_kwargs = {
                "partial_context_lines": context_to_preserve,
            }

        # call semantic_sliding_window with the chosen function and its specific kwargs
        subchunks: ASTChunks = semantic_sliding_window(
            chunk=intermediate_split_chunk,
            tokenizer=tokenizer,
            available_tokens=available_tokens,
            function_signature=current_function_signature,
            label=label,
            lang=lang,
            fn=fn_to_use,
            **fn_specific_kwargs,
        )
        all_processed_chunks.extend(subchunks)

    return all_processed_chunks


# ========================
# OUTPUT FILES
# ========================
@validate_filepath_extension
def save_outputs(chunks: Windows, filepath: str) -> None:
    """Saves the provided chunks  to a file. The output format (JSONL or CSV)
    is determined by the file extension of the 'filepath' argument. The
    decorator ensures 'filepath' has a valid extension (.jsonl or .csv).

    Args:
        chunks: An iterable of dictionaries, representing the data to save.
        filepath: The path to the output file, including the desired extension.
                  The decorator ensures this is either '.jsonl' or '.csv'.
    """
    _, ext = os.path.splitext(filepath)

    # <---- JSONL ---->
    if ext.lower() == ".jsonl":
        # save as JSONL
        try:
            with open(file=filepath, mode="w", encoding="utf-8") as f:
                for entry in chunks:
                    f.write(json.dumps(entry) + "\n")

            # 🗂️, 💾
            print(f"📂 Successfully saved {filepath} as JSONL 📂")

        except IOError as e:
            print(f"Error saving JSONL file {filepath}: {e}")

        except Exception as e:
            print(f"An unexpected error occurred while saving JSONL to {filepath}: {e}")
    # <---- JSONL ---->

    # <---- CSV ---->
    elif ext.lower() == ".csv":
        # save as CSV
        if not chunks:
            print(f"Warning: Chunks are empty. No CSV data to write to {filepath}.")
            return

        # these should match the keys in your chunk dictionaries.
        fieldnames = [
            "function_signature",
            "label",
            "lang",
            "parent",
            "text",
            "inner_context",
        ]

        # check if all chunks contain the required fieldnames
        for i, chunk in enumerate(chunks):
            for field in fieldnames:
                if field not in chunk:
                    print(
                        f"Warning: Chunk at index {i} is missing field '{field}'. "
                        f"CSV output for {filepath} might be incomplete or incorrect."
                    )
                    break

        try:
            with open(file=filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    lineterminator="\n",  # ensures consistent EOL
                )
                writer.writeheader()
                writer.writerows(chunks)
            print(f"📂 Successfully saved {filepath} as CSV 📂")

        except IOError as e:
            print(f"Error saving CSV file {filepath}: {e}")

        except Exception as e:
            print(f"An unexpected error occurred while saving CSV to {filepath}: {e}")

    # <---- CSV ---->

    else:
        # shouldn't be reached. Safeguard
        print(
            f"Error: Unhandled file extension '{ext}' for filepath: {filepath}. No data saved."
        )


# ========================
# Main CLI
# ========================
def main():

    long_code = """int calculate_stats(int *array, int n_elements, int filter_threshold) {
  int sum_positive = 0;
  #if 1>0
  int count_positive = 0;
  #endif
  int sum_negative = 0;
  int count_negative = 0;
  if (array == NULL || n_elements <= 0) {
      printf("Invalid input array or size.\\n");
      return -1;
  }
  for (int i = 0; i < n_elements; i++) {
      int current_val = array[i];
      if (current_val > 0) {
          if (current_val > filter_threshold) {
              sum_positive += current_val;
              count_positive++;
          }
      } else if (current_val < 0) {
          sum_negative += current_val;
          count_negative++;
      } else { 
      }
  }
  if (count_positive > 0) { 
      printf("Average positive: %f\\n", (double)sum_positive / count_positive);
  } else {
      printf("No positive numbers found.\\n");
  }
  if (count_negative > 0) {
      printf("Average negative: %f\\n", (double)sum_negative / count_negative);
  }
  return sum_positive + sum_negative;
}"""

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Process
    chunks: Windows = extract_structured_chunks_with_context(
        code=long_code,
        label="0",
        tokenizer=tokenizer,
        # max_tokens=usable_token_window,
        max_tokens=100,
        trimming_technique="line",  # "ast" or "line"
    )

    save_outputs(chunks=chunks, filepath="./data/chunks.jsonl")


if __name__ == "__main__":
    main()

import pytest
from unittest.mock import MagicMock
from transformers import AutoTokenizer

from core.chunking_and_streaming.shared.tree_sitter_parser import TreeSitterParser
from core.chunking_and_streaming.chunking import (
    _get_function_signature, get_node_structured_text,
    _find_parent_function, get_func_statements, save_chunks_to_jsonl,
    split_by_lines, _split_by_ast, generate_code_chunks
)


# --- Test Data: Sample C Code ---

C_CODE_SIMPLE = """void simple_function(int x) {
    int a = x + 1;
    if (a > 10) {
        printf("Greater");
    }
}"""

C_CODE_NESTED = """void nested_loop_function(int count) {
    for (int i = 0; i < count; i++) {
        if (i % 2 == 0) {
            while (1) {
                break;
            }
        }
    }
    return;
}"""

C_CODE_PREPROCESSOR = """void preprocessor_function(int mode) {
    #define MAX_VALUE 100
    int value = mode;
    #ifdef DEBUG
        printf("Mode is: %d\\n", mode);
    #else
        printf("Mode is: normal\\n");
    #endif
    if (value > MAX_VALUE) {
        value = MAX_VALUE;
    }
}
"""

C_CODE_NO_FUNC = "int global_var = 100;"

# --- Pytest Fixtures for Reusability ---
@pytest.fixture(scope="module")
def c_parser() -> TreeSitterParser:
    return TreeSitterParser(language_name="c")


@pytest.fixture(scope="module")
def mock_tokenizer():
    tokenizer = MagicMock()
    def simple_tcount(text, add_special_tokens=False):
        return text.split()

    tokenizer.encode.side_effect = simple_tcount

    return tokenizer

# --- Tests for Helper Functions ---

# <---- test cases ---->
test_signature = [
    pytest.param(C_CODE_SIMPLE, "void simple_function(int x)", id="simple_code"),
    pytest.param(C_CODE_NESTED, "void nested_loop_function(int count)", id="nested_code"),
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)"),
]
test_structured_text_data = [
    pytest.param(C_CODE_SIMPLE, "if_statement", '    if (a > 10) {\n        printf("Greater");\n    }', id="simple_code"),
    pytest.param(C_CODE_NESTED, "for_statement", "    for (int i = 0; i < count; i++) {\n        if (i % 2 == 0) {\n            while (1) {\n                break;\n            }\n        }\n    }", id="nested_code"),
    pytest.param(C_CODE_PREPROCESSOR, "preproc_ifdef", "    #ifdef DEBUG\n        printf(\"Mode is: %d\\n\", mode);\n    #else\n        printf(\"Mode is: normal\\n\");\n    #endif", id="preprocessor_code"),
]

test_parent_func = [
    pytest.param(C_CODE_SIMPLE, "declaration", "void simple_function(int x)", id="simple_code"),
    pytest.param(C_CODE_PREPROCESSOR, "call_expression", "void preprocessor_function(int mode)", id="nested_code"),
    pytest.param(C_CODE_NESTED, "break_statement", "void nested_loop_function(int count)", id="nested_code")
]
test_func_statements = [
    pytest.param(C_CODE_SIMPLE, ["declaration", "if_statement"], id="simple_code"),
    pytest.param(C_CODE_NESTED, ["for_statement", "return_statement"], id="nested_code"),
    pytest.param(C_CODE_PREPROCESSOR, ["preproc_def", "declaration", "preproc_ifdef", "if_statement"]),
]

class TestHelperFunctions:
    # <---- tests ---->
    @pytest.mark.parametrize("func, expected_signature", test_signature)
    def test_get_function_signature(self, func:str, expected_signature:str, c_parser:TreeSitterParser):
        tree = c_parser.parse(func)
        func_node = tree.root_node.children[0]
        assert func_node.type == "function_definition"

        func_signature = _get_function_signature(func_def_node=func_node, source_code=func)
        assert func_signature == expected_signature

    @pytest.mark.parametrize("func, target_node_type, target_node_text", test_structured_text_data)
    def test_get_node_structured_text(self, func:str, target_node_type:str, target_node_text:str, c_parser: TreeSitterParser):
        tree = c_parser.parse(func)
        target_node = next(n for n in c_parser.traverse_tree(tree.root_node) if n.type == target_node_type)
        text = get_node_structured_text(node=target_node, code=func.encode(encoding="utf-8"))

        assert text == target_node_text

    @pytest.mark.parametrize("func, start_node_type, signature", test_parent_func)
    def test_find_parent_function(self, func:str, start_node_type: str, signature:str, c_parser: TreeSitterParser):
        tree = c_parser.parse(func)
        start_node = next(n for n in c_parser.traverse_tree(tree.root_node) if n.type == start_node_type)
        assert start_node.type is not None

        parent_func = _find_parent_function(start_node=start_node)
        assert parent_func is not None
        assert parent_func.type == 'function_definition'

        func_signature:str = _get_function_signature(func_def_node=parent_func, source_code=func)
        assert func_signature == signature

    @pytest.mark.parametrize("func, statements_list", test_func_statements)
    def test_get_func_statements(self, func:str, statements_list:list[str], c_parser: TreeSitterParser):
        tree = c_parser.parse(func)
        func_node = tree.root_node.children[0]
        body_node = func_node.child_by_field_name("body")
        statements = get_func_statements(body_node=body_node, tsp=c_parser)

        assert len(statements) == len(statements_list)
        for idx, stmnt_type in enumerate(statements_list):
            assert statements[idx].type == stmnt_type


# --- Tests for Splitting Strategies ---
class TestSplittingStrategies:
    def test_split_by_lines(self, mock_tokenizer):
        units = [
            {"text": "line one;", "tokens": 2},
            {"text": "line two;", "tokens": 2},
            {"text": "line three;", "tokens": 2},
            {"text": "line four;", "tokens": 2},
            {"text": "line five;", "tokens": 2},
        ]

        chunks = split_by_lines(units, max_tokens=6, tokenizer=mock_tokenizer, overlap_stmts=1)

        assert len(chunks) == 2
        # Chunk 1: No context cost, so it takes 3 lines (6 tokens).
        assert chunks[0]["text"] == "line one;\nline two;\nline three;"
        # Chunk 2: Context is "line three;" (2 tokens), leaving 4 tokens for text.
        # It takes lines 4 & 5 (4 tokens).
        assert chunks[1]["text"] == "line three;\nline four;\nline five;"

    def test_split_by_ast(self, c_parser: TreeSitterParser, mock_tokenizer):
        tree = c_parser.parse(C_CODE_SIMPLE)
        code_bytes = C_CODE_SIMPLE.encode(encoding="utf-8")
        func_node = tree.root_node.children[0]
        body_node = func_node.child_by_field_name("body")
        nodes = get_func_statements(body_node=body_node, tsp=c_parser)

        chunks = _split_by_ast(
            text_nodes=nodes, tokenizer=mock_tokenizer,
            available_tokens=10, original_full_tree=tree,
            original_full_code_bytes=code_bytes
        )

        assert len(chunks) == 2
        assert "int a = x + 1;" in chunks[0]["text"]
        assert "if (a > 10)" not in chunks[0]["text"] # budget too small to fit `if_statement` in trailing_context

        assert "int a = x + 1;" not in chunks[1]["text"] # No inner context was passed
        assert "if (a > 10)" in chunks[1]["text"]
        assert "printf(\"Greater\");" in chunks[1]["text"]


# # --- Tests for the Main Orchestrator ---
test_orchestrator_mode_data = [
    pytest.param(C_CODE_SIMPLE, "void simple_function(int x)", "0", 10, "line", id="simple_code_line"),
    pytest.param(C_CODE_NESTED, "void nested_loop_function(int count)", "1", 10, "line", id="nested_code_line"),
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)", "0", 10, "line", id="preprocessor_code_line"),
    pytest.param(C_CODE_SIMPLE, "void simple_function(int x)", "1", 100, "ast", id="simple_code_ast"),
    pytest.param(C_CODE_NESTED, "void nested_loop_function(int count)", "1", 100, "ast", id="nested_code_ast"),
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)", "1", 100, "ast", id="preprocessor_code_ast"),
]
test_orchestrator_with_preprocessor = [
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)", "0", 15, "ast", id="prepsocessor_ast_small_budget"),
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)", "1", 100, "ast", id="prepsocessor_ast"),
    pytest.param(C_CODE_PREPROCESSOR, "void preprocessor_function(int mode)", "0", 10, "line", id="prepsocessor_line"),
]
class TestGenerateCodeChunks:
    @pytest.mark.parametrize("func, func_signature, label, token_budget, mode", test_orchestrator_mode_data)
    def test_orchestrator_mode_structure(
        self, func: str, func_signature: str,
        label: str, token_budget: int, mode: str,
        c_parser: TreeSitterParser, mock_tokenizer,
    ):
        chunks = generate_code_chunks(
            code=func, tsp=c_parser,
            label=label, tokenizer=mock_tokenizer,
            max_tokens=token_budget, trimming_technique=mode
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, dict)
            assert chunk["label"] == label
            assert chunk["function_signature"] == func_signature
            assert "text" in chunk
            assert "tcount" in chunk

    @pytest.mark.parametrize("func", [C_CODE_SIMPLE, C_CODE_NESTED, C_CODE_PREPROCESSOR])
    def test_orchestrator_line_content(self, func: str, c_parser: TreeSitterParser, mock_tokenizer):
        """Tests the content of chunks from the line-based splitter."""
        chunks = generate_code_chunks(
            code=func, tsp=c_parser, label="0", tokenizer=mock_tokenizer,
            max_tokens=15, trimming_technique="line"
        )
        assert len(chunks) > 0
        body_lines = func.split('{', 1)[1].split('}', 1)[0].strip().split('\n')
        assert body_lines[0].strip() in chunks[0]["text"]

    @pytest.mark.parametrize("func", [C_CODE_PREPROCESSOR])
    def test_orchestrator_ast_content(self, func: str, c_parser: TreeSitterParser, mock_tokenizer):
        """Tests the content of chunks from the AST-based splitter."""
        chunks = generate_code_chunks(
            code=func, tsp=c_parser, label="1", tokenizer=mock_tokenizer,
            max_tokens=20, trimming_technique="ast"
        )
        assert len(chunks) > 0
        tree = c_parser.parse(func)
        body_node = tree.root_node.children[0].child_by_field_name("body")
        assert body_node is not None
        first_statement_text = get_node_structured_text(body_node.children[1], func.encode('utf-8'))
        assert first_statement_text.strip() in chunks[0]["text"]

    def test_orchestrator_no_function_error(self, c_parser, mock_tokenizer):
        """AST mode should fail if no function definition is found."""
        with pytest.raises(ValueError, match="AST-based chunking requires a function definition."):
            generate_code_chunks(
                code=C_CODE_NO_FUNC,
                tsp=c_parser,
                label="0",
                tokenizer=mock_tokenizer,
                max_tokens=100,
                trimming_technique="ast"
            )

    def test_orchestrator_no_function_fallback(self, c_parser, mock_tokenizer):
        """Line mode should still work on code without a function definition."""
        chunks = generate_code_chunks(
            code=C_CODE_NO_FUNC,
            tsp=c_parser,
            label="0",
            tokenizer=mock_tokenizer,
            max_tokens=100,
            trimming_technique="line"
        )
        assert len(chunks) == 1
        assert chunks[0]["text"].strip() == "int global_var = 100;"
        assert chunks[0]["function_signature"] == ""


# --- Integration Test ---

class TestIntegration:
    @pytest.mark.parametrize("trimming_technique", ["ast", "line"])
    def test_end_to_end_pipeline(self, trimming_technique: str, c_parser: TreeSitterParser):

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
}
"""
        try:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except (OSError, ValueError):
            pytest.skip("Could not download tokenizer, skipping integration test.")

        chunks = generate_code_chunks(
            code=long_code,
            tsp=c_parser,
            label="0",
            tokenizer=tokenizer,
            max_tokens=100,
            trimming_technique=trimming_technique,
        )

        # 1. Basic sanity checks
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["function_signature"] == "int calculate_stats(int *array, int n_elements, int filter_threshold)"
            assert chunk["label"] == "0"
            assert len(chunk["text"]) > 0
            assert chunk["tcount"] > 0

        # 2. Content integrity check: Ensure all key parts of the original function are present
        full_chunked_text = " ".join(c["text"] for c in chunks)

        # Check for key statements from the beginning, middle, and end
        assert "int sum_positive = 0;" in full_chunked_text
        assert "sum_negative += current_val;" in full_chunked_text
        assert "return sum_positive + sum_negative;" in full_chunked_text
        assert "#if 1>0" in full_chunked_text
        save_chunks_to_jsonl(chunks, f"test_{trimming_technique}.jsonl")


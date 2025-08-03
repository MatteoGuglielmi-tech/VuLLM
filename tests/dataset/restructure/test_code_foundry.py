import pytest
import os
from tree_sitter import Query, QueryCursor

from dataset.restructure.proc_utils import write2file, spawn_clang_format
from dataset.restructure.tree_sitter_parser import C_LANGUAGE
from dataset.restructure.code_foundry import CodeFoundry

# =================================================================================
@pytest.fixture(scope="module")
def foundry_instance():
    yield CodeFoundry(ts_lang=C_LANGUAGE)


project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
clang_format_file_path: str = os.path.join(project_root, ".clang-format")

def _get_refactored_code(code: str, lang_name: str, fp: str):
    write2file(fp=fp, content=code)
    formatted_result = spawn_clang_format(fp, lang_name, clang_format_file_path)
    os.remove(fp)

    return formatted_result
# =================================================================================

dangling_directive_test_cases = [
    (
        "dangling_endif",
        """
        void func() {
          #if A
            int x = 1;
          #endif
          #endif // Dangling closer
        }
        """,
        """
        void func() {
          #if A
            int x = 1;
          #endif
        }
        """,
    ),
    (
        "dangling_else",
        """
        void func() {
          #else // Dangling intermediate
            int y = 2;
          #endif
        }
        """,
        """
        void func() {
        {
        int y = 2;
        }
        }
        """,
    ),
    (
        "dangling_elif",
        """
        void func() {
          #elif B // Dangling intermediate
            int z = 3;
          #endif
        }
        """,
        """
        void func() {
        {int z = 3;}
        }
        """,
    ),
    (
        "no_dangling_directives",
        """
        void func() {
          #if A
            #if B
              int x = 1;
            #else
              int y = 2;
            #endif
          #endif
        }
        """,
        """
        void func() {
          #if A
            #if B
              int x = 1;
            #else
              int y = 2;
            #endif
          #endif
        }
        """,
    ),
    (
        "mixed_dangling",
        """
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
        """,
        """
        void func() {
          #if A
            int x = 1;
          #else
            int y = 2;
          #endif
          int z = 3;
        }
        """,
    ),
    (
        "unbalanced_inner_block",
        """
        void func() {
            #if A
                #if B
                    int x = 1;
                #endif
                #endif
            #else
                int y = 2;
                #endif
            #endif
        }
        """,
        """
        void func() {
            #if A
                #if B
                    int x = 1;
                #endif
            #endif
            {
              int y = 2;
            }
        }
        """,
    ),
    (
        "simple_dangling_endif",
        """
        void log_event() {
            #if DEBUG
                printf("Event triggered\\n");
            #endif
            #endif
        }
        """,
        """
        void log_event() {
            #if DEBUG
                printf("Event triggered\\n");
            #endif
        }
        """,
    ),
    (
        "simple_dangling_else",
        """
        void setup_config() {
            #else
                config.port = 8080;
                config.retries = 3;
            #endif
        }
        """,
        """
        void setup_config() {
            {
                config.port = 8080;
                config.retries = 3;
            }
        }
        """,
    ),
    (
        "dangling_elif_with_content",
        """
        int get_value() {
            #elif defined(FEATURE_B) // Dangling elif.
                return 200;
            #endif
            return 0;
        }
        """,
        """
        int get_value() {
            {
                return 200;
            }
            return 0;
        }
        """,
    ),
    (
        "dangling_else_with_nested_valid_if",
        """
        // A common real-world case: a dangling block contains valid logic.
        void process_packet(Packet* p) {
            #else // This whole block is dangling.
                #if defined(VALIDATE_PACKETS)
                    if (!is_valid(p)) return;
                #endif
                dispatch(p);
            #endif
        }
        """,
        """
        // A common real-world case: a dangling block contains valid logic.
        void process_packet(Packet* p) {
            {
                #if defined(VALIDATE_PACKETS)
                    if (!is_valid(p)) return;
                #endif
                dispatch(p);
            }
        }
        """,
    ),
    (
        "multiple_dangling_directives",
        """
        #endif // Dangling 1
        void calculate() {
            #if A
                int x = 1;
            #endif
            #elif B // Dangling 2
                int y = 2;
            #endif
            #endif // Dangling 3
        }
        """,
        """
        void calculate() {
            #if A
                int x = 1;
            #endif
            {
                int y = 2;
            }
        }
        """,
    ),
    (
        "no_dangling_directives_should_not_change",
        """
        int complex_check() {
            #if defined(MODE_A)
                return 1;
            #elif defined(MODE_B)
                return 2;
            #else
                return 3;
            #endif
        }
        """,
        """
        int complex_check() {
            #if defined(MODE_A)
                return 1;
            #elif defined(MODE_B)
                return 2;
            #else
                return 3;
            #endif
        }
        """,
    )
]
@pytest.mark.parametrize(
    "test_id, input_code, expected_code",
    dangling_directive_test_cases,
    ids=[t[0] for t in dangling_directive_test_cases],
)
def test_fix_dangling_directives(foundry_instance: CodeFoundry, test_id: str, input_code: str, expected_code: str):
    result_code, new_block_ranges = foundry_instance._fix_dangling_directives(input_code)

    # format both the actual result and the expected output ✨
    formatted_result = _get_refactored_code(result_code, "c", "./tmp_in.c")
    formatted_output = _get_refactored_code(expected_code, "c", "./tmp_out.c")

    assert formatted_result.strip() == formatted_output.strip()

    local_tree = foundry_instance.ts_parser.parse(bytes(result_code, encoding="utf-8"))
    for start_byte, end_byte in new_block_ranges:
        found_node = local_tree.root_node.descendant_for_byte_range(start_byte, end_byte - 1)
        assert found_node is not None, f"No node found at range {start_byte}-{end_byte}"
        assert (
            found_node.type == "compound_statement"
        ), f"Expected a 'compound_statement' at the reported range, but found '{found_node.type}'"

    assert formatted_result.strip() == formatted_output.strip()
# =================================================================================

build_symbol_table_test_cases = [
    (
        "simple_test_case",
        """
        int g_var = 1;
        void func() {
            char f_var = 'a';
            if (true) {
                int b_var = 2;
            }
        }"""
    ),
]

@pytest.mark.parametrize(
    "test_id, input_code",
    build_symbol_table_test_cases,
    ids=[t[0] for t in build_symbol_table_test_cases],
)
def test_build_symbol_table(foundry_instance: CodeFoundry, test_id: str, input_code: str):
    tree = foundry_instance.ts_parser.parse(bytes(input_code,encoding="utf-8"))
    symbol_manager = foundry_instance._build_symbol_table(tree=tree)

    # Assertions for the scope tree and symbols
    root_scope = symbol_manager.root_scope
    assert "g_var" in root_scope.symbols
    assert len(root_scope.children) == 1

    func_scope = root_scope.children[0]
    assert "f_var" in func_scope.symbols
    assert len(func_scope.children) == 1

    if_scope = func_scope.children[0]
    assert "b_var" in if_scope.symbols


# =================================================================================
build_symbol_table_test_cases = [
    (
        "complex_test_case",
        """
        // Global scope
        int g_count = 0;
        const char* g_name = "global";

        void process_data(int p_id) { // Function Scope
            char p_status = 'A';

            for (int i = 0; i < 10; i++) { // For-Loop Scope
                if (p_status == 'A') { // If-Statement Scope
                    // This 'p_status' is a shadow variable
                    char p_status = 'B';
                    int err_code = i;
                }
            }
        }"""
    ),
]

@pytest.mark.parametrize(
    "test_id, input_code",
    build_symbol_table_test_cases,
    ids=[t[0] for t in build_symbol_table_test_cases],
)
def test_build_symbol_table_complex_nested(foundry_instance: CodeFoundry, test_id: str, input_code: str):
    tree = foundry_instance.ts_parser.parse(bytes(input_code, encoding="utf-8"))
    symbol_manager = foundry_instance._build_symbol_table(tree=tree)

    # 1. Check the Global Scope (Root)
    root_scope = symbol_manager.root_scope
    assert "g_count" in root_scope.symbols
    assert "g_name" in root_scope.symbols
    assert "p_id" not in root_scope.symbols # Should not be in global scope
    assert len(root_scope.children) == 1, "Global scope should contain the function's scope"

    # 2. Check the Function Scope
    func_scope = root_scope.children[0]
    assert "p_id" in func_scope.symbols # Function parameter
    assert "p_status" in func_scope.symbols
    assert func_scope.symbols["p_status"].node.start_point[0] == 6
    assert len(func_scope.children) == 1, "Function scope should contain the for-loop's scope"

    # 3. Check the For-Loop Scope
    # Note: tree-sitter often places the for-loop's body as the child scope
    for_scope = func_scope.children[0]
    assert "i" in for_scope.symbols # Loop variable
    assert len(for_scope.children) == 1, "For-loop scope should contain the if-statement's scope"

    # 4. Check the If-Statement Scope (Innermost)
    if_scope = for_scope.children[0]
    assert "err_code" in if_scope.symbols
    # Check for the shadowed variable
    assert "p_status" in if_scope.symbols 
    assert if_scope.symbols["p_status"].node.start_point[0] == 11

    # 5. Verify Parent-Child Links
    assert if_scope.parent == for_scope
    assert for_scope.parent == func_scope
    assert func_scope.parent == root_scope
# =================================================================================

refine_scopes_test_cases = [
    (
        "basic",
        """
        int main() {
            int status = 0;
            {
                int status = -1;
                log_error(status);
            }
            {
                int error_code = 1;
                report(error_code);
            }
            return status;
        }
        """,
        """
        int main() {
            int status = 0;
            {
                int status = -1;
                log_error(status);
            }
            int error_code = 1;
            report(error_code);
            return status;
        }
        """
    ),
    (
        "for_loop_scoping",
        """
        void loop_test() {
            for (int i = 0; i < 10; i++) {
                {
                    int i = 99;
                    printf("Shadowed i: %d\\n", i);
                }
                {
                    int temp = i;
                    process(temp);
                }
            }
        }
        """,
        """
        void loop_test() {
            for (int i = 0; i < 10; i++) {
                {
                    int i = 99;
                    printf("Shadowed i: %d\\n", i);
                }
                int temp = i;
                process(temp);
            }
        }
        """
    ),
]
@pytest.mark.parametrize(
    "test_id, input_code, expected_code",
    refine_scopes_test_cases,
    ids=[t[0] for t in refine_scopes_test_cases],
)
def test_refine_scopes(foundry_instance, test_id, input_code, expected_code):
    query = Query(C_LANGUAGE, "(compound_statement) @block_to_check")

    tree = foundry_instance.ts_parser.parse(bytes(input_code, encoding="utf-8"))
    compound_statements = QueryCursor(query).captures(tree.root_node).get("block_to_check", [])
    block_ranges = [(n.start_byte, n.end_byte) for n in compound_statements]

    symbol_manager = foundry_instance._build_symbol_table(tree=tree)
    result_code = foundry_instance._refine_scopes(input_code, tree, block_ranges, symbol_manager)

    result_code = _get_refactored_code(result_code, "c", f"./test_{test_id}_candidate.c")
    expected_code = _get_refactored_code(expected_code, "c", f"./tmp_{test_id}_expected.c")

    result_code = os.linesep.join([line for line in result_code.splitlines() if line and line.strip() != ""])
    assert result_code.strip() == expected_code.strip()

# =================================================================================

full_pipeline_test_cases = [
    # (
    #     "deeply_nested_conflict",
    #     """
    #     void nested_test() {
    #         int x = 1;
    #         {
    #             int y = 2;
    #             if (y > 0) {
    #                 {
    #                     int x = 3; 
    #                     printf("Inner x: %d\\n", x);
    #                 }
    #             }
    #         }
    #     }
    #     """,
    #     """
    #     void nested_test() {
    #         int x = 1;
    #         int y = 2;
    #         if (y > 0) {
    #                 int x = 3;
    #                 printf("Inner x: %d\\n", x);
    #         }
    #     }
    #     """
    # ),
    # (
    #     "full_pipe",
    #     """
    #     int main() {
    #         int status = 0;
    #         #else
    #             int status = -1; 
    #             log_error(status);
    #         #endif
    #
    #         #elif c
    #             int error_code = 1;
    #             report(error_code);
    #         #endif
    #         return status;
    #     }
    #     """,
    #     """
    #     int main() {
    #         int status = 0;
    #         {
    #             int status = -1;
    #             log_error(status);
    #         }
    #         int error_code = 1;
    #         report(error_code);
    #         return status;
    #     }
    #     """
    # ), 
    # (
    #     "triky",
    #     """
    #     int main() {
    #         int status = 0;
    #         #else
    #             if(status ==1){
    #                 int status = -1; 
    #                 log_error(status);
    #             }
    #         #endif
    #
    #         #elif c
    #             int error_code = 1;
    #             report(error_code);
    #         #endif
    #         return status;
    #     }
    #     """,
    #     """
    #     int main() {
    #         int status = 0;
    #         if(status == 1) {
    #             int status = -1;
    #             log_error(status);
    #         }
    #         int error_code = 1;
    #         report(error_code);
    #         return status;
    #     }
    #     """
    # ),
    (
            "faling",
            """
            int process_data(int *data, int size) {
                int total = 0;
                for (int i = 0; i < size; ++i) {
                    if (data[i] > 0) {
                    {
                        total += data[i];
                    }
                    }
                }
                return total;
            }
            """,
            """
            int process_data(int *data, int size) {
                int total = 0;
                for (int i = 0; i < size; ++i) {
                    if (data[i] > 0) {
                        total += data[i];
                    }
                }
                return total;
            }
            """
        )
]
@pytest.mark.parametrize(
    "test_id, input_code, expected_code",
    full_pipeline_test_cases,
    ids=[t[0] for t in full_pipeline_test_cases],
)
def test_run_multi_pass_fix_integration(foundry_instance, test_id, input_code, expected_code):
    result_code = foundry_instance.run_multi_pass_fix(input_code)
    result_code = _get_refactored_code(result_code, "c", f"./test_{test_id}_candidate.c")
    expected_code = _get_refactored_code(expected_code, "c", f"./tmp_{test_id}_expected.c")


    assert result_code.strip() == expected_code.strip()


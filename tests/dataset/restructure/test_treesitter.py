import pytest

from dataset.restructure.shared.tree_sitter_parser import TreeSitterParser

probe_codes = [
    (
        """void func(int a) {
      if (a > 0) {}
      return;
    }
    """,
        [
            b"translation_unit",
            b"function_definition",
            b"primitive_type",
            b"function_declarator",
            b"identifier",
            b"parameter_list",
            b"(",
            b"parameter_declaration",
            b"primitive_type",
            b"identifier",
            b")",
            b"compound_statement",
            b"{",
            b"if_statement",
            b"if",
            b"parenthesized_expression",
            b"(",
            b"binary_expression",
            b"identifier",
            b">",
            b"number_literal",
            b")",
            b"compound_statement",
            b"{",
            b"}",
            b"return_statement",
            b"return",
            b";",
            b"}",
        ],
    ),
    (
        """int complex_func(int x) {
      do {
        switch(x) {
    #if defined(A)
          case 1:
            break;
    #endif
        }
      } while (x--);
    }
    """,
        [
            b"translation_unit",
            b"function_definition",
            b"primitive_type",
            b"function_declarator",
            b"identifier",
            b"parameter_list",
            b"(",
            b"parameter_declaration",
            b"primitive_type",
            b"identifier",
            b")",
            b"compound_statement",
            b"{",
            b"do_statement",
            b"do",
            b"compound_statement",
            b"{",
            b"switch_statement",
            b"switch",
            b"parenthesized_expression",
            b"(",
            b"identifier",
            b")",
            b"compound_statement",
            b"{",
            b"preproc_if",
            b"#if",
            b"preproc_defined",
            b"defined",
            b"(",
            b"identifier",
            b")",
            b"\n",
            b"case_statement",
            b"case",
            b"number_literal",
            b":",
            b"break_statement",
            b"break",
            b";",
            b"#endif",
            b"}",
            b"}",
            b"while",
            b"parenthesized_expression",
            b"(",
            b"update_expression",
            b"identifier",
            b"--",
            b")",
            b";",
            b"}",
        ],
    ),
]


@pytest.mark.parametrize("func, nodes", probe_codes)
def test_traverse_tree_correctness(func: str, nodes: list[bytes]):
    """Validates that traverse_tree visits all nodes in the correct pre-order sequence."""

    c_parser = TreeSitterParser(language_name="c")
    tree = c_parser.parse(func)
    actual_node_types = [node.type.encode() for node in c_parser.traverse_tree(tree)]

    assert (actual_node_types == nodes), f"Traversal failed. Expected {len(nodes)} nodes but got {len(actual_node_types)}."

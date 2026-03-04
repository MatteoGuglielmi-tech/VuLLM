use tree_sitter::{Language, Parser};

/// Asserts that two C code snippets are structurally equivalent by comparing their syntax trees.
///
/// This function parses both the `actual` and `expected` code strings and compares
/// their S-expression representations. This makes the comparison insensitive to
/// differences in whitespace, indentation, and other formatting details.
///
/// # Panics
///
/// Panics if the `actual` and `expected` syntax trees are not identical, or if
/// either snippet fails to parse.
pub fn assert_code_eq(lang: Language, actual: &str, expected: &str, message: &str, name: &str) {
    let mut parser = Parser::new();
    parser
        .set_language(&lang)
        .expect("Failed to set language for AST comparison");

    let tree_actual = parser
        .parse(actual, None)
        .expect("Failed to parse 'actual' code for comparison");

    let tree_expected = parser
        .parse(expected, None)
        .expect("Failed to parse 'expected' code for comparison");

    let sexp_actual = tree_actual.root_node().to_sexp();
    let sexp_expected = tree_expected.root_node().to_sexp();

    // The core assertion: compare the string representations of the trees.
    assert_eq!(sexp_actual, sexp_expected, "{} {}", message, name);
}

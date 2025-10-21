#[cfg(test)]
mod tests {
    use data_processing::processor_lib::utils::{
        compute_cyclomatic_complexity, compute_token_count,
    };
    use rstest::{fixture, rstest};
    use tiktoken_rs::CoreBPE;
    use tiktoken_rs::cl100k_base;
    use tree_sitter::{Language, Parser};

    // A fixture to provide the tokenizer for token counting tests.
    #[fixture]
    fn tokenizer() -> CoreBPE {
        cl100k_base().unwrap()
    }

    #[fixture]
    fn language() -> Language {
        tree_sitter_c::LANGUAGE.into()
    }

    #[fixture]
    fn parser() -> Parser {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_c::LANGUAGE.into())
            .unwrap();

        parser
    }

    // --- CYCLOMATIC COMPLEXITY ---
    #[rstest]
    #[case(
        r#"int main() {
            return 0;
        }"#,
        1
    )]
    #[case(
        r#"void t(int a) {
            if (a > 0) return;
        }"#,
        2
    )]
    #[case(
        r#"void t(int a) {
            if (a > 0 && a < 10) {
                for(int i=0;i<a;i++){}
            }
        }"#,
        4
    )]
    #[case(
        r#"void t(int a) {
            while(a > 0 || a < -1) {
                a = a > 0 ? a-1:a+1;
            }
        }"#,
        4
    )]
    fn test_cyclomatic_complexity(mut parser: Parser, #[case] code: &str, #[case] expected: usize) {
        let tree = parser.parse(code, None).unwrap();
        println!("{:?}", tree.root_node().to_sexp());
        let complexity = compute_cyclomatic_complexity(code, &tree);
        assert_eq!(complexity, expected as u32);
    }

    // --- Token count tests ---
    #[rstest]
    #[case("hello world", 2)]
    #[case("int main() { return 0; }", 9)]
    #[case("", 0)]
    fn test_token_count(tokenizer: CoreBPE, #[case] code: &str, #[case] expected: usize) {
        let count = compute_token_count(code, &tokenizer);
        println!("Token count: {}", count);
        assert_eq!(count, expected);
    }
}

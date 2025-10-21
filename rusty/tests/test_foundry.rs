mod common;
mod test_helpers;

#[cfg(test)]
mod tests {
    use super::common;
    use super::test_helpers;
    use data_processing::processor_lib::{
        code_foundry::CodeFoundry, tree_sitter_parser::TreeSitterParser,
    };
    use rstest::{fixture, rstest};

    #[fixture]
    fn setup_foundry() -> CodeFoundry {
        CodeFoundry
    }

    #[fixture]
    fn setup_ts() -> TreeSitterParser {
        TreeSitterParser::new(tree_sitter_ext_c::language())
    }

    #[rstest]
    #[case(
        "lone dangling endif",
        r#"
    void main() {
        int x = 5;
    #endif
    }"#,
        r#"
    void main() {
        int x = 5;
    }"#
    )]
    #[case(
        "dangling endif",
        r#"
    void func() {
      #if A
        int x = 1;
      #endif
      #endif // Dangling closer
    }"#,
        r#"
    void func() {
      #if A
        int x = 1;
      #endif
    }"#
    )]
    #[case(
        "dangling_else",
        r#"
    void func() {
      #elif B
      int x = 1;
    }"#,
        r#"
    void func() {
      int x = 1;
    }"#
    )]
    #[case(
        "dangling_else",
        r#"
    void func() {
      #else
        int y = 2;
      #endif
    }
    "#,
        r#"
    void func() { }
    "#
    )]
    #[case(
        "no_dangling_directives_should_not_change",
        r#"int complex_check() {
        #if defined(MODE_A)
            return 1;
        #elif defined(MODE_B)
            return 2;
        #else
            return 3;
        #endif
    }"#,
        r#"int complex_check() {
        #if defined(MODE_A)
            return 1;
        #elif defined(MODE_B)
            return 2;
        #else
            return 3;
        #endif
    }"#
    )]
    #[case(
        "multiple_dangling_directives",
        r#"#endif
    void calculate() {
        #if A
            int x = 1;
        #endif
        #elif B
            int y = 2;
        #endif
        #endif
    }"#,
        r#"void calculate() {
        #if A
            int x = 1;
        #endif
    }"#
    )]
    #[case(
        "mixed_dangling",
        r##"#endif // Dangling
    void func() {
      #if A
        int x = 1;
      #else
        int y = 2;
      #endif
      #elif C // Dangling
        int z = 3;
    }"##,
        r#"void func() {
      #if A
        int x = 1;
      #else
        int y = 2;
      #endif
      int z = 3;
    }"#
    )]
    #[case(
        "unbalanced_inner_block",
        r#"void func() {
      #if A
        #if B
          int x = 1;
        #endif
      #endif
      #else
        int y = 2;
        #endif
      #endif
    }"#,
        r#"void func() {
      #if A
        #if B
            int x = 1;
        #endif
      #endif
    }"#
    )]
    #[case(
        "dangling_else_with_nested_valid_if",
        r#"void process_packet(Packet* p) {
      #else
        #if defined(VALIDATE_PACKETS)
          if (!is_valid(p)) return;
        #endif
        dispatch(p);
      #endif
    }"#,
        r#"void process_packet(Packet* p) {
       #if defined(VALIDATE_PACKETS)
         if (!is_valid(p)) return;
       #endif
       dispatch(p);
    }"#
    )]
    fn test_dangling_directive_fixes_with_rstest(
        setup_foundry: CodeFoundry,
        setup_ts: TreeSitterParser,
        #[case] name: &str,
        #[case] code: &str,
        #[case] expected: &str,
    ) {
        common::setup();
        let result = setup_foundry.remove_dangling_directives(code, &setup_ts);
        println!("Input code:\n{}", code);
        println!("Expected code:\n{}", expected);
        println!("Result:\n{}", result);
        test_helpers::assert_code_eq(
            tree_sitter_ext_c::language(),
            result.trim(),
            expected.trim(),
            "Test case failed: {}",
            name,
        );
    }
}

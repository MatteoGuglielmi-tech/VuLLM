mod test_helpers;

#[cfg(test)]
mod tests {
    use super::test_helpers;
    use data_processing::processor_lib::processor::Processor;
    use rstest::{fixture, rstest};
    use tree_sitter::Language;

    #[fixture]
    fn processor() -> Processor {
        Processor::new().expect("Test setup failed: Could not create Processor.")
    }

    #[fixture]
    fn language() -> Language {
        tree_sitter_ext_c::language()
    }

    #[rstest]
    #[case(
        "A simple, valid function",
        r#"int main(void) {
          return 0;
        }"#,
        "int main(void) { return 0; }"
    )]
    #[case(
        "Code with a dangling directive that should be removed",
        "#endif\n\nint foo() { return 1; }",
        "int foo() { return 1; }"
    )]
    #[case(
        "Code with a dangling directive that should be removed",
        "#ifdef __KERNEL__\nint foo() { return 1; }",
        ""
    )]
    #[case(
        "Code with leading garbage that should be removed",
        "**/ int foo() { return 1; }",
        "int foo() { return 1; }"
    )]
    #[case(
        "Code with leading garbage that should be removed",
        "}}}*/ int foo() { return 1; }",
        "int foo() { return 1; }"
    )]
    #[case(
        "Code with missing function body braces",
        "int simple_func() { return 42;",
        "int simple_func() { return 42; }"
    )]
    #[case(
        "A snippet that is not a function and should be rejected",
        "const int GLOBAL_VAR = 100;",
        ""
    )]
    #[case(
        "Code that is entirely removed by the C preprocessor",
        "#define EMPTY\nEMPTY",
        ""
    )]
    fn test_processing_pipeline(
        processor: Processor,
        language: Language,
        #[case] description: &str,
        #[case] input: &str,
        #[case] expected: &str,
    ) {
        let Ok(Some((sanitized_code, _))) = processor.process_snippet_fallible(input) else {
            return;
        };
        test_helpers::assert_code_eq(
            language,
            &sanitized_code,
            expected.trim(),
            "Test case failed: {}",
            description,
        );
    }
}

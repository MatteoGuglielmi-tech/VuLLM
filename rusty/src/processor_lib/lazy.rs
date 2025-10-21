use regex::{Regex, RegexSet};
use once_cell::sync::Lazy;
use tree_sitter::{Language, Query};

/* --- LANGUAGE LAZY --- */
pub static C_LANG: Lazy<Language> = Lazy::new(|| tree_sitter_c::LANGUAGE.into());
pub static CPP_LANG: Lazy<Language> = Lazy::new(|| tree_sitter_cpp::LANGUAGE.into());
// static EXT_C_LANG: Lazy<Language> = Lazy::new(tree_sitter_ext_c::language);

/* --- QUERY LAZY --- */
// used to extract and validate functions
pub static VALIDATION_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r#"
        (function_definition) @target
        (compound_statement) @target
    "#;

    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile `VALIDATION_QUERY`")
});

pub static DIRECTIVE_BALANCE_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r##"
        ; --- openers ---
        (preproc_if) @opener
        (preproc_ifdef) @opener

        ; --- intermediate ---
        (preproc_call
            directive: (preproc_directive) @_text
            (#any-of? @_text "#elif" "#else")
        ) @intermediate

        ; --- closers ---
        ; (token-based matching)
        (preproc_call
            directive: (preproc_directive) @_dir_content
            (#eq? @_dir_content "#endif")
        ) @closer
        "##;

    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile `DIRECTIVES_QUERY`")
});

pub static COMMENT_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r"(comment) @comment";
    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile `COMMENT_QUERY`")
});

pub static BRACE_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r#"
        "{" @open
        "}" @close
    "#;
    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile BRACE_QUERY")
});

pub static IS_CPP_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r#"[
        ; class
        (class_specifier)
        (access_specifier)
        (destructor_name)

        ; namespace
        (namespace_definition)
        (namespace_identifier) ; scope

        ; template
        (template_type)

        ; statement
        (field_initializer_list)
        (reference_declarator)
        (optional_parameter_declaration)
        (lambda_expression)
        (for_range_loop)
        (try_statement)
        (throw_statement)
        (catch_clause)
        (new_expression)
        (delete_expression)
        (this)

        ; C++11
        ; direct list initialization
        (init_declarator
          value: (initializer_list)
        )

    ] @cpp_feature"#;

    Query::new(&CPP_LANG, QUERY_STRING).expect("Failed to compile `IS_CPP_QUERY` query")
});

pub static DIRECTIVES_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r##"
        ; --- openers ---
        (preproc_if) @opener
        (preproc_ifdef) @opener

        ; --- intermediate ---
        (preproc_else)@intermediate
        (preproc_elif)@intermediate
        (preproc_call
            directive: (preproc_directive) @_text
            (#any-of? @_text "#elif" "#else")
        ) @intermediate

        ; --- closers ---
        (preproc_call
            directive: (preproc_directive) @_dir_content
            (#eq? @_dir_content "#endif")
        ) @closer
        "##;

    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile `DIRECTIVES_QUERY`")
});

pub static CYCLOMATIC_COMPLEXITY_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r#"
        (if_statement) @decision_point
        (case_statement) @decision_point
        (do_statement) @decision_point
        (while_statement) @decision_point
        (for_statement) @decision_point

        (conditional_expression) @decision_point
        (binary_expression operator: "&&" ) @decision_point
        (binary_expression operator: "||" ) @decision_point
        "#;

    Query::new(&C_LANG, QUERY_STRING)
        .expect("Failed to compile `CYCLOMATIC_COMPLEXITY_QUERY` query")
});

pub static IS_FUNCTION_QUERY: Lazy<Query> = Lazy::new(|| {
    const QUERY_STRING: &str = r#"(function_definition) @function"#;
    Query::new(&C_LANG, QUERY_STRING).expect("Failed to compile `IS_FUNCTION` query")
});

/* --- REGEX LAZY --- */

pub static LEADING_GARBAGE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\*+\s*\\*(/|\\)").unwrap());

// used to detect C++ functions
pub static CPP_PATTERNS: Lazy<RegexSet> = Lazy::new(|| {
    RegexSet::new([
        // OOP and access control
        r"\b(class)\s+\w+\s*(:)?.*\{",
        r"\b(public|protected|private)\s*:",
        // Templates and namespace
        r"\b(template)\s*<",
        r"\b(namespace)\b",
        r"\b\w+(?:<[^>]*>)?::\w+",
        // Exception handling and memory
        r"\b(try|catch|throw|new|delete)\b",
        // C++11 new features
        r"\b(static_cast|dynamic_cast|reinterpret_cast|const_cast)\s*<",
    ])
    .unwrap()
});

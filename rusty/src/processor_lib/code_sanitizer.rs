use anyhow::{Context, anyhow};
use rayon;
use std::borrow::Cow;
use std::io::{self, Write};
use std::process::{Command, Stdio};
use tree_sitter::{Node, QueryCursor};
use tree_sitter::{StreamingIterator, Tree};
use which::which;

use crate::processor_lib::lazy::{
    BRACE_QUERY, COMMENT_QUERY, DIRECTIVE_BALANCE_QUERY, IS_CPP_QUERY, VALIDATION_QUERY,
};
use crate::processor_lib::lazy::{CPP_PATTERNS, LEADING_GARBAGE_PATTERN};
use crate::processor_lib::tree_sitter_parser::TreeSitterParser;

#[derive(Debug)]
pub struct CodeSanitizer;

impl CodeSanitizer {
    /// Removes all non-ASCII characters from a string.
    pub fn remove_non_ascii(&self, code: &str) -> String {
        code.chars().filter(|c| c.is_ascii()).collect()
    }

    /// Helper function to remove frequent garbage patterns from the start of the code.
    fn cleanup_leading_garbage<'a>(&self, code: &'a str) -> Cow<'a, str> {
        LEADING_GARBAGE_PATTERN.replace(code, "")
    }

    pub fn remove_comments<'a>(&self, code: &'a str, ts_parser: &TreeSitterParser) -> Cow<'a, str> {
        let code_after_comments = ts_parser.with_tree(code, |_parser, tree| {
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&COMMENT_QUERY, tree.root_node(), code.as_bytes());

            // Collect the byte ranges (start, end) of the comments, not the Node objects.
            let mut comment_ranges: Vec<(usize, usize)> = Vec::new();

            while let Some(a_match) = matches.next() {
                let node = a_match.captures[0].node;
                comment_ranges.push((node.start_byte(), node.end_byte()));
            }

            if comment_ranges.is_empty() {
                return Cow::Borrowed(code);
            }

            // The rest of your logic now uses the collected ranges.
            let mut new_code = String::with_capacity(code.len());
            let mut last_end = 0;
            for (start_byte, end_byte) in comment_ranges {
                new_code.push_str(&code[last_end..start_byte]);
                last_end = end_byte;
            }
            new_code.push_str(&code[last_end..]);
            Cow::Owned(new_code)
        });

        let intermediate_cow = code_after_comments.unwrap_or(Cow::Borrowed(code));

        let cleaned_cow = self.cleanup_leading_garbage(&intermediate_cow);
        let final_str = cleaned_cow.trim_start();
        if final_str != code {
            Cow::Owned(final_str.to_string())
        } else {
            Cow::Borrowed(code)
        }
    }

    /// Validates that a code snippet contains a valid function-like construct
    /// and extracts the code from that point onward, discarding leading garbage.
    ///
    /// Returns a slice of the original code string if a valid construct is found, otherwise `None`.
    pub fn validate_and_extract_body<'a>(
        &self,
        code: &'a str,
        tree: &Tree,
    ) -> Option<Cow<'a, str>> {
        let mut cursor = QueryCursor::new();
        let mut matches_iterator =
            cursor.matches(&VALIDATION_QUERY, tree.root_node(), code.as_bytes());

        let mut first_node: Option<Node> = None;
        while let Some(query_match) = matches_iterator.next() {
            for capture in query_match.captures {
                let current_node = capture.node;

                match first_node {
                    Some(ref mut existing_node) => {
                        if current_node.start_byte() < existing_node.start_byte() {
                            *existing_node = current_node;
                        }
                    }
                    None => {
                        first_node = Some(current_node);
                    }
                }
            }
        }

        // After checking all matches, get the earliest node found.
        let earliest_node = first_node?;

        // Special handling for nested compound statements.
        if earliest_node.kind() == "compound_statement"
            && let Some(parent) = earliest_node.parent()
            && !["translation_unit", "ERROR"].contains(&parent.kind())
        {
            return None;
        }

        let start_byte = earliest_node.start_byte();
        if start_byte == 0 {
            Some(Cow::Borrowed(code))
        } else {
            Some(Cow::Owned(code[start_byte..].to_string()))
        }
    }

    pub fn balance_directives<'a>(
        &self,
        code: &'a str,
        ts_parser: &TreeSitterParser,
    ) -> Cow<'a, str> {
        let maybe_modified_code = ts_parser.with_tree(code, |_parser, tree| {
            let mut cursor = QueryCursor::new();
            let matches =
                cursor.matches(&DIRECTIVE_BALANCE_QUERY, tree.root_node(), code.as_bytes());
            let capture_names = DIRECTIVE_BALANCE_QUERY.capture_names();

            let balance = matches.fold(0, |acc, m| {
                for c in m.captures {
                    let name = capture_names[c.index as usize];
                    if name.starts_with('_') {
                        continue;
                    }
                    return match name {
                        "opener" => acc + 1,
                        "closer" => acc - 1,
                        _ => acc,
                    };
                }
                acc
            });

            if balance > 0 {
                let mut new_code = String::with_capacity(code.len() + (balance as usize * 7)); // "\n#endif" = 7 bytes
                new_code.push_str(code);
                for _ in 0..balance {
                    new_code.push_str("\n#endif");
                }
                Cow::Owned(new_code)
            } else {
                Cow::Borrowed(code)
            }
        });

        // If parsing failed (maybe_modified_code is None), return the original code.
        maybe_modified_code.unwrap_or(Cow::Borrowed(code))
    }

    pub fn add_missing_braces<'a>(&self, code: &'a str, tree: &Tree) -> Cow<'a, str> {
        let mut open_braces: usize = 0;
        let mut close_braces: usize = 0;
        let open_idx = BRACE_QUERY.capture_index_for_name("open").unwrap();
        let close_idx = BRACE_QUERY.capture_index_for_name("close").unwrap();

        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&BRACE_QUERY, tree.root_node(), code.as_bytes());

        while let Some(m) = matches.next() {
            for cap in m.captures {
                let node = cap.node;
                // filter out phantom, empty, or error nodes.
                if node.is_missing() || node.is_error() || node.start_byte() == node.end_byte() {
                    continue;
                }

                if cap.index == open_idx {
                    open_braces += 1;
                } else if cap.index == close_idx {
                    close_braces += 1;
                }
            }
        }

        let balance = open_braces.saturating_sub(close_braces);

        if balance > 0 {
            let trimmed_code = code.trim_end();
            let mut new_code = String::with_capacity(trimmed_code.len() + (balance * 2));
            new_code.push_str(trimmed_code);
            for _ in 0..balance {
                new_code.push_str("\n}");
            }
            Cow::Owned(new_code)
        } else {
            Cow::Borrowed(code)
        }
    }

    /// Finds and fixes C functions missing an explicit return type by prepending `int`.
    ///
    /// This method inspects the top-level nodes of the syntax tree for common patterns
    /// that tree-sitter produces when a return type is absent (e.g., an ERROR node
    /// followed by a compound statement).
    pub fn add_missing_return_types<'a>(&self, code: &'a str, tree: &Tree) -> Option<Cow<'a, str>> {
        let root_node = tree.root_node();
        let mut is_broken_pattern = false;

        let first_child = root_node.child(0)?;
        if first_child.kind() == "function_definition" {
            // happy path
            return Some(Cow::Borrowed(code));
        }

        // pattern 1: (ERROR) or (expression_statement) followed by a function body.
        if root_node.child_count() >= 2
            && let Some(second_child) = root_node.child(1)
            && ["ERROR", "expression_statement"].contains(&first_child.kind())
            && second_child.kind() == "compound_statement"
        {
            is_broken_pattern = true;
        }

        // Check for Pattern 2: The type is parsed as a macro.
        if !is_broken_pattern {
            if first_child.kind() == "macro_type_specifier" {
                is_broken_pattern = true;
            } else if first_child.kind() == "declaration"
                && let Some(named_child) = first_child.named_child(0)
                && named_child.kind() == "macro_type_specifier"
            {
                is_broken_pattern = true;
            }
        }

        if is_broken_pattern {
            let mut new_code = String::with_capacity(4 + code.len()); // "int " is 4 bytes
            new_code.push_str("int ");
            new_code.push_str(code);
            Some(Cow::Owned(new_code))
            // Some(Cow::Owned(format!("int {}", code)))
        } else {
            Some(Cow::Borrowed(code))
        }
    }

    // pub fn is_php_zend(&self, code: &str, tree: &Tree) -> bool {
    //     let mut cursor = QueryCursor::new();
    //     let mut matches = cursor.matches(&ZEND_QUERY, tree.root_node(), code.as_bytes());
    //     matches.next().is_some()
    // }

    pub fn call_gcc_preprocessor(
        &self,
        code: &str,
        flags: Option<&[&str]>,
    ) -> Result<String, String> {
        let mut command = Command::new("gcc");
        command
            .arg("-E") // run preprocessor
            .arg("-P") // no linemarkers
            .arg("-x") // force C
            .arg("c")
            .arg("-") // read stdin
            .stdin(Stdio::piped()) // don't expect keyboard input
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Add any extra flags if they were provided.
        if let Some(flags) = flags {
            command.args(flags);
        }

        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(
                    "❌ 'gcc' not found. Ensure it is installed and in the system's PATH."
                        .to_string(),
                );
            }
            Err(e) => {
                return Err(format!("❌ Failed to spawn gcc process: {}", e));
            }
        };

        // The `?` here converts a potential io::Error into our error String.
        child
            .stdin // 1. Access the stdin handle
            .take() // 2. Take ownership of the handle
            .unwrap() // 3. Unwrap the Option
            .write_all(code.as_bytes()) // 4. Write the data
            .map_err(|e| format!("❌ Failed to write to gcc stdin: {}", e))?;

        // Wait for the process to finish and capture its output.
        let output = child
            .wait_with_output()
            .map_err(|e| format!("❌ Failed to wait for gcc process: {}", e))?;

        if output.status.success() {
            // On success, convert stdout to a String.
            String::from_utf8(output.stdout)
                .map_err(|e| format!("❌ GCC output was not valid UTF-8: {}", e))
        } else {
            // On failure, convert stderr to a String to show the error.
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("❌ Error during preprocessing:\n{}", stderr))
        }
    }

    pub fn format_code(&self, code: &str) -> Result<String, anyhow::Error> {
        let clang_format_exe =
            which("clang-format").context("`clang-format` executable not found in PATH.")?;

        let mut child = Command::new(&clang_format_exe)
            .arg("-assume-filename=input.c") // avoids defaulting to C++ and conflict with
            // .clang-format
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context(format!(
                "Failed to spawn clang-format at '{}'",
                clang_format_exe.display()
            ))?;

        let mut stdin = child
            .stdin
            .take()
            .expect("Failed to open child process stdin");
        let code_to_write = code.to_string();

        // handle I/O concurrently within Rayon's thread pool.
        let (write_result, output_result) = rayon::join(
            // write to stdin.
            move || stdin.write_all(code_to_write.as_bytes()),
            // wait for the process to finish while consuming stdout and stderr
            || child.wait_with_output(),
        );

        write_result.context("Failed to write to clang-format stdin")?;
        let output = output_result.context("Failed to wait for clang-format to exit")?;

        if output.status.success() {
            Ok(String::from_utf8(output.stdout)?)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow!(
                "clang-format failed with exit code {}:\n{}",
                output.status,
                stderr
            ))
        }
    }

    pub fn is_cpp(&self, code: &str, cpp_parser: &TreeSitterParser) -> bool {
        if CPP_PATTERNS.is_match(code) {
            return true;
        }

        // Slower, more accurate tree-sitter check using the safe wrapper.
        let query_matches = cpp_parser.with_tree(code, |_parser, tree| {
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&IS_CPP_QUERY, tree.root_node(), code.as_bytes());
            matches.next().is_some()
        });

        query_matches.unwrap_or(false)
    }
}

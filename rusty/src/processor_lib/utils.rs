use crate::processor_lib::lazy::{CYCLOMATIC_COMPLEXITY_QUERY, IS_FUNCTION_QUERY};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use serde_json::Deserializer;
use std::borrow::Cow;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;
use tiktoken_rs::CoreBPE;
use tree_sitter::{Node, QueryCursor, QueryMatches, StreamingIterator, Tree};

use super::loader::Loader;


#[derive(Deserialize, Debug, PartialEq, Clone)]
pub struct JsonlEntry {
    pub func: String,
    pub target: u8,
    pub cwe: Vec<String>,
    pub project: String,
}

/// Reads a .jsonl file line by line, parsing each line into a serde_json::Value.
pub fn read_jsonl(input_file_path: &Path) -> Result<Vec<JsonlEntry>> {
    let desc_msg = format!("Reading data from '{}'", input_file_path.display());
    let _loader = Loader::new(&desc_msg, "Done.", Duration::from_millis(100));

    let file = fs::File::open(input_file_path)?;
    let reader = BufReader::new(file);
    let stream = Deserializer::from_reader(reader).into_iter::<JsonlEntry>();

    stream
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| anyhow::anyhow!("JSONL parse error: {}", e))
}

pub fn create_progress_bar(len: u64, template_prefix: &str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template(
            &format!("{}: {{percent}}% |{{bar:40.cyan/blue}}| {{pos}}/{{len}} [{{elapsed_precise}}] (ETA:{{eta}}, {{per_sec}})", template_prefix))
            .unwrap()
            // .progress_chars("#>-"),
            .progress_chars("■▢")
    );
    pb
}

/// Compute cyclomatic complexity for the input code.
pub fn compute_cyclomatic_complexity(code: &str, tree: &Tree) -> u32 {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(
        &CYCLOMATIC_COMPLEXITY_QUERY,
        tree.root_node(),
        code.as_bytes(),
    );

    let mut count = 0;
    while matches.next().is_some() {
        count += 1;
    }

    count + 1
}

/// Computes the number of tokens in a string using the cl100k_base tokenizer.
pub fn compute_token_count(code: &str, tokenizer: &CoreBPE) -> usize {
    tokenizer.encode_ordinary(code).len()
}

/// Address whether the piece of code is a valid function
pub fn is_function(code: &str, tree: &Tree) -> bool {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&IS_FUNCTION_QUERY, tree.root_node(), code.as_bytes());
    matches.next().is_some()
}

/// Convert elapsed time to `hh:mm:ss` format
pub fn format_duration(d: &Duration) -> String {
    let total_seconds = d.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

/// A helper struct to hold a directive's node and its classified type.
#[derive(Debug, Clone)]
pub struct Directive<'a> {
    pub node: Node<'a>,
    pub name: String,
}

/// Collects directives, prints detailed debug info, and returns the sorted list.
pub fn show_debug_info<'a>(
    code: &'a str,
    matches: &mut QueryMatches<'_, 'a, &'a [u8], &'a [u8]>,
    capture_names: &[&str],
) {
    let mut all_directives = Vec::new();

    while let Some(m) = matches.next() {
        for c in m.captures {
            let name = capture_names[c.index as usize];
            if name.starts_with('_') {
                continue; // Skip private captures
            }

            all_directives.push(Directive {
                node: c.node,
                name: name.to_string(),
            });
        }
    }
    all_directives.sort_by_key(|d| d.node.start_byte());

    // --- Start of Debug Printing ---
    println!(
        "--- Debugging Directives (Count: {}) ---",
        all_directives.len()
    );
    for (i, d) in all_directives.iter().enumerate() {
        let node_text_cow = d
            .node
            .utf8_text(code.as_bytes())
            .unwrap_or(&Cow::Borrowed("ERROR"));
        println!(
            "  [{}]: Capture='{}', Kind='{}', Text='{}', Node's byte Range='{:?}', Code text in byte range of node='{:?}'",
            i,
            d.name,
            d.node.kind(),
            node_text_cow.trim(),
            (d.node.start_byte(), d.node.end_byte()),
            &code[d.node.start_byte()..d.node.end_byte()]
        );
    }
    // --- End of Debug Printing ---
}

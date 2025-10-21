use super::tree_sitter_parser::TreeSitterParser;
use crate::processor_lib::lazy::DIRECTIVES_QUERY;
use std::borrow::Cow;
use std::collections::HashSet;
use tree_sitter::Tree;
use tree_sitter::{Node, QueryCursor, StreamingIterator};

pub struct CodeFoundry;

#[derive(Clone, Copy)]
struct Directive<'a> {
    node: Node<'a>,
    name: &'static str,
}

impl CodeFoundry {
    fn collect_unique_directives<'a>(&self, tree: &'a Tree, code: &'a str) -> Vec<Directive<'a>> {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&DIRECTIVES_QUERY, tree.root_node(), code.as_bytes());
        let capture_names = DIRECTIVES_QUERY.capture_names();
        let mut all_directives = Vec::new();

        while let Some(m) = matches.next() {
            for c in m.captures {
                let name = capture_names[c.index as usize];
                if name.starts_with('_') {
                    continue; // Skip private captures
                }

                all_directives.push(Directive {
                    node: c.node,
                    name: match name {
                        "opener" => "opener",
                        "intermediate" => "intermediate",
                        "closer" => "closer",
                        _ => unreachable!(),
                    },
                });
            }
        }
        all_directives.sort_by_key(|d| d.node.start_byte());

        let mut unique_directives = Vec::new();
        let mut processed_ranges: Vec<(usize, usize)> = Vec::new();

        for directive in all_directives {
            let range = (directive.node.start_byte(), directive.node.end_byte());

            let is_contained = processed_ranges
                .iter()
                .any(|(p_start, p_end)| range.0 >= *p_start && range.1 <= *p_end);

            if !is_contained {
                unique_directives.push(directive);
                processed_ranges.push(range);
            }
        }

        unique_directives
    }

    fn get_line_bounds(&self, node: Node, code_bytes: &[u8]) -> (usize, usize) {
        let node_start = node.start_byte();
        let mut end_byte = node.end_byte();

        // scan backwards from node_start to find the beginning of the line.
        let mut start_byte = node_start;
        while start_byte > 0 && code_bytes[start_byte - 1] != b'\n' {
            start_byte -= 1;
        }

        let kind = node.kind();
        if kind.starts_with("preproc_if") || kind == "preproc_elif" || kind == "preproc_else" {
            if let Some(slice) = code_bytes.get(start_byte..end_byte)
                && let Some(pos) = slice.iter().position(|&b| b == b'\n')
            {
                end_byte = start_byte + pos + 1;
            }
        } else {
            // For simple nodes (preproc_call), just find the trailing newline.
            if end_byte < code_bytes.len() && code_bytes[end_byte] == b'\r' {
                end_byte += 1;
            }
            if end_byte < code_bytes.len() && code_bytes[end_byte] == b'\n' {
                end_byte += 1;
            }
        }
        (start_byte, end_byte)
    }

    /// Uses a stack to find indices of dangling #else, #elif, and #endif directives.
    fn identify_mismatched_directives(
        &self,
        directives: &[Directive],
        code: &str,
    ) -> HashSet<usize> {
        let mut mismatched_indices = HashSet::new();
        let mut opener_stack: Vec<usize> = Vec::new();
        let mut seen_else_for_opener: HashSet<usize> = HashSet::new();

        for (i, directive) in directives.iter().enumerate() {
            match directive.name {
                "opener" => {
                    // Preserving original logic: only track `preproc_call` openers.
                    if directive.node.kind() == "preproc_call" {
                        opener_stack.push(i);
                    }
                }
                "intermediate" => {
                    if let Some(&opener_index) = opener_stack.last() {
                        // This #else/#elif has an opener, but is it valid?
                        // It's invalid if we've already seen an #else for this opener.
                        if seen_else_for_opener.contains(&opener_index) {
                            mismatched_indices.insert(i);
                        }
                        // Check if the current node is an #else directive.
                        if let Ok(text) = directive.node.utf8_text(code.as_bytes())
                            && text.trim().starts_with("#else")
                        {
                            seen_else_for_opener.insert(opener_index);
                        }
                    } else {
                        // Dangling: no opener for this #else/#elif.
                        mismatched_indices.insert(i);
                    }
                }
                "closer" => {
                    if let Some(opener_index) = opener_stack.pop() {
                        // This #endif correctly closes an opener.
                        seen_else_for_opener.remove(&opener_index);
                    } else {
                        // Dangling: no opener for this #endif.
                        mismatched_indices.insert(i);
                    }
                }
                _ => (),
            }
        }
        mismatched_indices
    }

    /// Calculates the byte ranges of code to remove based on mismatched indices.
    fn calculate_removal_ranges(
        &self,
        directives: &[Directive],
        mismatched_indices: &HashSet<usize>,
        code: &str,
    ) -> Vec<(usize, usize)> {
        let mut ranges_to_remove = Vec::new();
        let code_bytes = code.as_bytes();
        let mut i = 0;

        while i < directives.len() {
            if !mismatched_indices.contains(&i) {
                i += 1;
                continue;
            }

            let directive = &directives[i];
            match directive.name {
                "closer" => {
                    // A lone #endif, just remove its line.
                    ranges_to_remove.push(self.get_line_bounds(directive.node, code_bytes));
                    i += 1;
                }
                "intermediate" => {
                    // A dangling #else or #elif.
                    // Find its corresponding #endif to remove the whole block.
                    let start_byte = self.get_line_bounds(directive.node, code_bytes).0;
                    let mut end_byte = self.get_line_bounds(directive.node, code_bytes).1;
                    let mut block_ender_index = i;
                    let mut nesting_level = 0;

                    // Scan forward to find the matching #endif, respecting nested blocks.
                    // for j in (i + 1)..directives.len() {
                    for (j, dir) in directives.iter().enumerate().skip(i + 1) {
                        match dir.name {
                            "opener" => nesting_level += 1,
                            "closer" => {
                                if nesting_level == 0 {
                                    block_ender_index = j;
                                    end_byte = self.get_line_bounds(dir.node, code_bytes).1;
                                    break;
                                }
                                nesting_level -= 1;
                            }
                            _ => (),
                        }
                    }
                    ranges_to_remove.push((start_byte, end_byte));
                    i = block_ender_index + 1; // jump past the entire processed block.
                }
                _ => {
                    // unmatched openers are ignored, so just advance.
                    i += 1;
                }
            }
        }
        ranges_to_remove
    }

    /// Rebuilds a string by removing a given set of byte ranges.
    /// The ranges are expected to be sorted by their start byte.
    fn rebuild_string_without_ranges<'a>(
        &self,
        code: &'a str,
        mut ranges_to_remove: Vec<(usize, usize)>,
    ) -> Cow<'a, str> {
        if ranges_to_remove.is_empty() {
            return Cow::Borrowed(code);
        }

        // Ensure ranges are sorted to process them linearly.
        ranges_to_remove.sort_by_key(|r| r.0);

        let mut new_code = String::with_capacity(code.len());
        let mut last_end = 0;

        for (start, end) in ranges_to_remove {
            // Append the valid slice of code before the current range to remove.
            if start > last_end {
                new_code.push_str(&code[last_end..start]);
            }

            last_end = std::cmp::max(last_end, end);
        }

        new_code.push_str(&code[last_end..]);

        Cow::Owned(new_code)
    }

    /// Finds and fixes dangling pre-processor directives.
    ///
    /// This function identifies and removes pre-processor directives (`#else`, `#elif`, `#endif`)
    /// that do not have a corresponding opening directive (`#if`, `#ifdef`, etc.).
    ///
    /// For a lone `#endif`, it removes only that line. For a dangling `#else` or `#elif`,
    /// it removes the entire block, from the dangling directive up to its matching `#endif`.
    pub fn remove_dangling_directives<'a>(
        &self,
        code: &'a str,
        ts: &TreeSitterParser,
    ) -> Cow<'a, str> {
        let maybe_modified_code = ts.with_tree(code, |_parser, tree| {
            let deduplicated_directives = self.collect_unique_directives(tree, code);
            let mismatched_indices =
                self.identify_mismatched_directives(&deduplicated_directives, code);
            if mismatched_indices.is_empty() {
                return Cow::Borrowed(code);
            }
            let ranges_to_remove =
                self.calculate_removal_ranges(&deduplicated_directives, &mismatched_indices, code);

            self.rebuild_string_without_ranges(code, ranges_to_remove)
        });

        maybe_modified_code.unwrap_or(Cow::Borrowed(code))
    }
}

use crate::processor_lib::{
    processor::{ProcessingStats, Processor},
    utils::{
        JsonlEntry, compute_cyclomatic_complexity, compute_token_count, create_progress_bar,
        read_jsonl,
    },
};
use anyhow::{Result, anyhow};
use indicatif::ParallelProgressIterator;
use log::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use tiktoken_rs::CoreBPE;
use tiktoken_rs::cl100k_base;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedEntry {
    // Original fields we want to keep
    pub project: String,
    pub cwe: Vec<String>,
    pub target: u8,
    // Processed function body
    pub func: String,
    // New features
    pub cyclomatic_complexity: u32,
    pub token_count: u32,
    // future field
    pub cwe_desc: Vec<String>,
}

#[derive(Default, Debug)]
pub struct FeatureExtractor;

thread_local! {
    static PROCESSOR: RefCell<Processor> = RefCell::new(Processor::new().unwrap());
    static TOKENIZER: RefCell<CoreBPE> = RefCell::new(cl100k_base().unwrap());
}

impl FeatureExtractor {
    /// Reads a JSONL file, filters entries, processes them, and enriches them with features.
    pub fn extract_features(input_fp: &Path) -> Result<Vec<ProcessedEntry>> {
        let entries: Vec<JsonlEntry> = read_jsonl(input_fp)?;
        info!("Read {} entries from file.", entries.len());
        let total_entries = entries.len();

        let pb = create_progress_bar(total_entries as u64, "🚀 Premature filtering ");

        let already_empty_count = AtomicUsize::new(0);
        let comments_only_count = AtomicUsize::new(0);
        let cpp_count = AtomicUsize::new(0);

        // Track C++ label distribution
        let cpp_vulnerable_count = AtomicUsize::new(0);
        let cpp_safe_count = AtomicUsize::new(0);

        let filtered_entries: Vec<JsonlEntry> = entries
            .into_par_iter()
            .progress_with(pb)
            .filter_map(|mut entry| {
                PROCESSOR.with(|processor_cell| {
                    // borrow the processor for this thread
                    let processor = processor_cell.borrow();

                    // Check 1: Already empty BEFORE removing comments
                    if entry.func.trim().is_empty() {
                        already_empty_count.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }

                    // Step 1: remove comments
                    let code_without_comments = processor
                        .get_sanitizer()
                        .remove_comments(&entry.func, processor.get_main_ts());

                    // Check 2: Empty AFTER removing comments (comments only)
                    if code_without_comments.trim().is_empty() {
                        comments_only_count.fetch_add(1, Ordering::Relaxed);
                        return None;
                    }

                    // Step 2: Remove non-ASCII
                    let ascii_code = processor
                        .get_sanitizer()
                        .remove_non_ascii(&code_without_comments);

                    // Filter 2: C++ code
                    if processor
                        .get_sanitizer()
                        .is_cpp(&ascii_code, processor.get_cpp_ts())
                    {
                        cpp_count.fetch_add(1, Ordering::Relaxed);
                        if entry.target == 1 {
                            cpp_vulnerable_count.fetch_add(1, Ordering::Relaxed);
                        } else {
                            cpp_safe_count.fetch_add(1, Ordering::Relaxed);
                        }
                        return None;
                    }

                    // update func field
                    entry.func = ascii_code;
                    Some(entry)
                })
            })
            .collect();

        if filtered_entries.is_empty() {
            return Err(anyhow!("No valid C functions found after filtering"));
        }

        // Print statistics
        let kept_count = filtered_entries.len();
        let filtered_total = already_empty_count.load(Ordering::Relaxed)
            + comments_only_count.load(Ordering::Relaxed)
            + cpp_count.load(Ordering::Relaxed);

        println!("\n=== Filtering Results ===");
        println!("Total entries: {}", total_entries);
        println!("Filtered out:");
        println!(
            "  - Already empty: {} ({:.1}%)",
            already_empty_count.load(Ordering::Relaxed),
            (already_empty_count.load(Ordering::Relaxed) as f64 / total_entries as f64) * 100.0
        );
        println!(
            "  - Comments only: {} ({:.1}%)",
            comments_only_count.load(Ordering::Relaxed),
            (comments_only_count.load(Ordering::Relaxed) as f64 / total_entries as f64) * 100.0
        );
        println!(
            "  - C++ code: {} ({:.1}%)",
            cpp_count.load(Ordering::Relaxed),
            (cpp_count.load(Ordering::Relaxed) as f64 / total_entries as f64) * 100.0
        );
        println!(
            "  - Total filtered: {} ({:.1}%)",
            filtered_total,
            (filtered_total as f64 / total_entries as f64) * 100.0
        );
        println!(
            "Kept: {} ({:.1}%)",
            kept_count,
            (kept_count as f64 / total_entries as f64) * 100.0
        );

        let vulnerable_count = filtered_entries
            .iter()
            .filter(|entry| entry.target == 1)
            .count();

        println!("\n=== Label Distribution Comparison ===");
        println!(
            "{:<30} {:>10} {:>10} {:>10}",
            "", "Total", "Vulnerable", "Safe"
        );
        println!("{:-<62}", "");

        let cpp_vuln = cpp_vulnerable_count.load(Ordering::Relaxed);
        let cpp_safe = cpp_safe_count.load(Ordering::Relaxed);
        let cpp_total = cpp_count.load(Ordering::Relaxed);

        println!(
            "{:<30} {:>10} {:>10} ({:>5.1}%) {:>10} ({:>5.1}%)",
            "C++ (filtered out)",
            cpp_total,
            cpp_vuln,
            (cpp_vuln as f64 / cpp_total as f64) * 100.0,
            cpp_safe,
            (cpp_safe as f64 / cpp_total as f64) * 100.0
        );

        let c_vuln = vulnerable_count;
        let c_safe = kept_count - vulnerable_count;

        println!(
            "{:<30} {:>10} {:>10} ({:>5.1}%) {:>10} ({:>5.1}%)",
            "C (kept)",
            kept_count,
            c_vuln,
            (c_vuln as f64 / kept_count as f64) * 100.0,
            c_safe,
            (c_safe as f64 / kept_count as f64) * 100.0
        );

        info!(
            "Filtered down to {} entries. Starting sanitization and enrichment...",
            filtered_entries.len()
        );

        let pb = create_progress_bar(filtered_entries.len() as u64, "🔬 Sanitizing & Enriching");

        let stats = ProcessingStats::new();

        let processed_entries: Vec<ProcessedEntry> = filtered_entries
            .into_par_iter()
            .progress_with(pb)
            .filter_map(|entry| {
                PROCESSOR.with(|processor_cell| {
                    TOKENIZER.with(|tokenizer_cell| {
                        let processor = processor_cell.borrow();
                        let tokenizer = tokenizer_cell.borrow();

                        let Ok(Some((sanitized_code, tree))) =
                            processor.process_snippet_fallible(&entry.func, &stats)
                        else {
                            return None;
                        };

                        let cyclomatic_complexity =
                            compute_cyclomatic_complexity(&sanitized_code, &tree);
                        let token_count = compute_token_count(&sanitized_code, &tokenizer) as u32;

                        Some(ProcessedEntry {
                            project: entry.project,
                            cwe: entry.cwe,
                            target: entry.target,
                            func: sanitized_code,
                            cyclomatic_complexity,
                            token_count,
                            cwe_desc: Vec::new(), // initially empty
                        })
                    })
                })
            })
            .collect();

        stats.report();
        Ok(processed_entries)
    }

    pub fn count_vulnerable(entries: &[ProcessedEntry]) -> Result<()> {
        let vulnerable_count = entries.iter().filter(|entry| entry.target == 1).count();

        println!(
            "Binary labels distribution:\n\t(target=1): {}\n\t(target=0): {}",
            vulnerable_count,
            entries.len() - vulnerable_count
        );

        Ok(())
    }

    /// Saves the processed dataset to a new jsonl file.
    pub fn save_dataset(entries: &[ProcessedEntry], output_fp: &Path) -> Result<()> {
        let output_file = File::create(output_fp)?;
        let mut writer = BufWriter::new(output_file);

        for entry in entries {
            let json_string = serde_json::to_string(entry)?;
            writeln!(writer, "{}", json_string)?;
        }

        info!("{} entires saved to {:#?}", entries.len(), output_fp);
        Ok(())
    }
}

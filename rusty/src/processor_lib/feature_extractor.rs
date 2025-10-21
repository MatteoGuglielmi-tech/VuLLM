use crate::processor_lib::{
    processor::Processor,
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

        let pb = create_progress_bar(entries.len() as u64, "🚀 Premature filtering ");

        let filtered_entries: Vec<JsonlEntry> = entries
            .into_par_iter()
            .progress_with(pb)
            .filter_map(|mut entry| {
                PROCESSOR.with(|processor_cell| {
                    // borrow the processor for this thread
                    let processor = processor_cell.borrow();
                    let code_without_comments = processor
                        .get_sanitizer()
                        .remove_comments(&entry.func, processor.get_main_ts());

                    if code_without_comments.trim().is_empty() {
                        return None;
                    }

                    let ascii_code = processor
                        .get_sanitizer()
                        .remove_non_ascii(&code_without_comments);
                    if processor
                        .get_sanitizer()
                        .is_cpp(&ascii_code, processor.get_cpp_ts())
                    {
                        return None;
                    }

                    // update func field
                    entry.func = code_without_comments.into_owned();
                    Some(entry)
                })
            })
            .collect();

        if filtered_entries.is_empty() {
            return Err(anyhow!("No valid C functions found after filtering"));
        }

        info!(
            "Filtered down to {} entries. Starting sanitization and enrichment...",
            filtered_entries.len()
        );

        let pb = create_progress_bar(filtered_entries.len() as u64, "🔬 Sanitizing & Enriching");

        let processed_entries: Vec<ProcessedEntry> = filtered_entries
            .into_par_iter()
            .progress_with(pb)
            .filter_map(|entry| {
                PROCESSOR.with(|processor_cell| {
                    TOKENIZER.with(|tokenizer_cell| {
                        let processor = processor_cell.borrow();
                        let tokenizer = tokenizer_cell.borrow();

                        let Ok(Some((sanitized_code, tree))) =
                            processor.process_snippet_fallible(&entry.func)
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
                            cwe_desc: Vec::new() // initially empty
                        })
                    })
                })
            })
            .collect();

        Ok(processed_entries)
    }

    pub fn count_vulnerable(entries: &[ProcessedEntry]) -> Result<()> {
        let vulnerable_count = entries
            .iter()
            .filter(|entry| entry.target == 1)
            .count();

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

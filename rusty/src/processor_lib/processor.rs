use anyhow::anyhow;
use std::borrow::Cow;

use crate::processor_lib::lazy::{C_LANG, CPP_LANG};
use crate::processor_lib::{
    code_foundry::CodeFoundry, code_sanitizer::CodeSanitizer, tree_sitter_parser::TreeSitterParser,
    utils::is_function,
};
use tree_sitter::{InputEdit, Parser, Tree};

use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// Add these as fields in your Processor struct, or as static counters
pub struct ProcessingStats {
    pub gcc_direct_success: AtomicUsize,
    pub gcc_fallback_success: AtomicUsize,
    pub gcc_rejected: AtomicUsize,
    pub empty_after_gcc: AtomicUsize,
    pub invalid_tree: AtomicUsize,
}

impl ProcessingStats {
    pub fn new() -> Self {
        Self {
            gcc_direct_success: AtomicUsize::new(0),
            gcc_fallback_success: AtomicUsize::new(0),
            gcc_rejected: AtomicUsize::new(0),
            empty_after_gcc: AtomicUsize::new(0),
            invalid_tree: AtomicUsize::new(0),
        }
    }

    pub fn report(&self) {
        println!("=== Processing Statistics ===");
        println!(
            "GCC direct success:   {}",
            self.gcc_direct_success.load(Ordering::Relaxed)
        );
        println!(
            "GCC fallback success: {}",
            self.gcc_fallback_success.load(Ordering::Relaxed)
        );
        println!(
            "GCC rejected:         {}",
            self.gcc_rejected.load(Ordering::Relaxed)
        );
        println!(
            "Empty after GCC:      {}",
            self.empty_after_gcc.load(Ordering::Relaxed)
        );
        println!(
            "Invalid tree:         {}",
            self.invalid_tree.load(Ordering::Relaxed)
        );
    }
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self::new()
    }
}

/// The main struct that orchestrates the pre-processing pipeline.
pub struct Processor {
    foundry: CodeFoundry,
    sanitizer: CodeSanitizer,
    ts_main_parser: TreeSitterParser,
    ts_cpp_parser: TreeSitterParser,
    gcc_cache: Mutex<LruCache<String, String>>,
    clang_cache: Mutex<LruCache<String, String>>,
}

impl Processor {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            foundry: CodeFoundry,
            sanitizer: CodeSanitizer,
            ts_main_parser: TreeSitterParser::new(&C_LANG),
            ts_cpp_parser: TreeSitterParser::new(&CPP_LANG),
            gcc_cache: Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
            clang_cache: Mutex::new(LruCache::new(NonZeroUsize::new(1000).unwrap())),
        })
    }

    fn apply_sanitization<F>(
        &self,
        current_code: &mut Cow<'_, str>,
        tree: &mut Tree,
        callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: for<'a> FnOnce(&'a str, &'a Tree, &TreeSitterParser) -> anyhow::Result<Cow<'a, str>>,
    {
        let code_before = current_code.clone();

        let new_cow = callback(current_code, tree, &self.ts_main_parser)?;

        if let Cow::Owned(owned_code) = new_cow {
            let mut parser_guard = self.ts_main_parser.get_parser();
            if let Some(new_tree) =
                edit_and_reparse(&mut parser_guard, tree, &code_before, &owned_code)
            {
                *tree = new_tree;
                *current_code = Cow::Owned(owned_code);
            } else {
                return Err(anyhow!("Incremental parse failed after sanitization step."));
            }
        }
        Ok(())
    }

    /// The internal, fallible implementation of the processing pipeline.
    /// This uses `Result` and `?` for cleaner error handling.
    pub fn process_snippet_fallible(
        &self,
        code: &str,
        stats: &ProcessingStats,
    ) -> Result<Option<(String, Tree)>, anyhow::Error> {
        // --- GCC Pre-processing  ---
        let processed_code = {
            let mut cache = self.gcc_cache.lock().unwrap();
            if let Some(cached_code) = cache.get(code) {
                // Cache HIT: Use the cached result
                cached_code.clone()
            } else {
                // Cache MISS: Run the expensive GCC call
                let result = match self.sanitizer.call_gcc_preprocessor(code, None) {
                    Ok(c) => {
                        stats.gcc_direct_success.fetch_add(1, Ordering::Relaxed);
                        Ok(c)
                    }
                    Err(_) => {
                        let fixed_code = self
                            .foundry
                            .remove_dangling_directives(code, &self.ts_main_parser);
                        let balanced_code = self
                            .sanitizer
                            .balance_directives(&fixed_code, &self.ts_main_parser);

                        match self.sanitizer.call_gcc_preprocessor(&balanced_code, None) {
                            Ok(c) => {
                                stats.gcc_fallback_success.fetch_add(1, Ordering::Relaxed);
                                Ok(c)
                            }
                            Err(e) => {
                                stats.gcc_rejected.fetch_add(1, Ordering::Relaxed);
                                Err(e)
                            }
                        }
                    }
                }
                .map_err(anyhow::Error::msg)?;
                cache.put(code.to_string(), result.clone());
                result
            }
        };

        if processed_code.trim().is_empty() {
            stats.empty_after_gcc.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };

        // --- "Parse-on-Change" Pipeline ---
        let mut current_code = Cow::from(processed_code);

        let tree = {
            let mut parser_guard = self.ts_main_parser.get_parser();
            if let Some(initial_tree) = parser_guard.parse(&*current_code, None) {
                initial_tree
            } else {
                return Ok(None);
            }
        };

        let mut tree = tree;

        // --- Sequential Sanitization and Validation Steps ---
        self.apply_sanitization(&mut current_code, &mut tree, |code, tree, _parser| {
            Ok(self.sanitizer.add_missing_braces(code, tree))
        })?;

        self.apply_sanitization(&mut current_code, &mut tree, |code, tree, _| {
            self.sanitizer
                .add_missing_return_types(code, tree)
                .ok_or_else(|| anyhow!("Failed `add_missing_return_types`"))
        })?;

        self.apply_sanitization(&mut current_code, &mut tree, |code, tree, _| {
            self.sanitizer
                .validate_and_extract_body(code, tree)
                .ok_or_else(|| anyhow!("Failed `validate_and_extract_body`"))
        })?;

        let processed_code = current_code.into_owned();
        if !is_function(&processed_code, &tree) || tree.root_node().has_error() {
            stats.invalid_tree.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        let mut clang_cache = self.clang_cache.lock().unwrap();
        if let Some(cached_code) = clang_cache.get(&processed_code) {
            return Ok(Some((cached_code.clone(), tree)));
        }

        // cache miss: manually drop the lock before the expensive call.
        drop(clang_cache);
        let formatted_code = self.sanitizer.format_code(&processed_code)?;

        // Re-acquire the lock to update the cache.
        self.clang_cache
            .lock()
            .unwrap()
            .put(processed_code, formatted_code.clone());

        Ok(Some((formatted_code, tree)))
        // Ok(Some((final_code, tree)))
    }

    pub fn get_sanitizer(&self) -> &CodeSanitizer {
        &self.sanitizer
    }

    pub fn get_main_ts(&self) -> &TreeSitterParser {
        &self.ts_main_parser
    }

    pub fn get_cpp_ts(&self) -> &TreeSitterParser {
        &self.ts_cpp_parser
    }
}

fn edit_and_reparse(
    parser: &mut Parser,
    old_tree: &mut Tree,
    old_code: &str,
    new_code: &str,
) -> Option<Tree> {
    let start_byte = old_code
        .bytes()
        .zip(new_code.bytes())
        .take_while(|(a, b)| a == b)
        .count();

    let old_end_byte = old_code
        .bytes()
        .rev()
        .zip(new_code.bytes().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let new_end_byte = new_code.len() - old_end_byte;

    let edit = InputEdit {
        start_byte,
        old_end_byte: old_code.len() - old_end_byte,
        new_end_byte,
        start_position: Default::default(),
        old_end_position: Default::default(),
        new_end_position: Default::default(),
    };

    old_tree.edit(&edit);
    parser.parse(new_code, Some(old_tree))
}

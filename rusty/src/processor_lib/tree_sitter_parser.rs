use crate::processor_lib::parser_pool::{ParserPool, PooledParser};
use tree_sitter::{Language, Parser, Tree};

pub struct TreeSitterParser {
    pool: ParserPool,
}

impl TreeSitterParser {
    pub fn new(language: &'static Language) -> Self {
        Self {
            pool: ParserPool::new(language),
        }
    }

    /// Checks out a parser from the internal pool for temporary use.
    pub fn get_parser(&self) -> PooledParser<'_> {
        self.pool.get()
    }

    pub fn with_tree<F, R>(&self, code: &str, operation: F) -> Option<R>
    where
        F: FnOnce(&mut Parser, &Tree) -> R,
    {
        let mut parser_guard = self.pool.get();
        parser_guard.parse(code, None).map(|tree| operation(&mut parser_guard, &tree))
        // if let Some(tree) = parser_guard.parse(code, None) {
        //     Some(operation(&mut parser_guard, &tree))
        // } else {
        //     None
        // }
    }

}

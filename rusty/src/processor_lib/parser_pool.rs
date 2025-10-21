use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use tree_sitter::{Language, Parser};

pub struct ParserPool {
    pool: Mutex<Vec<Parser>>,
    language: &'static Language,
}

impl ParserPool {
    pub fn new(language: &'static Language) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            language,
        }
    }

    // The `get` method now returns our safe guard.
    pub fn get(&self) -> PooledParser<'_> {
        let mut pool = self.pool.lock().unwrap();
        // let mut parser = pool.pop().unwrap_or_else(Parser::new);
        let mut parser = pool.pop().unwrap_or_default();

        parser.set_language(self.language).unwrap();

        PooledParser {
            parser: Some(parser),
            pool: self,
        }
    }

    // This is now only called internally by the guard's drop method.
    pub fn release(&self, parser: Parser) {
        let mut pool = self.pool.lock().unwrap();
        pool.push(parser);
    }
}

// This struct "guards" a Parser, ensuring it's returned to the pool.
pub struct PooledParser<'a> {
    parser: Option<Parser>,
    pool: &'a ParserPool,
}

// This is the magic: when the guard is dropped, the parser is released.
impl<'a> Drop for PooledParser<'a> {
    fn drop(&mut self) {
        if let Some(parser) = self.parser.take() {
            self.pool.release(parser);
        }
    }
}

// These traits let us use the guard just like a real &Parser...
impl<'a> Deref for PooledParser<'a> {
    type Target = Parser;
    fn deref(&self) -> &Self::Target {
        self.parser.as_ref().unwrap()
    }
}

// ...and &mut Parser, making it highly ergonomic.
impl<'a> DerefMut for PooledParser<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.parser.as_mut().unwrap()
    }
}

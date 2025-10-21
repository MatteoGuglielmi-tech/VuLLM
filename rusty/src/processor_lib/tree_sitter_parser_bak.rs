use crate::processor_lib::parser_pool::ParserPool;
use std::collections::HashMap;
use tree_sitter::{
    Language, Node, Parser, Query, QueryCursor, QueryError, StreamingIterator, Tree,
};

pub struct TreeSitterParser {
    // parser: Parser,
    // language: Language,
    pool: ParserPool,
}

impl TreeSitterParser {
    // pub fn new(language_name: &str) -> Result<Self, String> {
    // let language = match language_name {
    //     "c" => tree_sitter_c::LANGUAGE.into(),
    //     "cpp" => tree_sitter_cpp::LANGUAGE.into(),
    //     "ext_c" => tree_sitter_ext_c::language(),
    //     _ => return Err(format!("Unsupported language: {}", language_name)),
    // };
    //
    // let mut parser = Parser::new();
    // parser.set_language(&language).map_err(|e| e.to_string())?;
    //
    // // if all went through, return updated struct
    // Ok(Self { parser, language })
    // }

    pub fn new() -> Self {
        Self {
            pool: ParserPool::new(),
        }
    }

    // pub fn language(&self) -> &Language {
    //     &self.language
    // }
    //
    // pub fn parse(&mut self, code: &str) -> Option<Tree> {
    //     self.parser.parse(code, None)
    // }
    //
    /// Performs a lazy, pre-order (DFS) traversal of the syntax tree.
    // pub fn traverse_tree_lazy<'a>(&self, node: Node<'a>) -> DfsIterator<'a> {
    //     DfsIterator { stack: vec![node] }
    // }
    //
    // pub fn get_parser(&mut self) -> &mut Parser {
    //     &mut self.parser
    // }
    //
    // /// Runs a query on a pre-parsed node.
    // /// Note: The original `code` is required to get the text of captures if needed.
    // pub fn run_query_on_node<'a>(
    //     &self,
    //     node: Node<'a>,
    //     query_str: &str,
    //     code: &'a str,
    // ) -> Result<HashMap<String, Vec<Node<'a>>>, QueryError> {
    //     let query = Query::new(&self.language, query_str)?;
    //     let mut cursor = QueryCursor::new();
    //     let mut captures_map = HashMap::new();
    //
    //     let mut matches_iterator = cursor.matches(&query, node, code.as_bytes());
    //     while let Some(query_match) = matches_iterator.next() {
    //         for capture in query_match.captures {
    //             // we gotta own the name here
    //             let capture_name = query.capture_names()[capture.index as usize].to_string();
    //             captures_map
    //                 .entry(capture_name)
    //                 .or_insert_with(Vec::new)
    //                 .push(capture.node);
    //         }
    //     }
    //
    //     Ok(captures_map)
    // }
    //
    // /// Runs a query on a pre-parsed tree.
    // pub fn run_query_on_tree<'a>(
    //     &self,
    //     tree: &'a Tree,
    //     query_str: &str,
    //     code: &'a str,
    // ) -> Result<HashMap<String, Vec<Node<'a>>>, QueryError> {
    //     self.run_query_on_node(tree.root_node(), query_str, code)
    // }
    //
    // /// Checks if the tree contains any error nodes.
    // pub fn has_error_nodes(&self, tree: &Tree) -> bool {
    //     self.traverse_tree_lazy(tree.root_node())
    //         .any(|node| node.is_error())
    // }
    //
    // /// Checks if the tree contains any missing nodes.
    // pub fn has_missing_nodes(&self, tree: &Tree) -> bool {
    //     self.traverse_tree_lazy(tree.root_node())
    //         .any(|node| node.is_missing())
    // }
    //
    // pub fn is_broken_tree(&self, tree: &Tree) -> bool {
    //     tree.root_node().has_error()
    //     // self.traverse_tree_lazy(tree.root_node())
    //     //     .any(|node| node.is_error() || node.is_missing())
    // }
}

// pub struct DfsIterator<'a> {
//     stack: Vec<Node<'a>>,
// }
//
// impl<'a> Iterator for DfsIterator<'a> {
//     type Item = Node<'a>;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         let node = self.stack.pop()?;
//
//         let mut cursor = node.walk();
//         let children: Vec<Node> = node.children(&mut cursor).collect();
//         for child in children.into_iter().rev() {
//             self.stack.push(child);
//         }
//
//         Some(node) // return node with value
//     }
// }

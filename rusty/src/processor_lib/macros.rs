// #[macro_export]
// macro_rules! reparse_if_owned {
//     ($cow:expr, $old_code:expr, $current_code:expr, $tree:expr, $parser:expr) => {
//         if let Cow::Owned(owned_code) = $cow {
//             if let Some(new_tree) = $crate::processor_lib::macros::edit_and_reparse($parser, &mut $tree, $old_code, &owned_code)
//             {
//                 // Update the tree and the code string for the next step.
//                 $tree = new_tree;
//                 $current_code = Cow::from(owned_code);
//             } else {
//                 // If incremental parse fails, it's an invalid state.
//                 return Ok(None);
//             }
//         }
//     };
// }

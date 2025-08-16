from tree_sitter import Language
from ._binding import language as _language_pointer

LANGUAGE = Language(_language_pointer()) # pyright: ignore[reportDeprecated]
__all__ = ["LANGUAGE"]

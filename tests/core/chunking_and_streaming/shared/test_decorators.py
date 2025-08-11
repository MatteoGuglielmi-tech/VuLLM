import pytest
from core.chunking_and_streaming.shared.decorators import *


# --- Fixtures to provide reusable test data ---
@pytest.fixture
def mock_chunk():
    """A fixture that provides a sample chunk dictionary."""
    return {"text": "int main() { return 0; }"}


@pytest.fixture
def mock_tokenizer():
    """A fixture that provides a mock tokenizer object."""
    return "mock_tokenizer_instance"


@pytest.fixture
def mock_ast_args():
    """A fixture for AST-related arguments."""
    return {
        "original_full_tree": "mock_tree_object",
        "parser": "mock_parser_instance",
    }


# --- Mock function ---
@prepare_sub_chunker_args
def apply_semantic_chunking(
    chunk: dict,
    tokenizer: Any,
    available_tokens: int,
    fn: Callable,
    _callable_args: dict | None = None,
    **fn_kwargs: Any,
):
    """A simplified version that calls the sub-function."""

    if _callable_args is None:
        raise RuntimeError("Decorator did not run!")
    return fn(**_callable_args)


# --- 2. Test functions that use the fixtures ---
def test_successful_simple_chunker(mock_chunk, mock_tokenizer):
    """Tests a successful call with a simple chunker that only needs core arguments."""

    def simple_line_chunker(chunk_metadata, tokenizer, available_tokens):
        assert chunk_metadata["text"] == "int main() { return 0; }"
        return "Success"

    result = apply_semantic_chunking(
        chunk=mock_chunk,
        tokenizer=mock_tokenizer,
        available_tokens=512,
        fn=simple_line_chunker,
    )
    assert result == "Success"


def test_successful_ast_chunker(mock_chunk, mock_tokenizer, mock_ast_args):
    """Tests a successful call with an AST chunker that requires extra arguments."""

    def ast_chunker(
        chunk_metadata, tokenizer, available_tokens, original_full_tree, parser
    ):
        assert original_full_tree == "mock_tree_object"
        return "AST Success"

    result = apply_semantic_chunking(
        chunk=mock_chunk,
        tokenizer=mock_tokenizer,
        available_tokens=512,
        fn=ast_chunker,
        **mock_ast_args,
    )

    assert result == "AST Success"


def test_failing_chunker_raises_error(mock_chunk, mock_tokenizer):
    """
    Tests that f ValueError is raised when required arguments are missing.
    This is the clean way to test for exceptions.
    """

    def failing_ast_chunker(chunk_metadata, tokenizer, original_full_tree, parser):
        pass

    with pytest.raises(ValueError) as excinfo:
        apply_semantic_chunking(
            chunk=mock_chunk,
            tokenizer=mock_tokenizer,
            available_tokens=512,
            fn=failing_ast_chunker,
        )

    assert "missing required parameter 'original_full_tree'" in str(excinfo.value)

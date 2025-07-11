# tests/core/chunking_and_streaming/test_chunking.py

import pytest
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer


# Import the function to be tested
from core.chunking_and_streaming.chunking import (
    extract_structured_chunks_with_context,
)

# =================================================================================
# 1. SETUP: Pytest Fixture for a reusable Tokenizer
# =================================================================================


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizer:
    """Provides a real tokenizer for accurate token counting in tests."""
    # Using a smaller model's tokenizer can be faster for testing if needed,
    # but using the target model's tokenizer is more accurate.
    t = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    t.pad_token_id = t.eos_token_id
    return t


# =================================================================================
# 2. THE TESTS
# =================================================================================


def test_small_code_fits_one_chunk(tokenizer: PreTrainedTokenizer):
    """
    Tests that a small C function which fits within max_tokens results in exactly one chunk.
    """
    small_code = """
    int add(int a, int b) {
        // This is a simple function.
        return a + b;
    }
    """
    max_tokens = 512  # A large enough window

    chunks = extract_structured_chunks_with_context(
        code=small_code,
        label="0",
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        trimming_technique="ast",
    )

    assert len(chunks) == 1, "Small code should not be split into multiple chunks"
    assert "return a + b;" in chunks[0]["text"]
    assert chunks[0]["function_signature"] == "int add(int a, int b)"


def test_large_code_line_based_chunking(tokenizer: PreTrainedTokenizer):
    """
    Tests that a large C function is correctly split into multiple chunks
    using the 'line' trimming technique.
    """
    long_code = """
    int calculate_stats(int *array, int n_elements, int filter_threshold) {
      int sum_positive = 0;
      #if 1>0
      int count_positive = 0;
      #endif
      int sum_negative = 0;
      int count_negative = 0;
      if (array == NULL || n_elements <= 0) {
          printf("Invalid input array or size.\\n");
          return -1;
      }
      for (int i = 0; i < n_elements; i++) {
          int current_val = array[i];
          if (current_val > 0) {
              if (current_val > filter_threshold) {
                  sum_positive += current_val;
                  count_positive++;
              }
          } else if (current_val < 0) {
              sum_negative += current_val;
              count_negative++;
          } else { 
          }
      }
      if (count_positive > 0) { 
          printf("Average positive: %f\\n", (double)sum_positive / count_positive);
      }
      return sum_positive + sum_negative;
    }
    """
    max_tokens = 100  # A small window to force chunking

    chunks = extract_structured_chunks_with_context(
        code=long_code,
        label="1",
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        trimming_technique="line",
    )

    assert len(chunks) > 1, "Long code should be split into multiple chunks"

    # Verify each chunk respects the max_tokens limit
    for chunk in chunks:
        # Re-create the full text content of the chunk
        full_chunk_text = " ".join(
            filter(
                None,
                [
                    chunk["parent"],
                    chunk["inner_context"],
                    chunk["text"],
                    chunk["trailing_context"],
                ],
            )
        )
        token_count = len(tokenizer.encode(full_chunk_text))
        # Add a small buffer for special tokens and minor inaccuracies
        assert (
            token_count <= max_tokens + 10
        ), f"Chunk exceeded token limit: {token_count} > {max_tokens}"


def test_large_code_ast_based_chunking(tokenizer: PreTrainedTokenizer):
    """
    Tests that a large C function is correctly split into multiple chunks
    using the 'ast' trimming technique.
    """
    long_code = """
    int calculate_stats(int *array, int n_elements, int filter_threshold) {
      int sum_positive = 0;
      for (int i = 0; i < n_elements; i++) {
          int current_val = array[i];
          if (current_val > 0) {
              if (current_val > filter_threshold) {
                  sum_positive += current_val;
              }
          }
      }
      return sum_positive;
    }
    """
    max_tokens = 70  # A small window to force chunking

    chunks = extract_structured_chunks_with_context(
        code=long_code,
        label="0",
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        trimming_technique="ast",
    )

    assert len(chunks) > 1, "Long code should be split using AST"
    assert all(c["label"] == "0" for c in chunks)
    assert all("calculate_stats" in c["function_signature"] for c in chunks)

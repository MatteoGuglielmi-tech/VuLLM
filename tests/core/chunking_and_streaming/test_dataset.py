import pytest
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

# Import the class to be tested
from core.chunking_and_streaming.dataset import DatasetHandler


@pytest.fixture
def dummy_tokenizer():
    """Mocks the Hugging Face tokenizer to avoid downloading real models."""
    tokenizer = MagicMock()
    # Simulate the encode method to return a list of dummy IDs
    tokenizer.encode.return_value = [1, 2, 3]
    # Set model_max_length for the _get_max_tokens_and_step method
    tokenizer.model_max_length = 2048
    return tokenizer


@pytest.fixture
def dummy_raw_data_file(tmp_path: Path) -> str:
    """Creates a fake raw JSON dataset in a temporary directory."""
    raw_data = {
        "file1.c": {
            "func": "int main() { return 0; }",
            "target": "0",
            "project": "proj_A",
        },
        "file2.c": {
            "func": "void func() { strcpy(a,b); }",
            "target": "1",
            "project": "proj_A",
        },
        "file3.c": {
            "func": "int helper() { return 1; }",
            "target": "0",
            "project": "proj_B",
        },
        "file4.c": {
            "func": "char* fn() { gets(s); }",
            "target": "1",
            "project": "proj_C",
        },
    }
    file_path = tmp_path / "dummy_dataset.json"
    with open(file_path, "w") as f:
        json.dump(raw_data, f)
    return str(file_path)


def test_dataset_handler_full_workflow(dummy_tokenizer, dummy_raw_data_file: str):
    """
    Tests the full data processing workflow from loading to chunking.
    """
    # --- Setup ---
    # Use the temporary file provided by the fixtures
    data_handler = DatasetHandler(
        pth_raw_dataset=dummy_raw_data_file,
        pth_inline_dataset="./data/inline_dataset.jsonl",
        tokenizer=dummy_tokenizer,
        max_chunk_tokens=1024,
    )

    # --- Execution ---
    data_handler.DATASET_load_raw_dataset()
    data_handler.DATASET_project_based_split()
    data_handler.DATASET_chunk()

    # --- Assertions ---
    # Assert that the raw data was loaded
    assert len(data_handler.raw_dataset) == 4

    # Assert that the split files were created
    assert os.path.exists("./data/train_data.jsonl")
    assert os.path.exists("./data/val_data.jsonl")
    assert os.path.exists("./data/test_data.jsonl")

    # Assert that the final chunked files were created
    assert os.path.exists(data_handler.pth_chunked_train)
    assert os.path.exists(data_handler.pth_chunked_val)
    assert os.path.exists(data_handler.pth_chunked_test)

    # Assert that the chunked_data attribute is populated
    assert data_handler.chunked_data is not None
    assert "train" in data_handler.chunked_data

import pytest
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from core.chunking_and_streaming.dataset import DatasetHandler
from core.chunking_and_streaming.shared.utils import save_to_jsonl
from core.chunking_and_streaming.prompt_strategies import Llama3Strategy


pytestmark: pytest.MarkDecorator = pytest.mark.slow_api

@pytest.fixture
def dummy_tokenizer():
    """Mocks the Hugging Face tokenizer to avoid downloading real models."""

    tokenizer: MagicMock = MagicMock()

    def mock_call(text, truncation=True, max_length=None, return_attention_mask=True):
        if isinstance(text, list):
            input_ids = text[0].split()
        else:
            input_ids = text.split()
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}

    tokenizer.side_effect = mock_call
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: text.split()
    tokenizer.model_max_length = 2048
    return tokenizer


@pytest.fixture
def dummy_raw_data_file(tmp_path: Path) -> Path:
    """Creates a fake raw JSON dataset in a temporary directory."""

    raw_data = [
        {
            "func": "int main() { return 0; }",
            "target": "0",
            "project": "proj_A",
            "language": "c",
            "function_description": "A simple main function.",
            "vulnerability_description": "N/A"
        },
        {
            "func": "void func() { strcpy(a,b); }",
            "target": "1",
            "project": "proj_A",
            "function_description": "A function with a buffer copy.",
            "language": "c",
            "vulnerability_description": "CWE-120: Buffer Copy without Checking Size of Input ('Classic Buffer Overflow')"
        },
        {
            "func": "int helper() { return 1; }",
            "target": "0",
            "project": "proj_B",
            "language": "c",
            "function_description": "A helper function.",
            "vulnerability_description": "N/A"
        },
        {
            "func": "char* fn() { gets(s); }",
            "target": "1",
            "project": "proj_C",
            "language": "c",
            "function_description": "A function that gets user input.",
            "vulnerability_description": "CWE-242: Use of Inherently Dangerous Function"
        },
    ]

    file_path: Path = tmp_path / "dummy_dataset.jsonl"
    save_to_jsonl(dataset=raw_data, filepath=file_path)

    return file_path


@pytest.fixture
def dummy_config_file(tmp_path: Path, dummy_raw_data_file: Path) -> Path:
    """Creates a dummy config.json file pointing to temporary paths."""
    config_data = {
        "default_input_path": str(dummy_raw_data_file),
        "pth_intermediate_train": str(tmp_path / "intermediate_train.jsonl"),
        "pth_intermediate_val": str(tmp_path / "intermediate_val.jsonl"),
        "pth_intermediate_test": str(tmp_path / "intermediate_test.jsonl"),
        "pth_final_train": str(tmp_path / "final_train.jsonl"),
        "pth_final_val": str(tmp_path / "final_val.jsonl"),
        "pth_final_test": str(tmp_path / "final_test.jsonl"),
        "pth_tokenized_train": str(tmp_path / "tokenized_train.jsonl"),
        "pth_tokenized_val": str(tmp_path / "tokenized_val.jsonl"),
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path

@pytest.fixture
def configured_data_handler(dummy_tokenizer, dummy_config_file: Path, monkeypatch):
    """Provides a fully configured DatasetHandler instance for testing."""
    with open(dummy_config_file, "r") as f:
        mock_config = json.load(f)

    monkeypatch.setattr("core.chunking_and_streaming.dataset.load_json_config", lambda filepath: mock_config)

    handler = DatasetHandler(
        tokenizer=dummy_tokenizer,
        max_chunk_tokens=50,
        trimming_technique="ast",
        prompting_strategy=Llama3Strategy()
    )
    return handler, mock_config

# --- Main Integration Test ---
def test_dataset_handler_full_workflow(dummy_tokenizer, dummy_config_file: Path, monkeypatch):
    with open(dummy_config_file, "r") as f:
        mock_config = json.load(f)

    monkeypatch.setattr(
        "core.chunking_and_streaming.dataset.load_json_config",
        lambda filepath: mock_config,
    )

    data_handler = DatasetHandler(
        tokenizer=dummy_tokenizer,
        max_chunk_tokens=50,
        trimming_technique="ast",
        prompting_strategy=Llama3Strategy()
    )

    # --- Execution ---
    data_handler.load_raw_dataset()
    assert data_handler.raw_dataset is not None
    assert len(data_handler.raw_dataset) == 4

    data_handler.project_based_split()
    assert os.path.exists(mock_config["pth_intermediate_train"])
    assert os.path.exists(mock_config["pth_intermediate_val"])
    assert os.path.exists(mock_config["pth_intermediate_test"])

    data_handler.chunk()
    assert os.path.exists(mock_config["pth_final_train"])
    assert os.path.exists(mock_config["pth_final_val"])
    assert os.path.exists(mock_config["pth_final_test"])

    processed_data = data_handler.tokenize_chunks_train()
    # Assert that the final processed data is populated and has the correct structure
    assert processed_data is not None
    assert "train" in processed_data

    # Check the content of one of the splits
    train_split = list(processed_data["train"])
    assert len(train_split) > 0
    first_item = train_split[0]

    assert "input_ids" in first_item
    assert "attention_mask" in first_item

    assert "text" not in first_item

    data_handler.cleanup_intermediate_files()
    assert not os.path.exists(mock_config["pth_intermediate_train"])
    assert not os.path.exists(mock_config["pth_intermediate_val"])
    assert not os.path.exists(mock_config["pth_intermediate_test"])


def test_dataset_handler_caching_workflow(configured_data_handler, mocker):
    """Tests the full data processing pipeline, specifically verifying that the
    tokenization caching mechanism works correctly.
    """

    data_handler, _ = configured_data_handler
    spy_execute_base = mocker.spy(data_handler, "execute_base")
    spy_tokenize_train = mocker.spy(data_handler, "tokenize_chunks_train")

    # --- 1. First Run: Generate and save the tokenized cache ---
    print("\n--- Running pipeline for the first time ---")

    # Assert that the tokenized files do not exist yet
    assert not os.path.exists(data_handler.pth_tokenized_train)
    assert not os.path.exists(data_handler.pth_tokenized_val)

    data_handler.execute_base()
    tokenized_data_first_run = data_handler.tokenize_chunks_train()

    spy_execute_base.assert_called_once()
    spy_tokenize_train.assert_called_once()

    assert os.path.exists(data_handler.pth_tokenized_train)
    assert os.path.exists(data_handler.pth_tokenized_val)

    assert "train" in tokenized_data_first_run
    assert "val" in tokenized_data_first_run
    assert len(list(tokenized_data_first_run["train"])) > 0
    assert len(list(tokenized_data_first_run["val"])) >= 0

    # --- 2. Second Run: Load directly from the cache ---
    print("\n--- Running pipeline for the second time to test caching ---")
    spy_execute_base.reset_mock()
    spy_tokenize_train.reset_mock()

    # Simulate the main script's caching logic
    if os.path.exists(data_handler.pth_tokenized_train) and os.path.exists(data_handler.pth_tokenized_val):
        print("Cache found. Loading from disk.")
        tokenized_data_second_run = data_handler.load_tokenized_dataset()
    else:
        pytest.fail("Caching logic failed: tokenized files were not found on the second run.")

    spy_execute_base.assert_not_called()
    spy_tokenize_train.assert_not_called()

    assert "train" in tokenized_data_second_run
    assert "val" in tokenized_data_second_run
    assert len(list(tokenized_data_second_run["train"])) > 0
    assert len(list(tokenized_data_second_run["val"])) >= 0


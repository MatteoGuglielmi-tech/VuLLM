import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock

# Import the class to be tested
from core.chunking_and_streaming.dataset import DatasetHandler
from core.chunking_and_streaming.shared.utils import save_to_jsonl
from core.chunking_and_streaming.prompt_strategies import Llama3Strategy

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
def config_paths(tmp_path: Path) -> dict:
    """Creates a dictionary of temporary paths for intermediate and final files."""
    return {
        "pth_intermediate_train": tmp_path / "intermediate" / "intermediate_train.jsonl",
        "pth_intermediate_val": tmp_path / "intermediate" / "intermediate_val.jsonl",
        "pth_intermediate_test": tmp_path / "intermediate" / "intermediate_test.jsonl",
        "pth_final_train": tmp_path / "final" / "final_train.jsonl",
        "pth_final_val": tmp_path / "final" / "final_val.jsonl",
        "pth_final_test": tmp_path / "final" / "final_test.jsonl",
    }

# --- Main Integration Test ---
def test_dataset_handler_full_workflow(dummy_tokenizer, dummy_raw_data_file: Path, config_paths, monkeypatch):
    for key, value in config_paths.items():
        monkeypatch.setattr(DatasetHandler, key, value, raising=False)

    data_handler = DatasetHandler(
        pth_raw_dataset=dummy_raw_data_file,
        tokenizer=dummy_tokenizer,
        max_chunk_tokens=50,
        trimming_technique="ast",
        prompting_strategy=Llama3Strategy()
    )
    # Manually set the paths after __post_init__ is called by the dataclass constructor
    for key, value in config_paths.items():
        setattr(data_handler, key, value)

    # --- Execution ---
    data_handler.load_raw_dataset()
    assert data_handler.raw_dataset is not None
    assert len(data_handler.raw_dataset) == 4

    data_handler.project_based_split()
    assert os.path.exists(config_paths["pth_intermediate_train"])
    assert os.path.exists(config_paths["pth_intermediate_val"])
    assert os.path.exists(config_paths["pth_intermediate_test"])

    data_handler.chunk()
    assert os.path.exists(config_paths["pth_final_train"])
    assert os.path.exists(config_paths["pth_final_val"])
    assert os.path.exists(config_paths["pth_final_test"])

    processed_data = data_handler.get_processed_data()
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
    assert not os.path.exists(config_paths["pth_intermediate_train"])
    assert not os.path.exists(config_paths["pth_intermediate_val"])
    assert not os.path.exists(config_paths["pth_intermediate_test"])

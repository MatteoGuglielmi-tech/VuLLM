import pytest
import json
import os
from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock

from core.chunking_and_streaming.unsloth_test import UnslothTestPipeline


@pytest.fixture
def dummy_lora_model_dir(tmp_path: Path) -> str:
    """Creates a fake LoRA model directory with the necessary config file."""

    lora_dir = tmp_path / "lora_model"
    os.makedirs(lora_dir)

    config_path = lora_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump({"base_model_name_or_path": "mock/base-model"}, f)

    return str(lora_dir)


@pytest.fixture
def dummy_test_dataframe() -> pd.DataFrame:
    """Provides a dummy DataFrame of pre-chunked test data."""

    data = {
        "text": [
            "prompt for vulnerable code",
            "prompt for safe code 1",
            "prompt for safe code 2",
        ],
        "ground_truth": ["YES", "NO", "NO"]
    }
    return pd.DataFrame(data)


def test_inference_pipeline_full_workflow(mocker, dummy_lora_model_dir: str, dummy_test_dataframe: pd.DataFrame):
    """Tests the full evaluate_on_test_set and calculate_and_save_metrics workflow."""

    # --- Mock the heavyweight methods ---
    mock_load_model = mocker.patch("core.chunking_and_streaming.unsloth_test.UnslothTestPipeline._unsloth_load_finetuned_model")
    mock_run_inference = mocker.patch(
        "core.chunking_and_streaming.unsloth_test.UnslothTestPipeline.run_inference",
        side_effect=["YES", "NO", "NO"]
    )

    # --- Setup ---
    test_pipeline = UnslothTestPipeline(lora_model_dir=dummy_lora_model_dir, max_seq_length=128)
    test_pipeline.model = MagicMock()
    test_pipeline.tokenizer = MagicMock()

    # --- Run the methods being tested ---
    results_df = test_pipeline.evaluate_on_test_set(df_test_data=dummy_test_dataframe)

    test_pipeline.calculate_and_save_metrics(
        y_true=results_df["ground_truth"].to_list(),
        y_pred=results_df["predicted_label"].to_list(),
        output_dir=test_pipeline.output_dir,
    )

    # --- Assertions ---
    # Assert that the model loader was called
    mock_load_model.assert_called_once()

    # Assert that the inference function was called for each row in the DataFrame
    assert mock_run_inference.call_count == len(dummy_test_dataframe)

    assert "predicted_label" in results_df.columns
    expected_predictions = ["YES", "NO", "NO"]
    assert results_df["predicted_label"].to_list() == expected_predictions

    # Assert that the output artifacts were created
    output_dir = Path(test_pipeline.output_dir)
    assert (output_dir / "classification_metrics.json").exists()
    assert (output_dir / "classification_report.tex").exists()
    assert (output_dir / "confusion_matrix.svg").exists()


def test_run_inference_uses_logits_processor(mocker, tmp_path: Path):
    """Verifies that the logits_processor created in __post_init__ is correctly
    passed to the model.generate() call within the run_inference method.
    """

    mocker.patch("core.chunking_and_streaming.unsloth_test.UnslothTestPipeline._unsloth_load_finetuned_model")
    fake_lora_dir = str(tmp_path / "lora_model")
    os.makedirs(fake_lora_dir, exist_ok=True)
    with open(os.path.join(fake_lora_dir, "adapter_config.json"), "w") as f:
        f.write('{"base_model_name_or_path": "unsloth/fake-model-bnb-4bit"}')

    test_pipeline = UnslothTestPipeline(lora_model_dir=fake_lora_dir, max_seq_length=128)

    test_pipeline.model = MagicMock()
    test_pipeline.tokenizer = MagicMock()
    test_pipeline.tokenizer.apply_chat_template.return_value = "dummy tensor"
    test_pipeline.tokenizer.batch_decode.return_value = ["dummy decoded output"]
    test_pipeline.tokenizer.encode.side_effect = [[123], [456]]  # dummy token IDs
    test_pipeline.logits_processor = [MagicMock()]
    test_pipeline.run_inference("some test prompt")

    test_pipeline.model.generate.assert_called_once()

    call_kwargs = test_pipeline.model.generate.call_args.kwargs

    assert "logits_processor" in call_kwargs
    assert call_kwargs["logits_processor"] is test_pipeline.logits_processor

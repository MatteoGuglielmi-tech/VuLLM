import pytest
import os
from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock

# Import the class you want to test
from core.chunking_and_streaming.unsloth_inference import InferencePipeline


@pytest.fixture
def dummy_test_df() -> pd.DataFrame:
    """Provides a sample pandas DataFrame for testing."""
    test_data = {
        "text": [
            "Prompt for a vulnerable case.",
            "Prompt for a non-vulnerable case.",
            "Prompt for another vulnerable case.",
            "Prompt for a case that will be misclassified.",
        ],
        "ground_truth": ["YES", "NO", "YES", "NO"],
    }
    return pd.DataFrame(test_data)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> str:
    """Creates a temporary directory for test artifacts."""
    return str(tmp_path)


def test_evaluation_workflow(mocker, dummy_test_df: pd.DataFrame, temp_output_dir: str):
    """
    Tests the full evaluate_on_test_set and calculate_and_save_metrics workflow.
    """
    # --- Mock the heavyweight methods ---
    mocker.patch(
        "core.chunking_and_streaming.unsloth_inference.InferencePipeline._unsloth_load_finetuned_model",
        return_value=None,
    )

    mock_predictions = [
        "YES",
        "NO",
        "YES",
        "YES",
    ]  # The last one is a misclassification

    mocker.patch(
        "core.chunking_and_streaming.unsloth_inference.InferencePipeline.run_inference",
        side_effect=mock_predictions,
    )

    # --- Instantiate the class ---
    fake_lora_dir = os.path.join(temp_output_dir, "lora_model")
    inference_pipeline = InferencePipeline(lora_model_dir=fake_lora_dir)

    # This simulates a successful model load.
    inference_pipeline.model = mocker.MagicMock()
    inference_pipeline.tokenizer = mocker.MagicMock()

    # --- Run the methods being tested ---
    results_df = inference_pipeline.evaluate_on_test_set(df_test_data=dummy_test_df)

    InferencePipeline.calculate_and_save_metrics(
        y_true=results_df["ground_truth"].to_list(),
        y_pred=results_df["predicted_label"].to_list(),
        output_dir=inference_pipeline.output_dir,
    )

    # --- Assert and Verify the results ---
    assert "predicted_label" in results_df.columns
    assert results_df["predicted_label"].tolist() == ["YES", "NO", "YES", "YES"]
    assert os.path.exists(os.path.join(temp_output_dir, "classification_metrics.json"))
    assert os.path.exists(os.path.join(temp_output_dir, "confusion_matrix.svg"))
    assert os.path.exists(os.path.join(temp_output_dir, "classification_report.tex"))

    print("\n✅ Test passed: Evaluation workflow functions as expected.")


def test_run_inference_uses_logits_processor(mocker, tmp_path: Path):
    """
    Verifies that the logits_processor created in __post_init__ is correctly
    passed to the model.generate() call within the run_inference method.
    """
    # --- 1. Arrange: Mock the model loading ---
    # We mock the loading process to avoid using a real model or GPU.
    mocker.patch(
        "core.chunking_and_streaming.unsloth_inference.InferencePipeline._unsloth_load_finetuned_model"
    )

    # --- 2. Setup: Instantiate the pipeline ---
    # The real __post_init__ will run, but _load_finetuned_model will do nothing.
    fake_lora_dir = str(tmp_path / "lora_model")
    os.makedirs(fake_lora_dir, exist_ok=True)
    with open(os.path.join(fake_lora_dir, "adapter_config.json"), "w") as f:
        f.write('{"base_model_name_or_path": "unsloth/fake-model-bnb-4bit"}')

    pipeline = InferencePipeline(lora_model_dir=fake_lora_dir)

    # --- 3. Prepare Mocks for Inference ---
    # Manually set the model and tokenizer, as loading was skipped.
    # We also mock the methods that will be called inside run_inference.
    pipeline.model = MagicMock()
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.apply_chat_template.return_value = "dummy tensor"
    pipeline.tokenizer.batch_decode.return_value = ["dummy decoded output"]

    # Manually run the part of __post_init__ that creates the processor,
    # since the real tokenizer wasn't loaded to do it automatically.
    pipeline.tokenizer.encode.side_effect = [[123], [456]]  # Dummy token IDs
    pipeline.logits_processor = [MagicMock()]  # Just needs to be a list with a mock

    # --- 4. Act: Call the method we want to test ---
    pipeline.run_inference("some test prompt")

    # --- 5. Assert: Check how model.generate was called ---
    # Assert that the generate method was called exactly once.
    pipeline.model.generate.assert_called_once()

    # Inspect the keyword arguments passed to the generate() call.
    call_kwargs = pipeline.model.generate.call_args.kwargs

    # Assert that 'logits_processor' was one of the arguments
    # and that its value is the one we set up in our pipeline instance.
    assert "logits_processor" in call_kwargs
    assert call_kwargs["logits_processor"] is pipeline.logits_processor

    print(
        "\n✅ Test passed: `logits_processor` is correctly passed to `model.generate`."
    )

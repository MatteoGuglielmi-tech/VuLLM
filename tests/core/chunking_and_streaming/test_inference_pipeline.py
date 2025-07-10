import pytest
import pandas as pd
import os
from pathlib import Path

# Import the class you want to test
from core.chunking_and_streaming.inference_pipeline import InferencePipeline


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
    mocker.patch("core.chunking_and_streaming.inference_pipeline.InferencePipeline._unsloth_load_finetuned_model", return_value=None)

    mock_predictions = [
        "YES",
        "NO",
        "YES",
        "YES",
    ]  # The last one is a misclassification

    mocker.patch("core.chunking_and_streaming.inference_pipeline.InferencePipeline.run_inference", side_effect=mock_predictions)

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

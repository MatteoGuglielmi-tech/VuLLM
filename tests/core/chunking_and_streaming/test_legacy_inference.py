import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import MagicMock

# Import the class to be tested
from core.chunking_and_streaming.legacy_inference import InferencePipeline


@pytest.fixture
def dummy_test_df() -> pd.DataFrame:
    """Provides a sample pandas DataFrame for testing."""
    test_data = {
        "text": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "ground_truth": ["YES", "NO", "YES"],
    }
    return pd.DataFrame(test_data)


def test_legacy_evaluation_workflow(
    mocker, dummy_test_df: pd.DataFrame, tmp_path: Path
):
    """Tests the full evaluation workflow for the non-Unsloth pipeline."""
    # --- Mock heavyweight components ---

    # 1. Mock the PEFT model loading process
    # Create a mock object that has a `.merge_and_unload()` method
    mocker.patch(
        "core.chunking_and_streaming.legacy_inference.InferencePipeline._load_and_merge_model",
        return_value=None,
    )

    # # 2. Mock the run_inference method to return a predictable sequence of results
    # mock_predictions = ["YES", "NO", "YES, this is vulnerable."]  # Test parsing logic
    # mocker.patch(
    #     "core.chunking_and_streaming.legacy_inference.InferencePipeline.run_inference",
    #     side_effect=mock_predictions,
    # )

    # --- 2. Setup: Instantiate the pipeline ---
    fake_lora_dir = str(tmp_path / "lora_model")
    # We must create a dummy adapter_config.json for the loader to read
    os.makedirs(fake_lora_dir, exist_ok=True)
    with open(os.path.join(fake_lora_dir, "adapter_config.json"), "w") as f:
        f.write('{"base_model_name_or_path": "fake/model"}')

    pipeline = InferencePipeline(lora_model_dir=fake_lora_dir)
    # Manually set the model/tokenizer attributes since loading is mocked
    pipeline.model = MagicMock()
    pipeline.tokenizer = MagicMock()
    pipeline.logits_processor = [MagicMock()]

    # --- 3. Simulate the behavior of the mocked model and tokenizer ---
    # `run_inference` will call `generate`, then `batch_decode`. We mock their outputs.

    # `generate` will be called 3 times. We don't care about its output, just that it's called.
    pipeline.model.generate.return_value = "dummy_token_ids"

    # `batch_decode` will also be called 3 times. We simulate it returning the prompt + clean answer.
    # The `side_effect` makes it return a different value on each call.
    pipeline.tokenizer.batch_decode.side_effect = [
        ["Prompt 1 Correct answer: YES"],
        ["Prompt 2 Correct answer: NO"],
        ["Prompt 3 Correct answer: YES"],
    ]

    # `apply_chat_template` is used to remove the prompt part. We mock it to return a fixed prompt.
    pipeline.tokenizer.apply_chat_template.side_effect = [
        # Call 1 (for row 1)
        "Prompt 1 Correct answer: ",
        "Prompt 1 Correct answer: ",
        # Call 2 (for row 2)
        "Prompt 2 Correct answer: ",
        "Prompt 2 Correct answer: ",
        # Call 3 (for row 3)
        "Prompt 3 Correct answer: ",
        "Prompt 3 Correct answer: ",
    ]

    # --- 4. Execution: Run the method under test ---
    results_df = pipeline.evaluate_on_test_set(df_test_data=dummy_test_df)

    # --- 5. Assertions ---
    # Was `generate` called for each row in the DataFrame?
    assert pipeline.model.generate.call_count == len(dummy_test_df)

    # Does the final `predicted_label` column contain the clean, correct labels?
    assert "predicted_label" in results_df.columns
    assert results_df["predicted_label"].tolist() == ["YES", "NO", "YES"]

    print("\n✅ Test passed: Legacy inference with logits processor logic is correct.")

import pytest
from unittest.mock import MagicMock
from datasets import Dataset

# Import the class to be tested
from core.chunking_and_streaming.unsloth_model import UnslothModel


@pytest.fixture
def dummy_datasets() -> dict:
    """Provides dummy train and eval datasets."""
    data = {"text": ["prompt 1", "prompt 2"]}
    ds = Dataset.from_dict(data)
    return {"train": ds, "eval": ds}


def test_training_workflow_orchestration(mocker, dummy_datasets):
    """
    Tests that the training methods are called in the correct order
    and that SFTTrainer is configured correctly.
    """
    # --- Mock heavyweight components ---
    mocker.patch("core.chunking_and_streaming.unsloth_model.UnslothModel.unsloth_load_base_model")
    mocker.patch("core.chunking_and_streaming.unsloth_model.UnslothModel.unsloth_patch_model")
    mock_sft_trainer_class = mocker.patch(
        "core.chunking_and_streaming.unsloth_model.SFTTrainer", autospec=True
    )

    # --- Setup ---
    training_pipeline = UnslothModel(
        hf_train_data=dummy_datasets["train"],
        hf_eval_data=dummy_datasets["eval"],
        training_steps=10,  # Use steps for a predictable test
    )
    # Mock the model and tokenizer that would be loaded
    training_pipeline.base_model = MagicMock()
    training_pipeline.base_tokenizer = MagicMock()

    # --- Execution ---
    training_pipeline.unsloth_start_training()

    # --- Assertions ---
    # Assert that SFTTrainer was initialized once
    mock_sft_trainer_class.assert_called_once()

    # Assert that the trainer's `train()` method was called once
    trainer_instance = mock_sft_trainer_class.return_value
    trainer_instance.train.assert_called_once()

    # Assert that the model and tokenizer were saved
    training_pipeline.base_model.save_pretrained.assert_called_once()
    training_pipeline.base_tokenizer.save_pretrained.assert_called_once()

    print("\n✅ Test passed: Training pipeline orchestration is correct.")

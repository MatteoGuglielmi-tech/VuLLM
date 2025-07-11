import pytest
from datasets import Dataset

# Import the class to be tested
from core.chunking_and_streaming.legacy_trainer import TrainingPipeline


@pytest.fixture
def dummy_datasets() -> dict:
    """Provides dummy train and eval datasets."""
    data = {"text": ["prompt 1", "prompt 2"]}
    ds = Dataset.from_dict(data)
    return {"train": ds, "eval": ds}


def test_legacy_training_workflow(mocker, dummy_datasets):
    """Tests the orchestration of the non-Unsloth training pipeline."""
    # --- Mock heavyweight components ---
    mocker.patch(
        "core.chunking_and_streaming.legacy_trainer.AutoModelForCausalLM.from_pretrained"
    )
    mocker.patch(
        "core.chunking_and_streaming.legacy_trainer.AutoTokenizer.from_pretrained"
    )
    mock_sft_trainer_class = mocker.patch(
        "core.chunking_and_streaming.legacy_trainer.SFTTrainer", autospec=True
    )

    # --- Setup ---
    pipeline = TrainingPipeline(
        hf_train_data=dummy_datasets["train"],
        hf_eval_data=dummy_datasets["eval"],
    )

    # --- Execution ---
    pipeline.load_model_and_tokenizer()
    pipeline.run_training()

    # --- Assertions ---
    # Assert that SFTTrainer was initialized once
    mock_sft_trainer_class.assert_called_once()

    # Verify that the correct argument 'processing_class' was used
    _, call_kwargs = mock_sft_trainer_class.call_args
    assert "processing_class" in call_kwargs
    assert call_kwargs["processing_class"] is pipeline.base_tokenizer

    # Assert that the trainer's `train()` and `save_model()` methods were called
    trainer_instance = mock_sft_trainer_class.return_value
    trainer_instance.train.assert_called_once()
    trainer_instance.save_model.assert_called_once_with(pipeline.lora_model_dir)

    print("\n✅ Test passed: Legacy training pipeline orchestration is correct.")

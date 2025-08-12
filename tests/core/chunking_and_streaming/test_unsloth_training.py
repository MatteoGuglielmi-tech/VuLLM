import pytest
import json
import os 
from pathlib import Path
from unittest.mock import MagicMock
from datasets import Dataset

from core.chunking_and_streaming.unsloth_trainer import UnslothModel


@pytest.fixture
def dummy_datasets() -> dict[str, Dataset]:
    """Provides dummy train and eval datasets."""
    data: dict[str, list[str]] = {"text": ["prompt 1", "prompt 2"]}
    ds: Dataset = Dataset.from_dict(data)
    return {"train": ds, "eval": ds}


def test_training_workflow_orchestration(mocker: MagicMock, dummy_datasets: dict[str,Dataset], tmp_path:Path):
    """Tests the full orchestration of the unsloth_start_training method,
    ensuring all dependencies are called correctly and that the final
    LoRA adapter is saved with coherent content.
    """

    # --- Mock Dependencies ---
    mocker.patch("core.chunking_and_streaming.unsloth_trainer.UnslothModel.unsloth_load_base_model")
    mocker.patch("core.chunking_and_streaming.unsloth_trainer.UnslothModel.unsloth_patch_model")
    mock_sft_trainer_class = mocker.patch("core.chunking_and_streaming.unsloth_trainer.SFTTrainer", autospec=True)
    mocker.patch("core.chunking_and_streaming.unsloth_trainer.UnslothModel._unsloth_config_training")

    # --- Setup ---
    training_pipeline = UnslothModel(hf_train_data=dummy_datasets["train"], hf_eval_data=dummy_datasets["eval"], training_steps=10)

    lora_save_path = tmp_path / "lora_model"
    training_pipeline.lora_model_dir = str(lora_save_path)

    def simulate_save(save_directory):
        """Simulates saving the LoRA adapter files."""

        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "adapter_config.json") # dummy adapter config
        with open(config_path, "w") as f:
            json.dump({"peft_type": "LORA", "r": 16, "lora_alpha": 16}, f)
        model_path = os.path.join(save_directory, "adapter_model.safetensors") # dummy model file
        open(model_path, "a").close()

    # mock the model and tokenizer that would be loaded
    training_pipeline.base_model = MagicMock()
    training_pipeline.base_tokenizer = MagicMock()
    training_pipeline.base_model.save_pretrained.side_effect = simulate_save

    # --- Execution ---
    training_pipeline.unsloth_start_training()

    # --- Assertions ---
    mock_sft_trainer_class.assert_called_once()
    trainer_instance = mock_sft_trainer_class.return_value
    trainer_instance.train.assert_called_once()
    training_pipeline.base_model.save_pretrained.assert_called_once_with(save_directory=str(lora_save_path))
    training_pipeline.base_tokenizer.save_pretrained.assert_called_once_with(save_directory=str(lora_save_path))

    assert os.path.isdir(lora_save_path)

    adapter_config_file = lora_save_path / "adapter_config.json"
    adapter_model_file = lora_save_path / "adapter_model.safetensors"
    assert os.path.isfile(adapter_config_file)
    assert os.path.isfile(adapter_model_file)

    with open(adapter_config_file, "r") as f:
        config_data = json.load(f)

    assert config_data["peft_type"] == "LORA"
    assert config_data["r"] == 16
    assert config_data["lora_alpha"] == 16

    print("\n✅ Test passed: Training pipeline orchestration is correct.")

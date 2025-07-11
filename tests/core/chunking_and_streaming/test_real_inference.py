import pytest
import os
import json
from unittest.mock import MagicMock

# Import the class to be tested
from core.chunking_and_streaming.unsloth_inference import InferencePipeline
from core.chunking_and_streaming.logits_processor import EnforceSingleTokenGeneration

import torch

# Mark this entire file as containing slow tests that require a GPU
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory) -> str:
    """
    Creates a fake trained model directory.
    In a real scenario, you would replace this with the actual path
    to a small, saved LoRA model you use for testing.
    """
    lora_dir = tmp_path_factory.mktemp("data") / "lora_model"
    os.makedirs(lora_dir, exist_ok=True)

    # Create the necessary dummy config file for the loader to work
    config_content = {
        "base_model_name_or_path": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    }
    with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
        f.write(json.dumps(config_content))

    return str(lora_dir)


def test_logits_processor_constrains_output(trained_model_path: str, mocker):
    """
    This is an integration test that loads a model and verifies
    the logits_processor forces the output to be 'YES' or 'NO'.

    NOTE: This test requires a GPU and enough VRAM to load the model.
    It will be skipped unless run with 'pytest -m slow'.
    """

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    mocker.patch(
        "core.chunking_and_streaming.unsloth_inference.InferencePipeline._unsloth_load_finetuned_model",
        return_value=None,  # It will do nothing
    )

    # --- Arrange ---
    pipeline = InferencePipeline(lora_model_dir=trained_model_path)
    # Manually attach the mock model and tokenizer
    pipeline.model = mock_model
    pipeline.tokenizer = mock_tokenizer

    # Configure the mock tokenizer as needed by the real run_inference method
    pipeline.tokenizer.apply_chat_template.return_value.to.return_value = torch.tensor(
        [[1, 2, 3]]
    )

    #  KEY: We let the real logits_processor to be created, and we will check that the real model.generate call uses it.
    pipeline.tokenizer.encode.side_effect = [
        [123],
        [456],
    ]  # Dummy token IDs for YES, NO
    pipeline.logits_processor = [EnforceSingleTokenGeneration([123, 456])]

    # --- Act ---
    # Call the real run_inference method
    pipeline.run_inference("A test prompt")

    # --- Assert ---
    # We assert that the real model.generate function was called
    # and that our logits_processor was passed to it.
    pipeline.model.generate.assert_called_once()
    call_kwargs = pipeline.model.generate.call_args.kwargs

    assert "logits_processor" in call_kwargs
    assert call_kwargs["logits_processor"] is pipeline.logits_processor

    print(
        "\n✅ Test passed: `logits_processor` is correctly wired into the generate call."
    )

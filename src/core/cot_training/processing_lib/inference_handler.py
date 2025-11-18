from collections.abc import Generator
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import json
import torch
import logging

from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from datasets import Dataset, DatasetDict, load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizer

from .prompt_config import Messages, ParsedResponse, VulnerabilityPromptConfig
from ..utilities import is_main_process, rich_progress, rich_progress_manual


logger = logging.getLogger(name=__name__)


@dataclass
class TestHandler:
    lora_model_dir: Path|str
    evaluated_testset_path: Path
    max_seq_length: int
    max_new_tokens: int
    chat_template: str|None = None

    model: FastLanguageModel|None = field(init=False, default=None, repr=False)
    tokenizer: PreTrainedTokenizer|None = field(init=False, default=None, repr=False)

    prompt_config = VulnerabilityPromptConfig()
    _counter_fails: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        self._validate_inputs()

        logger.info(f"🔧 Initializing TestHandler with:")
        logger.info(f"   LoRA checkpoint: {self.lora_model_dir}")
        logger.info(f"   Max sequence length: {self.max_seq_length}")
        logger.info(f"   Max new tokens per answer: {self.max_new_tokens}")
        logger.info(f"   Custom chat template: {self.chat_template is not None}")

        self._load_finetuned_model()

    def _validate_inputs(self):
        """Validate constructor inputs."""

        if isinstance(self.lora_model_dir, str):
            self.lora_model_dir = Path(self.lora_model_dir)

        # Check checkpoint exists
        if not self.lora_model_dir.exists():
            raise FileNotFoundError(f"LoRA checkpoint directory not found: {self.lora_model_dir}")

        # Check for adapter files (LoRA checkpoint validation)
        self.adapter_config = self.lora_model_dir / "adapter_config.json"
        adapter_model = self.lora_model_dir / "adapter_model.safetensors"

        if not self.adapter_config.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {self.lora_model_dir}. "
                f"Is this a valid LoRA checkpoint?"
            )

        if not adapter_model.exists() and not (self.lora_model_dir / "adapter_model.bin").exists():
            logger.warning(
                f"⚠️  No adapter weights found in {self.lora_model_dir}. "
                f"Expected adapter_model.safetensors or adapter_model.bin"
            )

        # Validate numeric parameters
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")

        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")

        if self.max_new_tokens > self.max_seq_length:
            logger.warning(
                f"⚠️  max_new_tokens ({self.max_new_tokens}) > max_seq_length ({self.max_seq_length}). "
                f"This may cause issues during generation."
            )

    def _set_padding_strategy(self):
        """Set padding side based on model architecture.

        - Decoder-only (GPT, Llama): Left padding (for batch generation)
        - Encoder-decoder (T5, BART): Right padding
        """

        decoder_only_models: set[str] = { "llama", "mistral", "qwen", "opt", "phi", "gemma"}
        model_type = getattr(self.model.config, "model_type", "").lower()

        if any(arch in model_type for arch in decoder_only_models):
            self.tokenizer.padding_side = "left"
            logger.info(f"📍 Set padding_side='left' for decoder-only model ({model_type})")
        else:
            self.tokenizer.padding_side = "right"
            logger.info(f"📍 Set padding_side='right' for encoder-decoder model ({model_type})")

        # ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"🔧 Set pad_token to eos_token: {self.tokenizer.eos_token}")

    def _configure_tokenizer(self):
        """Configure tokenizer settings (chat template, padding, special tokens)."""

        if self.chat_template is not None:
            logger.info(f"🎨 Applying custom chat template: {self.chat_template}")
            try:
                self.tokenizer = get_chat_template(
                    self.tokenizer, 
                    chat_template=self.chat_template
                )
            except ValueError as e:
                logger.error(f"Invalid chat template: {self.chat_template}")
                logger.error(f"Available templates: {list(CHAT_TEMPLATES.keys())}")
                raise ValueError(f"Chat template '{self.chat_template}' not found") from e

        self._set_padding_strategy()

    def _load_finetuned_model(self):
        """Loads the quantized base model and applies fine-tuned LoRA adapter for inference.

        Process:
            1. Extract base model name from adapter config
            2. Load base model in 4-bit quantization
            3. Apply LoRA adapter weights
            4. Enable Unsloth fast inference mode
            5. Configure tokenizer (chat template, padding)
        """

        if self.model is not None:
            logger.info("Model already loaded, skipping...")
            return

        with open(file=self.adapter_config, mode="r") as f:
            adapter_config = json.load(f)

        base_model_name = adapter_config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                "base_model_name_or_path not found in adapter_config.json. "
                "Cannot determine which base model to load."
            )

        logger.info(f"📦 Base model: {base_model_name}")
        logger.info(f"🔧 Loading from: {self.lora_model_dir}")

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                load_in_4bit=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer failed to load")

            logger.info("✅ Base model and tokenizer loaded successfully")

            self.model.load_adapter(self.lora_model_dir)  # type: ignore
            logger.info("✅ LoRA adapter applied successfully.")

            FastLanguageModel.for_inference(model=self.model)
            logger.info("⚡ Unsloth fast inference mode enabled (2x speedup)")

            self._configure_tokenizer()

        except torch.cuda.OutOfMemoryError as oom:
            logger.critical("💥 OUT OF MEMORY!")
            logger.critical(f"Model requires more VRAM than available.")
            if is_main_process():
                print(f"Solutions:")
                print(f"  1. Use a smaller model")
                print(f"  2. Reduce max_seq_length (current: {self.max_seq_length})")
                print(f"  3. Use 8-bit quantization instead of 4-bit")
            torch.cuda.empty_cache()
            raise oom

        except Exception as e:
            logger.critical(f"❌ Failed to load model: {e}")
            raise

    def run_inference(self, c_code_input: str) -> ParsedResponse:
        """Performs inference on a single C code snippet using the CoT format.

        Parameters
        ----------
        c_code_input : str
            The raw C function code to be analyzed.

        Returns
        -------
        str
            The model's generated reasoning and final answer (assistant response only).

        Raises
        ------
        RuntimeError
            If model/tokenizer not loaded or if generation fails.
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before running inference.")

        # build message structure
        messages: list[dict] = self.prompt_config.as_messages(func_code=c_code_input)

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,  # type: ignore
            return_tensors="pt",
            # padding=True,
            padding=False,
            max_length=self.max_seq_length,
            truncation=True,
        ).to(self.model.device)  # type: ignore

        # Generate response
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95,
                    min_p=0.1,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Model generation failed: {e}") from e

        generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
        decoded_output = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, cleanup_tokenization_spaces=True
        )[0]

        return self._parse_response(decoded_output.strip())

    def evaluate_on_test_set(
        self,
        test_dataset: Dataset,
        batch_size: int,
        use_batching: bool = True,
    ) -> Dataset:
        """Run inference on test dataset and return predictions.

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset with 'func' field containing code samples.
        batch_size : int, default=16
            Batch size for inference.
        use_batching : bool, default=True
            Whether to use batched inference (faster) or sequential (debugging).

        Returns
        -------
        Dataset
            - Dataset with added 'model_prediction' column
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded.")

        if not len(test_dataset) > 0:
            raise ValueError("Provided dataset is emtpy")

        self.n_samples: int = len(test_dataset)

        logger.info(
            f"🔍 Evaluating on {self.n_samples} test samples "
            f"(batch_size={batch_size}, batching={use_batching})..."
        )

        if use_batching:
            predictions = self._batched_inference(
                test_dataset=test_dataset, batch_size=batch_size
            )
        else:
            predictions = self._sequential_inference(test_dataset=test_dataset)

        # add predictions to dataset
        results_dataset: Dataset = test_dataset.add_column( # type: ignore
            name="model_prediction", column=predictions
        )

        self.save_evaluation_results(results_dataset=results_dataset)

        return results_dataset

    # todo: create utility function for this
    # similar method in `DatasetHandler`
    def save_evaluation_results(
        self,
        results_dataset: Dataset,
        split_name: str = "test"
    ):
        """Save evaluation results with custom split name.

        Parameters
        ----------
        results_dataset : Dataset
            Dataset containing evaluation results
        output_dir : Path
            Directory to save to
        split_name : str
            Name of the split (e.g., "test", "validation", "test_ood")
        """

        dataset_dict = DatasetDict({split_name: results_dataset})

        dataset_dict.save_to_disk(dataset_dict_path=self.evaluated_testset_path)
        logger.info(
            f"✅ Saved {len(results_dataset)} samples to {self.evaluated_testset_path} (split: {split_name})"
        )

    # todo: create utility function for this
    # similar method in `DatasetHandler`
    @staticmethod
    def load_test_dataset(input_dir: Path, split_name: str = "test") -> Dataset:
        """Load evaluation results from disk.
        Dataset needs to be previously saved via [~TestHandler.save_evaluation_results].

        Parameters
        ----------
        input_dir : Path
            Directory to load from
        split_name : str
            Name of the split to load

        Returns
        -------
        Dataset
            Loaded dataset
        """
        loaded: Dataset|DatasetDict = load_from_disk(dataset_path=input_dir)

        if isinstance(loaded, DatasetDict):
            if split_name not in loaded:
                raise KeyError(
                    f"`{split_name}` split not found. Available: {list(loaded.keys())}"
                )
            return loaded["test"]
        elif isinstance(loaded, Dataset):
            return loaded
        else:
            raise TypeError(
                f"Expected Dataset or DatasetDict, got {type(loaded).__name__}"
            )

    def _sequential_inference(self, test_dataset: Dataset) -> list[dict[str, Any]]:
        """Run inference sequentially using run_inference() method."""

        predictions: list[dict[str, Any]] = []
        n_failures: int = 0
        n_ok: int = 0
        with rich_progress_manual(
            total=self.n_samples, description="Sequential Inference"
        ) as pbar:
            for index in range(self.n_samples):
                try:
                    prediction = self.run_inference(test_dataset[index]["func"])
                    predictions.append(prediction.to_dict())
                    n_ok += 1
                except Exception:
                    n_failures += 1
                    predictions.append(
                        ParsedResponse(
                            reasoning={},
                            vulnerabilities=[],
                            verdict={},
                            parse_error=True,
                        ).to_dict()
                    )
                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix({
                        "✓ Ok": n_ok,
                        "✗ Error": n_failures,
                        "% Error rate": (
                            f"{(n_failures/self.n_samples):.1%}"
                            if n_failures > 0
                            else "0%"
                        ),
                    })

                if index == self.n_samples-1:
                    pbar.set_description(description="✅ Evaluation complete.")

        return predictions

    def _create_message_batches(
        self, dataset: Dataset, *, batch_size: int
    ) -> Generator[list[Messages], None, None]:
        """Generate batches of formatted messages for inference."""

        batch: list[Messages] = []

        for func in dataset["func"]:
            messages = self.prompt_config.as_messages(func_code=func)
            batch.append(messages)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def _batched_inference(self, test_dataset: Dataset, batch_size: int) -> list[dict[str, Any]]:
        """Run batched inference for speed. Processes multiple samples simultaneously."""

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before running evaluation.")

        all_predictions: list[dict[str, Any]] = []
        num_batches = (len(test_dataset["func"]) + batch_size - 1) // batch_size

        for batch_messages in rich_progress(
            self._create_message_batches(test_dataset, batch_size=batch_size),
            total=num_batches,
            description="Batched Inference",
            status_fn=lambda b: f"{len(b)} samples"
        ):
            input_texts = self.tokenizer.apply_chat_template(
                batch_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                input_texts, # type: ignore
                return_tensors="pt",
                padding=True,
                max_length=self.max_seq_length,
                truncation=True,
            ).to(self.model.device)  # type: ignore

            try:
                with torch.inference_mode():
                    outputs = self.model.generate( # type: ignore
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.2,
                        top_p=0.95,
                        min_p=0.1,
                        repetition_penalty=1.05,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
                decoded_predictions = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                all_predictions.extend(
                    [
                        self._parse_response(p.strip()).to_dict()
                        for p in decoded_predictions
                    ]
                )

            except Exception:
                # if generation fails, proceed with next batch
                # hopefully this never happens
                logger.exception(f"Batch inference failed")
                continue

        return all_predictions

    def _parse_response(self, response: str) -> ParsedResponse:
        """Parse judge response into structured ParsedResponse.

        Parameters
        ----------
        response : str
            Raw text response from the judge model

        Returns
        -------
        ParsedResponse
            Parsed text for generated output
        """

        try:
            # extract json content
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                self._counter_fails += 1
                raise ValueError("No JSON found in response")

            return ParsedResponse(
                reasoning=result.get("reasoning"),
                vulnerabilities=result.get("vulnerabilities"),
                verdict=result.get("verdict"),
                parse_error=False,
            )

        except (json.JSONDecodeError, ValueError, KeyError):
            return ParsedResponse(
                reasoning={},
                vulnerabilities=[],
                verdict={},
                parse_error=True,
            )


# Perform Detailed Error Analysis
# Find False Positives (model predicted 1, but the label was 0):
# false_positives = results.filter(
#     lambda ex: ex["target"] == 0 and ex["predicted_label"] == 1
# )
# Find False Negatives (model predicted 0, but the label was 1):
# false_negatives = results.filter(
#     lambda ex: ex["target"] == 1 and ex["predicted_label"] == 0
# )
#
# You can analyze if the model has biases or performs differently on subsets of your data.
# df = results.to_pandas()
# # Calculate accuracy for each project
# project_accuracy = df.groupby("project").apply(
#     lambda x: (x["target"] == x["predicted_label"]).mean()
# )
# print(project_accuracy)


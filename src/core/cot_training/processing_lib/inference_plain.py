from pydantic import ValidationError
from unsloth import FastLanguageModel, is_bfloat16_supported
# from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import json
import torch
import logging

from typing import Literal, overload, Iterator
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, field

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


from .prompt_config import VulnerabilityPromptConfig
from .datatypes import (
    AssumptionMode,
    EmptyReasoningError,
    ExpectedModelResponse,
    MismatchCWEError,
    PromptPhase,
    TestDatasetSchema,
    TypedDataset,
    GenerationError,
)

from .cwe_diagostic_mixin import CWEDiagnosticMixin
from .evaluation_handler import CWEPair
from .prompt_config import Messages, VulnerabilityPromptConfig
from ..utilities import (
    RichColors,
    is_main_process,
    rich_progress_manual,
    build_table,
    rich_panel,
    rich_progress,
    # validate_filepath_extension
    dump_yaml
)

logger = logging.getLogger(name=__name__)


@dataclass
class TestHandlerPlain(CWEDiagnosticMixin):
    lora_path: Path | str
    evaluated_testset_path: Path
    max_seq_length: int
    max_new_tokens: int
    prompt_mode: PromptPhase
    assumption_mode: AssumptionMode
    add_hierarchy: bool
    chat_template: str | None = None

    model: FastLanguageModel | None = field(init=False, default=None, repr=False)
    tokenizer: PreTrainedTokenizer | None = field(init=False, default=None, repr=False)

    prompt_config = VulnerabilityPromptConfig()
    _counter_fails: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        self._validate_inputs()

        d_table = {
            "LoRA checkpoint:": self.lora_path,
            "Max sequence length": f"{self.max_seq_length:,}",
            "Max new tokens per answer": f"{self.max_new_tokens:,}",
            "Custom chat template": self.chat_template is not None,
        }

        tb = build_table(data=d_table, show_header=False)
        rich_panel(
            tb,
            panel_title=f"🔧 Initializing TestHandler with LoRA checkpoint: {self.lora_path}",
            border_style=RichColors.MEDIUM_PURPLE1,
        )

        self._load_finetuned_model()

    def _validate_inputs(self):
        """Validate constructor inputs."""

        if isinstance(self.lora_path, str):
            self.lora_path = Path(self.lora_path)

        if self.prompt_mode in [
            PromptPhase.CONSTRAINED_TRAINING,
            PromptPhase.FREE_TRAINING,
        ]:
            raise ValueError(
                f"THIS IS INFERENCE! Provided {self.prompt_mode} prompt mode!"
            )

        # Check checkpoint exists
        if not self.lora_path.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint directory not found: {self.lora_path}"
            )

        # Check for adapter files (LoRA checkpoint validation)
        self.adapter_config = self.lora_path / "adapter_config.json"
        adapter_model = self.lora_path / "adapter_model.safetensors"

        if not self.adapter_config.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in {self.lora_path}. "
                f"Is this a valid LoRA checkpoint?"
            )

        if (
            not adapter_model.exists()
            and not (self.lora_path / "adapter_model.bin").exists()
        ):
            logger.warning(
                f"⚠️  No adapter weights found in {self.lora_path}. "
                f"Expected adapter_model.safetensors or adapter_model.bin"
            )

        # Validate numeric parameters
        if self.max_seq_length <= 0:
            raise ValueError(
                f"max_seq_length must be positive, got {self.max_seq_length}"
            )

        if self.max_new_tokens <= 0:
            raise ValueError(
                f"max_new_tokens must be positive, got {self.max_new_tokens}"
            )

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

        decoder_only_models: set[str] = { "llama", "mistral", "qwen", "opt", "phi", "gemma"} # deepseek uses "llama"
        model_type = getattr(self.model.config, "model_type", "").lower()  # type: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]

        if any(arch in model_type for arch in decoder_only_models):
            self.tokenizer.padding_side = "left"  # type: ignore[reportOptionalMemberAccess]
            logger.info(f"📍 Set padding_side='left' for decoder-only model ({model_type})")
        else:
            self.tokenizer.padding_side = "right"  # type: ignore[reportOptionalMemberAccess]
            logger.info(f"📍 Set padding_side='right' for encoder-decoder model ({model_type})")

        # ensure pad token exists
        if not (self.tokenizer.pad_token or self.tokenizer.pad_token_id):  # type: ignore[reportOptionalMemberAccess]
            self.tokenizer.pad_token = self.tokenizer.eos_token  # type: ignore[reportOptionalMemberAccess]
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # type: ignore[reportOptionalMemberAccess]

            logger.info(f"🔧 Set pad_token to eos_token: {self.tokenizer.eos_token}")  # type: ignore[reportOptionalMemberAccess]

    def _configure_tokenizer(self):
        """Configure tokenizer settings (chat template, padding, special tokens)."""

        model_name_lower = self.base_model_name.lower()

        # DeepSeek models
        if "deepseek" in model_name_lower:
            logger.info("🔍 Detected DeepSeek model - applying fixed chat template")

            self.tokenizer.chat_template = (  # type: ignore[reportOptionalMemberAccess]
                "{% if not add_generation_prompt is defined %}"
                "{% set add_generation_prompt = false %}"
                "{% endif %}"
                "{{ bos_token }}"
                "{%- for message in messages %}"
                "    {%- if message['role'] == 'system' %}"
                "{{ message['content'] + '\\n' }}"
                "    {%- else %}"
                "        {%- if message['role'] == 'user' %}"
                "{{ '### Instruction:\\n' + message['content'] + '\\n' }}"
                "        {%- else %}"
                "{{ '### Response:\\n' + message['content'] + '\\n' + eos_token + '\\n' }}"
                "        {%- endif %}"
                "    {%- endif %}"
                "{%- endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '### Response:\\n' }}"
                "{% endif %}"
            )
            logger.info("✅ Applied fixed DeepSeek chat template")

        # CodeLlama models
        elif "codellama" in model_name_lower:
            logger.info("🔍 Detected CodeLlama model - applying Llama 2 chat template")
            from unsloth.chat_templates import get_chat_template

            self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama")
            logger.info("✅ Applied Llama 2 chat template")

        # Custom template
        elif self.chat_template is not None:
            logger.info(f"🎨 Applying chat template: {self.chat_template}")
            try:
                from unsloth.chat_templates import get_chat_template

                self.tokenizer = get_chat_template(
                    self.tokenizer, chat_template=self.chat_template
                )
                logger.info(f"✅ Applied {self.chat_template} chat template")
            except ValueError as e:
                logger.error(f"Invalid chat template: {self.chat_template}")
                raise ValueError(
                    f"Chat template '{self.chat_template}' not found"
                ) from e

        # Default
        else:
            if (
                hasattr(self.tokenizer, "chat_template")
                and self.tokenizer.chat_template  # type: ignore[reportOptionalMemberAccess]
            ):
                logger.info("ℹ️  Using model's default chat template")
            else:
                logger.warning("⚠️  No chat template found or specified!")

        self._set_padding_strategy()

    @contextmanager
    def _tokenizer_config(self):
        try:
            self.tokenizer.padding_side = self.original_tokenizer_params["padding_side"]  # type: ignore[reportOptionalMemberAccess]
            self.tokenizer.pad_token = self.original_tokenizer_params["pad_token"]  # type: ignore[reportOptionalMemberAccess]

            if hasattr(self.model, "tokenizer"):
                self.model.tokenizer.tokenizer.padding_side = (  # type: ignore[reportOptionalMemberAccess]
                    self.original_tokenizer_params["padding_side"]
                )

                self.model.tokenizer.pad_token = (  # type: ignore[reportOptionalMemberAccess]
                    self.original_tokenizer_params["pad_token"]
                    or self.original_tokenizer_params["eos_token"]
                )

            yield
        finally:
            pass

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

        self.base_model_name = adapter_config.get("base_model_name_or_path")
        if not self.base_model_name:
            raise ValueError(
                "base_model_name_or_path not found in adapter_config.json. "
                "Cannot determine which base model to load."
            )

        logger.info(f"📦 Base model: {self.base_model_name}")
        logger.info(f"🔧 Loading from: {self.lora_path}")

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_name,
                max_seq_length=self.max_seq_length,
                dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                load_in_4bit=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("Model or tokenizer failed to load")

            logger.info("✅ Base model and tokenizer loaded successfully")

            self.model.load_adapter(self.lora_path)  # type: ignore
            logger.info("✅ LoRA adapter applied successfully.")

            FastLanguageModel.for_inference(model=self.model)
            logger.info("⚡ Unsloth fast inference mode enabled (2x speedup)")

            self._configure_tokenizer()

            self.original_tokenizer_params = {
                "padding_side": self.tokenizer.padding_side,
                "pad_token": self.tokenizer.pad_token,
                "eos_token": self.tokenizer.eos_token,
            }

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

    def run_inference(
        self, input_code: str, n_retries: int = 3, **override_params
    ) -> ExpectedModelResponse:
        """
        Performs prediction (with retry logic) and validates output for a single code snippet.

        Parameters
        ----------
        input_code : str
            The raw C function code to be analyzed
        n_retries : int, default=3
            Number of retry attempts on validation failure
        **override_params
            Override default generation parameters

        Returns
        -------
        str
            The model's generated JSON response (validated)

        Raises
        ------
        RuntimeError
            If model/tokenizer not loaded
        GenerationError
            If generation fails after all retries
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError(
                "Model and tokenizer must be loaded before running inference."
            )

        # build message structure
        messages: list[dict] = self.prompt_config.as_messages(
            func_code=input_code,
            phase=self.prompt_mode,
            mode=self.assumption_mode,
            add_hierarchy=self.add_hierarchy
        )

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,  # type: ignore
            return_tensors="pt",
            padding=False,
            max_length=self.max_seq_length,
            truncation=True,
        ).to(self.model.device)  # type: ignore

        # Merge generation params
        gen_params = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 50,
            "min_p": 0.1,
            "repetition_penalty": 1.05,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **override_params,
        }

        for attempt in range(n_retries):
            try:
                with self._tokenizer_config(), torch.inference_mode():
                    outputs = self.model.generate(**inputs, **gen_params)  # type: ignore[reportAttributeAccessIssue]
                    generated_tokens = outputs[:, inputs.input_ids.shape[1] :]
                    decoded_output = self.tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True,
                        cleanup_tokenization_spaces=True,
                    )[0]

                    try:
                        return ExpectedModelResponse.model_validate_json(decoded_output)

                    except (
                        json.JSONDecodeError,
                        ValidationError,
                        MismatchCWEError,
                        EmptyReasoningError,
                    ) as e:
                        if attempt < n_retries - 1:
                            logger.warning(
                                f"Validation failed (attempt {attempt+1}/{n_retries}): {e}"
                            )
                            continue  # Retry generation
                        else:
                            # Last attempt failed
                            raise GenerationError(
                                f"Validation failed after {n_retries} attempts: {e}"
                            )

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM on attempt {attempt+1}: {e}")
                torch.cuda.empty_cache()  # Try to free memory
                if attempt < n_retries - 1:
                    continue
                else:
                    raise GenerationError(f"CUDA OOM after {n_retries} attempts")

            except KeyboardInterrupt:
                raise  # Pass through user interrupt

            except Exception as e:
                logger.error(
                    f"Generation failed (attempt {attempt+1}/{n_retries}): {e}"
                )
                if attempt < n_retries - 1:
                    continue
                else:
                    raise GenerationError(
                        f"Generation failed after {n_retries} attempts: {e}"
                    )

        # Should never reach here
        raise GenerationError("Unexpected: retry loop completed without return")

    def evaluate_on_test_set(
        self,
        test_dataset: Dataset,
        batch_size: int = 4,
        n_retries: int = 3,
        use_batching: bool = False,
    ) -> TypedDataset[TestDatasetSchema]:
        """
        Run inference on test dataset and return predictions.

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset with 'func' field
        batch_size : int, default=4
            Batch size for inference (if use_batching=True)
        n_retries : int, default=3
            Number of retries per sample for failed generations
        use_batching : bool, default=False
            Whether to use batch inference (sequential mode is more reliable)

        Returns
        -------
        TypedDataset[TestDatasetSchema]
            Dataset with model predictions (aligned with successful samples).
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
            predictions, success_indices = self._batched_inference(
                test_dataset=test_dataset, batch_size=batch_size, n_retries=n_retries
            )
        else:
            predictions, success_indices = self._sequential_inference(
                test_dataset=test_dataset, n_retries=n_retries
            )

        if not predictions:
            raise RuntimeError("All samples failed inference!")

        # add predictions to dataset
        aligned_dataset = test_dataset.select(success_indices)
        results_dataset = aligned_dataset.add_column(  # type: ignore[reportCallIssue]
            name="model_prediction",
            column=[p.model_dump_json() for p in predictions],
        )

        self.save_evaluation_results(results_dataset=results_dataset)

        return TypedDataset[TestDatasetSchema](results_dataset)

    def save_evaluation_results(self, results_dataset: Dataset) -> None:
        """Save evaluation results with custom split name.
        It saved the datset both in a hugging face compatible format as well
        as in json line format for human inspection.

        Parameters
        ----------
        results_dataset : Dataset
            Dataset containing evaluation results
        """

        from ..utilities import save_dataset

        huggingface: Path = self.evaluated_testset_path / "huggingface"
        jsonl: Path = self.evaluated_testset_path / "json"
        huggingface.mkdir(exist_ok=True, parents=True)

        save_dataset(
            dataset=results_dataset,
            output_location=huggingface,
            split_name="test",
        )

        jsonl.mkdir(exist_ok=True, parents=True)
        results_dataset.to_json(path_or_buf=(jsonl / "eval_test.jsonl"))

    @staticmethod
    @overload
    def load_test_dataset(
        input_dir: Path, split_name: str = "test", *, with_eval: Literal[True]
    ) -> TypedDataset[TestDatasetSchema]: ...

    @staticmethod
    @overload
    def load_test_dataset(
        input_dir: Path, split_name: str = "test", *, with_eval: Literal[False] = False
    ) -> Dataset: ...

    @staticmethod
    def load_test_dataset(
        input_dir: Path, split_name: str = "test", *, with_eval: bool = False
    ) -> Dataset | TypedDataset[TestDatasetSchema]:
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

        Raise
        -----
        ValueError
            Shouldn't happen but it's here for safety in case of type mismatched
        """

        from ..utilities import load_dataset_from_disk

        test_data: Dataset = load_dataset_from_disk(
            input_dir=input_dir, split_name=split_name
        )
        return TypedDataset[TestDatasetSchema](test_data) if with_eval else test_data

    def _sequential_inference(
        self, test_dataset: Dataset, n_retries: int = 3
    ) -> tuple[list[ExpectedModelResponse], list[int]]:
        """
        Run inference sequentially with retry logic and proper error tracking.

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset with 'func' field
        n_retries : int, default=3
            Number of retries per sample

        Returns
        -------
        tuple[list[ExpectedModelResponse], list[int]]
            (predictions, success_indices)
        """

        predictions: list[ExpectedModelResponse] = []
        success_indices: list[int] = []
        n_failures: int = 0
        n_ok: int = 0

        with rich_progress_manual(
            total=self.n_samples, description="Sequential Inference"
        ) as pbar:

            for index in range(self.n_samples):
                try:
                    prediction = self.run_inference(
                        test_dataset[index]["func"], n_retries=n_retries
                    )

                    predictions.append(prediction)
                    success_indices.append(index)
                    n_ok += 1

                except GenerationError as e:
                    # Failed after all retries
                    logger.error(f"Sample {index} failed: {e}")
                    n_failures += 1
                    # Don't add to predictions or success_indices

                except KeyboardInterrupt:
                    logger.warning("Interrupted by user")
                    raise

                except Exception as e:
                    # Unexpected error
                    logger.exception(f"Unexpected error on sample {index}: {e}")
                    n_failures += 1

                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix(
                        {
                            "✓ Ok": n_ok,
                            "✗ Error": n_failures,
                            "% Error rate": (
                                f"{(n_failures/self.n_samples):.1%}"
                                if n_failures > 0
                                else "0%"
                            ),
                        }
                    )

            pbar.set_description("✅ Sequential inference complete")

        return predictions, success_indices

    def _create_message_batches(
        self, dataset: Dataset, *, batch_size: int
    ) -> Iterator[list[Messages]]:
        """Generate batches of formatted messages for inference."""

        batch: list[Messages] = []

        for func in dataset["func"]:
            messages = self.prompt_config.as_messages(
                func_code=func,
                phase=self.prompt_mode,
                mode=self.assumption_mode,
                add_hierarchy=self.add_hierarchy
            )
            batch.append(messages)

            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def _batched_inference(
        self, test_dataset: Dataset, batch_size: int, n_retries: int = 3
    ) -> tuple[list[ExpectedModelResponse], list[int]]:
        """
        !!! Unused and unmaintained
        Run batched inference with sequential fallback for failures.

        Strategy:
        1. Try batch generation ONCE (no retries at batch level)
        2. Validate each prediction
        3. Sequential retry (with n_retries) for failed samples only
        4. If entire batch fails, sequential fallback for whole batch

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset with 'func' field
        batch_size : int
            Number of samples per batch
        n_retries : int, default=3
            Number of retries for sequential fallback

        Returns
        -------
        tuple[list[str], list[int]]
            (predictions, success_indices)
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError(
                "Model and tokenizer must be loaded before running evaluation."
            )

        all_predictions: list[ExpectedModelResponse] = []
        success_indices: list[int] = []

        n_failures: int = 0
        n_ok: int = 0
        num_samples = len(test_dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size

        with rich_progress_manual(
            total=num_samples,
            description="Batched Inference",
            initial_status="Starting...",
        ) as pbar:
            sample_offset = 0

            for idx, batch_messages in enumerate(
                self._create_message_batches(test_dataset, batch_size=batch_size)
            ):
                batch_len = len(batch_messages)

                input_texts = self.tokenizer.apply_chat_template(
                    batch_messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.tokenizer(
                    input_texts,  # type: ignore
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_seq_length,
                    truncation=True,
                ).to(self.model.device)  # type: ignore

                try:
                    with self._tokenizer_config(), torch.inference_mode():
                        outputs = self.model.generate(  # type: ignore
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

                    # validation
                    batch_results: list[ExpectedModelResponse | None] = []
                    failed_indices: list[int] = []

                    for i, pred in enumerate(decoded_predictions):
                        pred = pred.strip()
                        try:
                            batch_results.append(ExpectedModelResponse.model_validate_json(json_data=pred))
                        except (
                            json.JSONDecodeError,
                            ValidationError,
                            MismatchCWEError,
                            EmptyReasoningError,
                        ):
                            batch_results.append(None)
                            failed_indices.append(i)

                    # sequential retry for failures only
                    if failed_indices:
                        logger.info(
                            f"Batch {idx+1}: {len(failed_indices)}/{batch_len} "
                            f"failed validation, retrying sequentially..."
                        )

                        with rich_progress_manual(
                            total=len(failed_indices),
                            description=f"Retrying {len(failed_indices)} failed samples sequentially",
                        ) as pbar:
                            for i in failed_indices:
                                sample_idx = sample_offset + i
                                sample = test_dataset[sample_idx]
                                try:
                                    seq_pred = self.run_inference(
                                        input_code=sample["func"], n_retries=n_retries
                                    )
                                    batch_results[i] = seq_pred

                                except GenerationError as e:
                                    logger.error(
                                        f"Sample {sample_idx} failed sequential retry: {e}"
                                    )
                                    # Leave as None

                                except Exception as e:
                                    logger.exception(
                                        f"Unexpected error on sample {sample_idx}: {e}"
                                    )
                                    # Leave as None

                    # Collect successful results
                    for i, result in enumerate(batch_results):
                        if result is not None:
                            all_predictions.append(result)
                            success_indices.append(sample_offset + i)

                except Exception as e:
                    # Entire batch failed - sequential fallback for whole batch
                    logger.error(
                        f"Batch {idx+1} generation failed: {e}. "
                        f"Using sequential fallback for entire batch."
                    )

                    for i in range(batch_len):
                        sample_idx = sample_offset + i
                        sample = test_dataset[sample_idx]

                        try:
                            seq_pred = self.run_inference(
                                input_code=sample["func"], n_retries=n_retries
                            )
                            all_predictions.append(seq_pred)
                            success_indices.append(sample_idx)

                        except GenerationError as e:
                            logger.error(f"Sample {sample_idx} failed: {e}")
                            # Don't add to predictions

                        except Exception as e:
                            logger.exception(
                                f"Unexpected error on sample {sample_idx}: {e}"
                            )
                            # Don't add to predictions
                finally:
                    pbar.update(advance=batch_len)
                    pbar.set_postfix(
                        {
                            "Batch": f"{idx+1}/{num_batches}",
                            "✓ Ok": n_ok,
                            "✗ Error": n_failures,
                            "% Error rate": (
                                f"{(n_failures/self.n_samples):.1%}"
                                if n_failures > 0
                                else "0%"
                            ),
                        }
                    )
                    sample_offset += batch_len

            pbar.set_description("✅ Batch inference complete")

        return all_predictions, success_indices

    def diagnose_model(self, test_dataset: Dataset, output_dir: str, n_samples: int = 5) -> None:
        """Diagnose model behavior on known-vulnerable samples."""
        # import yaml
        from datetime import datetime

        dir = Path(output_dir)
        dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = dir / f"diagnosis_{timestamp}.yaml"

        vuln_samples = [ex for ex in test_dataset if ex["target"] == 1][:n_samples]  # type: ignore[reportCallIssue, reportArgumentType]

        with open(file=output_path, mode="w") as f:
            for i, sample in rich_progress(
                enumerate(vuln_samples),
                total=len(vuln_samples),
                description="Diagnosing",
                status_fn=lambda x: f"Sample {x[0] + 1}/{len(vuln_samples)}",
            ):
                result = self.run_inference(sample["func"])  # type: ignore[reportCallIssue, reportArgumentType]
                pair = CWEPair(
                    cwes_gt=sample["cwe"],  # type: ignore[reportCallIssue, reportArgumentType]
                    cwes_pred=result.cwe_list,
                )

                record = {
                    "sample_id": i + 1,
                    "ground_truth_cwes": pair.cwes_gt,
                    "predicted_cwes": pair.cwes_pred,
                    "detected_vulnerable": result.is_vulnerable,
                    "strict_match": pair.is_strict_match,
                    "hierarchical_match": pair.is_hierarchical_match,
                    "reasoning": result.reasoning,
                    "verdict": {
                        "is_vulnerable": result.verdict.is_vulnerable,
                        "cwe_list": result.verdict.cwe_list,
                    },
                }

                dump_yaml(data=record, stream=f)
                f.write("---\n")

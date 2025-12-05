import json
from unsloth import FastLanguageModel, get_chat_template

import torch
import logging
import sys

from typing import Any, Literal,Union, Iterator, cast, overload
from pydantic import BaseModel, ValidationError
from contextlib import contextmanager
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer
from outlines import Generator, from_transformers
from outlines.types import JsonSchema
from genson import SchemaBuilder

from .datatypes import GenerationError, TypedDataset, TestDatasetSchema
from .prompt_config import Messages, VulnerabilityPromptConfig as PromptConfig
from ..utilities import (
    stateless_progress,
    rich_progress_manual,
    rich_print,
    rich_panel,
    rich_exception,
    build_table,
    RichColors
)

if sys.version_info >= (3, 12):
    from typing import _TypedDictMeta  # type: ignore
else:  # pragma: no cover
    from typing_extensions import _TypedDictMeta  # type ignore

ChatTemplate = str | list[int] | list[str] | list[list[int]] | BatchEncoding
ConvertedSchema = Union[str, dict, type[BaseModel], _TypedDictMeta, type, SchemaBuilder]
ModelDump = dict[str, Any]
MidTermInferenceBatch = list[ModelDump | None]
FinalInferenceBatch = list[ModelDump]


logger = logging.getLogger(__name__)


@dataclass
class BatchInferenceStats:
    """Statistics from batch inference run.

    Attributes:
    total_samples: int
        Total amount of samples
    successful_samples: int
        Number of samples successfully validated
    successful_sample_indices: list[int]
        List of indices of successfully validated samples. 
        This is used for dataset alignment
    failed_samples: int
        Number of problematic samples
    failed_sample_indices: list[int]
        List of failed indices.
        This is used for retry logic.
    """
    total_samples: int = 0
    successful_samples: int = 0
    successful_sample_indices: list[int] = field(default_factory=list)
    failed_samples: int = 0
    failed_sample_indices: list[int] = field(default_factory=list)

    @property
    def to_dict(self) -> dict[str,Any]:
        return {
            "Total samples": self.total_samples,
            "Successful": self.successful_samples,
            "Failed": self.failed_samples,
        }


class JSONGenerator:
    """Wrapper for Unsloth model with schema-enforced generation."""

    def __init__(
        self,
        model: FastLanguageModel,
        tokenizer: PreTrainedTokenizer,
        prompt_config: PromptConfig,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.2,
        top_p: float = 0.95,
        min_p: float = 0.1,
        repetition_penalty: float = 1.05,
    ) -> None:
        """
        Initialize analyzer.

        Parameters
        ----------
        model: FastLanguageModel
            Model instance previously loaded with [~FastLanguageModel.from_pretrained(...)]
        tokenizer: PreTrainedTokenizer
            Tokenizer instance previously loaded with [~FastLanguageModel.from_pretrained(...)]
        prompt_config : PromptConfig
            Your prompt configuration
        max_new_tokens : int
            Maximum tokens to generate
        do_sample : bool
        temperature : float
            Sampling temperature
        top_p : float
            Nucleus sampling parameter
        min_p : float
            Minimum probability threshold
        repetition_penalty : float
            Penalty for repeating tokens
        """

        self.tokenizer = tokenizer
        self.original_tokenizer_params = {
            "padding_side": self.tokenizer.padding_side,
            "pad_token": self.tokenizer.pad_token,
            "eos_token": self.tokenizer.eos_token
        }

        self.prompt_config = prompt_config

        # Store generation parameters
        self.generation_params = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        self.outlines_model = from_transformers(
            model=cast(PreTrainedModel, model), tokenizer_or_processor=tokenizer
        )

        # define output structure
        schema_dict = {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "vulnerabilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cwe_id": {"type": "integer"},
                            "description": {"type": "string"},
                        },
                        "required": ["cwe_id", "description"],
                    },
                },
                "verdict": {
                    "type": "object",
                    "properties": {
                        "is_vulnerable": {"type": "boolean"},
                        "cwe_list": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["is_vulnerable", "cwe_list"],
                },
            },
            "required": ["reasoning", "vulnerabilities", "verdict"],
        }

        self.json_schema = JsonSchema(
            schema_dict,
            whitespace_pattern=r"[ ]?",  # Allow compact JSON
        )
        self.pydantic_schema = self.convert_json_schema(target_type="pydantic")

        self.generator_json = Generator(
            model=self.outlines_model, output_type=self.json_schema
        )

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["pydantic"] = ...,
    ) -> type[BaseModel]: ...

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["typeddict"],
    ) -> _TypedDictMeta: ...

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["dict"],
    ) -> dict: ...

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["str"],
    ) -> str: ...

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["dataclass"],
    ) -> type: ...

    @overload
    def convert_json_schema(
        self,
        target_type: Literal["genson"],
    ) -> SchemaBuilder: ...

    def convert_json_schema(
        self,
        target_type: Literal[
            "str", "dict", "pydantic", "typeddict", "dataclass", "genson"
        ] = "pydantic",
    ) -> ConvertedSchema:
        return JsonSchema.convert_to(schema=self.json_schema, target_types=[target_type])

    def _format_prompt(self, c_code_input: str) -> ChatTemplate:
        """Format code input using proper chat template.

        Parameters
        ----------
        c_code_input: str
            Function implementation

        Returns
        -------
        ChatTemplate
            Formatted prompt structure from [~tokenizer.apply_chat_template()]
        """

        # build message structure
        messages: Messages = self.prompt_config.as_messages(func_code=c_code_input)

        # apply model-specific chat template (CRITICAL!)
        formatted_prompt: ChatTemplate = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return formatted_prompt

    @contextmanager
    def _tokenizer_config(self):
        try:
            self.tokenizer.padding_side = self.original_tokenizer_params["padding_side"]
            self.tokenizer.pad_token = self.original_tokenizer_params["pad_token"]

            if hasattr(self.outlines_model, "tokenizer"):
                self.outlines_model.tokenizer.tokenizer.padding_side = ( 
                    self.original_tokenizer_params["padding_side"]
                )

                self.outlines_model.tokenizer.pad_token = (
                    self.original_tokenizer_params["pad_token"]
                    or self.original_tokenizer_params["eos_token"]
                )

            yield
        finally:
            pass

    def run_inference(
        self, data: str | ChatTemplate, n_retries: int | None = None, **override_params
    ) -> ModelDump:
        """
        Perform inference on a single code sample.

        Parameters
        ----------
        c_code_input : str
            C function code to analyze
        n_retries : int | None
            Number of retry attempts remaining
        **override_params
            Override default generation parameters for this call

        Returns
        -------
        ModelDump
            JSON like dictionary
        """

        default_msg: str = "Generating enforced JSON output"
        if n_retries:
            default_msg = (
                f"Generating enforced JSON output (retries left = {n_retries})"
            )

        gen_params: dict[str, Any] = {**self.generation_params, **override_params}

        if isinstance(data, str):
            data = self._format_prompt(data)  # enforce ChatTemplate

        # generate with schema enforcement
        with (
            self._tokenizer_config(),
            torch.inference_mode(),
            stateless_progress(description="Running sequential inference") as status,
        ):
            try:
                status.update(default_msg)
                result_json = self.generator_json(data, **gen_params)
                validated: BaseModel = self.pydantic_schema.model_validate_json(result_json)  # type: ignore[reportArgumentError]
                return validated.model_dump(mode="json")

            except (torch.cuda.OutOfMemoryError, RuntimeError, ValidationError):
                if n_retries and n_retries > 0:
                    return self.run_inference(
                        data=data,
                        n_retries=n_retries - 1,
                        **override_params,
                    )

                # retries exhausted
                raise GenerationError(
                    f"Sequential inference failed: retries exhausted!"
                )

            except KeyboardInterrupt:
                # User interrupt - pass through
                raise

            except Exception:
                # unexpected error
                raise GenerationError(
                    f"Sequential inference failed: unexpected Exception!"
                )

    def run_sequential_inference_on_dataset(
        self,
        test_dataset: Dataset,
        n_retries: int = 3,
    ) -> tuple[FinalInferenceBatch, list[int]]:
        """
        Run sequential inference on dataset. Slower but more robust.
        """
        all_predictions: FinalInferenceBatch = []
        stats = BatchInferenceStats(total_samples=len(test_dataset["func"]))

        with rich_progress_manual(
            total=stats.total_samples,
            description="Sequential Inference",
        ) as pbar:
            for idx, func_code in enumerate(test_dataset["func"]):
                try:
                    # generate and validate
                    output: ModelDump = self.run_inference(data=func_code, n_retries=n_retries)

                except (GenerationError, ValidationError):
                    # oh-oh
                    stats.failed_samples += 1
                    stats.failed_sample_indices.append(idx)

                else:
                    # success
                    all_predictions.append(output)
                    stats.successful_samples += 1
                    stats.successful_sample_indices.append(idx)

                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix(
                        {
                            "✓": stats.successful_samples,
                            "✗": stats.failed_samples,
                            "% Error rate": (
                                f"{(stats.failed_samples/stats.total_samples):.1%}"
                                if stats.failed_samples > 0
                                else "0%"
                            ),
                        }
                    )

        # Final summary
        logger.info(
            f"\nBatch inference complete:\n"
            f"  Total: {stats.total_samples}\n"
            f"  Successful: {stats.successful_samples} "
            f"({stats.successful_samples/stats.total_samples:.1%})\n"
            f"  Failed: {stats.failed_samples} "
            f"({stats.failed_samples/stats.total_samples:.1%})"
        )

        rich_panel(
            tables=build_table(data=stats.to_dict, columns=["", "Value"]),
            panel_title="Batch inference report",
            border_style=RichColors.ROYAL_BLUE1,
        )
        return all_predictions, stats.successful_sample_indices

    def _create_batches(
        self, dataset: Dataset, *, batch_size: int, num_samples
    ) -> Iterator[list[str]]:
        """
        Generate batches of `func`s for inference.

        Parameters
        ----------
        dataset : Dataset
            Dataset to batch
        batch_size : int
            Size of each batch
        num_samples : int
            Total number of samples to process (len(dataset))

        Yields
        ------
        list[str]
            Batch of function strings
        """

        current_batch_idx = 0
        while current_batch_idx < num_samples:
            end_idx = min(current_batch_idx + batch_size, num_samples)  # boundary check
            batch: Dataset = dataset.select(range(current_batch_idx, end_idx))

            yield batch["func"]

            current_batch_idx = end_idx  # update batch pointer

    def run_batch_inference(
        self,
        input_batch: list[str],
        n_retries: int | None = None,
        **override_params,
    ) -> MidTermInferenceBatch:
        """
        Run batch inference with sequential fallback for failures.

        Parameters
        ----------
        input_batch : list[str]
            Batch of C function codes
        n_retries : int, optional
            Number of retries for failed samples
        **override_params
            Override generation parameters

        Returns
        -------
        list[ModelDump | None]
            Validated results (None for failures)
        """

        eventually_failed_idxs: int = 0
        gen_params: dict[str, Any] = {**self.generation_params, **override_params}
        formatted_batch: list[ChatTemplate] = [self._format_prompt(code) for code in input_batch]
        batch_size: int = len(input_batch)

        with self._tokenizer_config(), torch.inference_mode():
            try:  # batch generation
                raw_results: list[str] = self.generator_json.batch(formatted_batch, **gen_params)  # pyright: ignore[reportAssignmentType]

                # validate samples
                validated_results: list[dict[str, Any] | None] = []
                failed_indeces: list[int] = []

                for idx, raw_response in enumerate(raw_results):
                    try:
                        validated: BaseModel = self.pydantic_schema.model_validate_json(raw_response)
                        validated_results.append(validated.model_dump(mode="json"))
                    except (ValidationError, json.JSONDecodeError):
                        validated_results.append(None)
                        failed_indeces.append(idx)  # idx 4 fallback

                # sequential retry
                if failed_indeces:
                    with rich_progress_manual(
                        total=len(failed_indeces),
                        description=f"Retrying {len(failed_indeces)} failed samples sequentially",
                    ) as pbar:
                        for idx in failed_indeces:
                            try:
                                validated_response: ModelDump = self.run_inference(data=formatted_batch[idx], n_retries=n_retries)
                                validated_results[idx] = validated_response

                            except GenerationError:
                                eventually_failed_idxs +=1
                                rich_exception()
                                # leave None at this index for future filtering

                            finally:
                                pbar.update()
                                pbar.set_postfix({
                                    "[bold green]✓[/bold green] Recovered": (len(failed_indeces) - eventually_failed_idxs),
                                    "[bold red]✗[/bold red] Failed": eventually_failed_idxs,
                                })

                success_count = sum(1 for r in validated_results if r is not None)
                logger.info(f"Batch complete: {success_count}/{batch_size} succeeded")

                return validated_results

            except Exception as e:
                # Entire batch generation failed - fallback to full sequential
                logger.error(f"Batch generation failed entirely: {e}")
                logger.info(f"Falling back to sequential for entire batch")

                validated_results = []
                for finput in formatted_batch:
                    try:
                        validated_response = self.run_inference(data=finput, n_retries=n_retries)
                        # validated = self.pydantic_schema.model_validate_json(raw_response)
                        validated_results.append(validated_response)
                    except GenerationError:
                        rich_exception()
                        validated_results.append(None)

                return validated_results

    def run_batch_inference_on_dataset(
        self,
        test_dataset: Dataset,
        batch_size: int,
        n_retries: int = 3,
    ) -> tuple[FinalInferenceBatch, list[int]]:
        """
        Run batch inference on dataset with retries.

        Parameters
        ----------
        test_dataset : Dataset
            Dataset containing 'func' column with C code
        batch_size : int
            Number of samples per batch
        n_retries : int
            Number of retries per batch

        Returns
        -------
        tuple[list[dict | None], BatchInferenceStats]
            Predictions aligned with dataset (None for failures) and statistics
        """

        sample_offset: int = 0
        all_predictions: FinalInferenceBatch = []
        stats = BatchInferenceStats(total_samples=len(test_dataset["func"]))

        num_batches = (stats.total_samples + batch_size - 1) // batch_size
        logger.info(
            f"Starting batch inference: {stats.total_samples} samples, "
            f"{num_batches} batches of size {batch_size}"
            f"Progress bar will be displayed per batch!"
        )

        for func_batch in self._create_batches(
            test_dataset, batch_size=batch_size, num_samples=len(test_dataset)
        ):
            batch_len = len(func_batch)
            outputs: MidTermInferenceBatch = self.run_batch_inference(input_batch=func_batch, n_retries=n_retries)

            # filter None
            if all(res is not None for res in outputs):
                # if all good, no checks, just add
                all_predictions.extend(cast(FinalInferenceBatch, outputs))
                stats.successful_samples += batch_len
            else:
                for local_idx, res in enumerate(outputs):
                    if res is None:
                        stats.failed_samples += 1
                        stats.failed_sample_indices.append(sample_offset + local_idx)
                    else:
                        all_predictions.append(res)
                        stats.successful_samples += 1
                        stats.successful_sample_indices.append(sample_offset + local_idx)

            sample_offset += batch_len

        # Final summary
        logger.info(
            f"\nBatch inference complete:\n"
            f"  Total: {stats.total_samples}\n"
            f"  Successful: {stats.successful_samples} "
            f"({stats.successful_samples/stats.total_samples:.1%})\n"
            f"  Failed: {stats.failed_samples} "
            f"({stats.failed_samples/stats.total_samples:.1%})"
        )

        rich_panel(
            tables=build_table(data=stats.to_dict, columns=["", "Value"]),
            panel_title="Batch inference report",
            border_style=RichColors.ROYAL_BLUE1,
        )

        return all_predictions, stats.successful_sample_indices

    def evaluate_test_set(
        self,
        test_dataset: Dataset,
        use_batching: bool,
        batch_size: int = 8,
    ) -> TypedDataset[TestDatasetSchema]:
        """Run inference on test dataset and return predictions.

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset with 'func' field containing code samples.
        use_batching: bool
            Select either `batch` or `sequential` modes.
        batch_size : int, default=16
            Batch size for inference.

        Returns
        -------
        Dataset
            - Dataset with added 'model_prediction' column
        """

        self.n_samples: int = len(test_dataset)

        if not self.n_samples > 0:
            raise ValueError("Provided dataset is empty")

        logger.info(
            f"🔍 Evaluating on {self.n_samples} test samples "
            f"(mode={'batch' if use_batching else 'sequential'}, "
            f"batch_size={batch_size if use_batching else 'N/A'})..."
        )

        if use_batching:
            predictions, success_indices = self.run_batch_inference_on_dataset(
                test_dataset=test_dataset, batch_size=batch_size, n_retries=3
            )
        else:
            predictions, success_indices = self.run_sequential_inference_on_dataset(
                test_dataset=test_dataset, n_retries=3
            )

        if not predictions:
            raise RuntimeError("All samples failed inference!")

        # add predictions to dataset
        aligned_dataset = test_dataset.select(success_indices)
        results_dataset = aligned_dataset.add_column(  # type: ignore[reportCallIssue]
            name="model_prediction", column=predictions
        )

        return TypedDataset[TestDatasetSchema](results_dataset)

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.padding_side = "left"
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    analyzer = JSONGenerator(
        model=model,
        tokenizer=tokenizer,
        prompt_config=PromptConfig(),
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.95,
    )

    code_sample = """void foo(char *input) {
    char buf[64];
    strcpy(buf, input);
}"""
    code1 = code_sample
    code2 = """void foo(char *input) {
    char buf[64];
}"""

    code3 = (
        "int PHP_METHOD(Phar, running) {\n"
        "  char *fname, *arch, *entry;\n"
        "  int fname_len, arch_len, entry_len;\n" "  zend_bool retphar = 1;\n"
        "  if(zend_parse_parameters(ZEND_NUM_ARGS(), TSRMLS_CC, \"|b\", &retphar) == FAILURE) { return; }\n"
        "  fname = (char *)zend_get_executed_filename(TSRMLS_C);\n"
        "  fname_len = strlen(fname);\n"
        "  if(fname_len > 7 && !memcmp(fname, \"phar://\", 7) && SUCCESS == \n"
        "  phar_split_fname(fname, fname_len, &arch, &arch_len, &entry, &entry_len, 2, 0, TSRMLS_CC)) {\n"
        "    efree(entry);\n"
        "    if(retphar) {\n"
        "      RETVAL_STRINGL(fname, arch_len + 7, 1);\n"
        "      efree(arch);\n"
        "      return;\n"
        "    } else {\n"
        "      RETURN_STRINGL(arch, arch_len, 0);\n"
        "    }\n"
        "  }\n"
        "  RETURN_STRINGL(\"\", 0, 1);\n"
        "}"
    )

    result_json = analyzer.run_inference(code3)
    print("=" * 80)
    print("JSON")
    if result_json: 
        rich_print(result_json,is_json=True)

    batch = [code1, code2, code3]
    with stateless_progress("Running batch inference") as status:
        batch_results = analyzer.run_batch_inference(batch)
        status.stop()

    if batch_results:
        for o in batch_results:
            rich_print(o, is_json=True)

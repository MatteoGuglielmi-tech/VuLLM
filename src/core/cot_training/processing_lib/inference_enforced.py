from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import json
import torch
import logging

from typing import Literal, overload
from pathlib import Path
from dataclasses import dataclass, field

from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from .prompt_config import VulnerabilityPromptConfig
from .datatypes import TestDatasetSchema, TypedDataset
from .schema_generation import JSONGenerator
from ..utilities import RichColors, is_main_process, build_table, rich_panel

logger = logging.getLogger(name=__name__)


@dataclass
class TestHandler:
    lora_model_dir: Path|str
    evaluated_testset_path: Path
    max_seq_length: int
    max_new_tokens: int
    prompt_mode: str
    chat_template: str|None = None

    model: FastLanguageModel|None = field(init=False, default=None, repr=False)
    tokenizer: PreTrainedTokenizer|None = field(init=False, default=None, repr=False)

    prompt_config = VulnerabilityPromptConfig()
    _counter_fails: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        self._validate_inputs()

        d_table = {
            "LoRA checkpoint:": self.lora_model_dir,
            "Max sequence length": f"{self.max_seq_length:,}",
            "Max new tokens per answer": f"{self.max_new_tokens:,}",
            "Custom chat template": self.chat_template is not None,
        }

        tb = build_table(data=d_table, show_header=False)
        rich_panel(
            tb,
            panel_title=f"🔧 Initializing TestHandler with LoRA checkpoint: {self.lora_model_dir}",
            border_style=RichColors.MEDIUM_PURPLE1,
        )

        self._load_finetuned_model()
        self.json_generator = JSONGenerator(
            model=self.model, # type: ignore
            tokenizer=self.tokenizer, # type: ignore
            prompt_config=self.prompt_config,
            prompt_mode=self.prompt_mode,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            min_p=0.1,
            repetition_penalty=1.05,
        )

    def _validate_inputs(self):
        """Validate constructor inputs."""

        if isinstance(self.lora_model_dir, str):
            self.lora_model_dir = Path(self.lora_model_dir)

        if self.prompt_mode in ["training", "training_no_assumptions"]:
            raise ValueError(f"THIS IS INFERENCE! Provided {self.prompt_mode} prompt mode!")

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

        if self.model is None or self.tokenizer is None: # to silence diagnostics
            raise RuntimeError("Model or tokenizer not loaded")

        decoder_only_models: set[str] = { "llama", "mistral", "qwen", "opt", "phi", "gemma"}
        model_type = getattr(self.model.config, "model_type", "").lower() # type: ignore

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

    def evaluate_on_test_set(
        self,
        test_dataset: Dataset,
        batch_size: int,
        use_batching: bool = True,
    ) -> TypedDataset[TestDatasetSchema]:
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
            f"(mode={'batch' if use_batching else 'sequential'}, "
            f"batch_size={batch_size if use_batching else 'N/A'})..."
        )

        dataset_with_evaluations: TypedDataset[TestDatasetSchema] = self.json_generator.evaluate_test_set(
            test_dataset=test_dataset, batch_size=batch_size, use_batching=use_batching
        )

        self.save_evaluation_results(results_dataset=dataset_with_evaluations.raw)

        return dataset_with_evaluations

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
        from datetime import datetime

        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")

        prefix = self.evaluated_testset_path / date / time
        huggingface: Path = prefix / "huggingface"
        jsonl: Path = prefix / "json"
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

        test_data: Dataset = load_dataset_from_disk(input_dir=input_dir, split_name=split_name)
        return TypedDataset[TestDatasetSchema](test_data) if with_eval else test_data

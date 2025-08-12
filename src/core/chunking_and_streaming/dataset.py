import gc
import random
import shutil
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict

from datasets import IterableDataset, load_dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from .shared.utils import load_json_config, save_to_jsonl, load_from_jsonl
from .shared.tree_sitter_parser import TreeSitterParser
from .shared.typedef import DatasetEntry, StreamedSplit, FinalChunk, JsonlData
from .prompt_strategies import PromptingStrategy, GenericStrategy
from .chunking import generate_code_chunks

import logging
from .shared.stdout import MY_LOGGER_NAME
logger = logging.getLogger(MY_LOGGER_NAME)

@dataclass
class DatasetHandler:
    pth_raw_dataset: Path
    tokenizer: PreTrainedTokenizer
    max_chunk_tokens: int = 1024  # as per default for SFTTrainer from HF
    trimming_technique: str = "ast"

    # fields
    raw_dataset: JsonlData = field(default_factory=list, init=False)
    prompting_strategy: PromptingStrategy = field(default_factory=GenericStrategy)
    chunked_data: dict[str,IterableDataset]|None = field(default=None)
    tokenized_data: StreamedSplit|None = field(default=None)


    def __post_init__(self):
        """Initializes prompt skeletons and configuration paths."""

        self.SYSTEM_PROMPT = (
            "You are a cyber-security and vulnerability expert.\n"
            "Your task is to determine whether the provided piece of code contains a security vulnerability.\n"
            "NOTE: input code may be chunked thus may not be a valid."
            "This is intended, merging by function signature (accounting for overlap) leads to a valid, complete code."
        )

        self.USER_PROMPT_SKELETON = (
            "Define whether the following code is vulnerable.\n"
            "**Output Requirements:**\n"
            "- answer 'YES' if vulnerable or 'NO' otherwise.\n"
            "- **DO NOT** use any special non-ASCII characters.\n"
            "- **DO NOT** use any special formatting or markdown.\n"
            "--- SAMPLE START ---\n"
            "Function signature:\n{signature}\n"
            "Function description:\n{func_descr}\n"
            "Code Fragment:\n{subchunk}\n"
            "--- SAMPLE END ---\n"
            "Correct answer:\n{ground_truth}\n"
            "CWE description:\n{cwe_descr}"
        ).strip()

        self.INFERENCE_USER_PROMPT_SKELETON = (
            "Define whether the following code is vulnerable.\n"
            "**Output Requirements:**\n"
            "- answer 'YES' if vulnerable or 'NO' otherwise.\n"
            "- **DO NOT** use any special non-ASCII characters.\n"
            "- **DO NOT** use any special formatting or markdown.\n"
            "--- SAMPLE START ---\n"
            "Function signature:\n{signature}\n"
            "Function description:\n{func_descr}\n"
            "Code Fragment:\n{subchunk}\n"
            "--- SAMPLE END ---"
        ).strip()

        # path handling
        script_dir = Path(__file__).parent
        project_root = script_dir.parents[2]
        config_path = project_root / "config.json"
        config = load_json_config(filepath=str(config_path))

        # paths for saving chunked data
        self.pth_intermediate_train: Path = Path(config["pth_intermediate_train"])
        self.pth_intermediate_val: Path = Path(config["pth_intermediate_val"])
        self.pth_intermediate_test: Path = Path(config["pth_intermediate_test"])
        self.pth_final_train: Path = Path(config["pth_final_train"])
        self.pth_final_val: Path = Path(config["pth_final_val"])
        self.pth_final_test: Path = Path(config["pth_final_test"])

    def load_raw_dataset(self) -> None:
        """Loads the entire raw dataset into memory from a JSONL file.

        This is a preliminary step required to perform a project-based
        train/validation/test split before chunking the data.
        """

        logger.info(f"Loading entire raw dataset from {self.pth_raw_dataset}...")
        try:
            self.raw_dataset: JsonlData = load_from_jsonl(filepath=self.pth_raw_dataset)
            logger.info(f"Successfully loaded {len(self.raw_dataset)} entries.")
        except (IOError, ValueError) as e:
            logger.error(f"An error occurred while loading the dataset: {e}")
            raise

    def _group_entries_by_project(self) -> defaultdict[str, JsonlData]:
        """Groups dataset entries from `self.raw_dataset` by their 'project' field.

        Returns
        -------
            defaultdict[str, list[dict[str, Any]]]
                A dictionary where keys are project names and values are lists of
                dataset entries (without the 'project' key).
        """

        if not self.raw_dataset:
            logger.warning("Attempted to group an empty raw dataset.")
            return defaultdict(list)

        grouped_entries: defaultdict[str, JsonlData] = defaultdict(list)
        for entry in self.raw_dataset:
            try:
                current_project: str = entry["project"]
                new_entry: DatasetEntry = { key: value for key, value in entry.items() if key != "project" }
                grouped_entries[current_project].append(new_entry)
            except KeyError:
                logger.error("Dataset entry is missing the 'project' key. Skipping entry.")
                continue

        logger.info(f"Grouped {len(self.raw_dataset)} entries into {len(grouped_entries)} projects.")

        return grouped_entries

    def project_based_split(self) -> None:
        """Performs a project-based train/validation/test split and saves
        the resulting datasets to JSONL files.

        This split approach ensures all entries from a single project are
        contained within a single split, avoiding data leakage.

        Note: if split files are already present, step is skippet to save evaluation time.
        """

        # check if splits are already present
        if all(p.exists() for p in [self.pth_final_train, self.pth_final_val, self.pth_final_test]):
            return

        # ensure the raw dataset is loaded before proceeding
        if not self.raw_dataset: self.load_raw_dataset()

        project_groups: defaultdict[str, JsonlData] = self._group_entries_by_project()
        project_names: list[str] = list(project_groups.keys())
        random.shuffle(project_names) # randomly shuffle projects

        nb_projects: int = len(project_names)
        train_projects: list[str] = project_names[:int(0.8 * nb_projects)] # ~80%
        val_projects: list[str] = project_names[int(0.8 * nb_projects):int(0.9 * nb_projects)] # ~10%
        test_projects: list[str] = project_names[int(0.9 * nb_projects):] # ~10%

        train_data: JsonlData = [entry for proj in train_projects for entry in project_groups[proj]]
        val_data: JsonlData = [entry for proj in val_projects for entry in project_groups[proj]]
        test_data: JsonlData = [entry for proj in test_projects for entry in project_groups[proj]]

        # ensure the output directory exists
        output_dir = self.pth_intermediate_train.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # serialize splits to intermediate files
        save_to_jsonl(dataset=train_data, filepath=self.pth_intermediate_train)
        save_to_jsonl(dataset=val_data, filepath=self.pth_intermediate_val)
        save_to_jsonl(dataset=test_data, filepath=self.pth_intermediate_test)

        logger.info(msg="💾 Project-based split JSONL data saved. 💾")
        logger.info(
            msg=f"Train samples: {len(train_data)/nb_projects * 100} %"
            f"Val samples: {len(val_data)/nb_projects * 100} %"
            f"Test samples: {len(test_data)/nb_projects * 100} %"
        )

    def _save_iterable_dataset_to_jsonl(self, dataset: IterableDataset, filepath: Path):
        """Saves an IterableDataset to a JSONL file.

        Parameters
        ----------
        dataset: IterableDataset
            The dataset to save.
        filepath: Path
            The Path object to the output file.
        """

        output_dir = filepath.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        save_to_jsonl(dataset=dataset, filepath=filepath)

    def _load_all_splits_streaming(self, split_paths: dict[str, Path]) -> dict[str,IterableDataset]:
        """Loads all dataset splits in streaming mode from a single call.

        Parameters
        ----------
        split_paths: dict[str, Path]
            A dictionary mapping split names to their file paths.

        Returns
        -------
        StreamingDataset
            An IterableDatasetDict containing the loaded splits.
        """

        logger.info("Loading all dataset splits in streaming mode from a dictionary of paths...")

        if not all(path.exists() for path in split_paths.values()):
            missing_paths = [str(path) for path in split_paths.values() if not path.exists()]
            logger.error(f"One or more split files not found: {missing_paths}")
            raise FileNotFoundError(f"Missing split files: {missing_paths}")

        return load_dataset("json", data_files={name: str(path) for name, path in split_paths.items()}, streaming=True) # type:ignore

    def _get_chunking_token_budget(self, manual_cap: int|None = None) -> int:
        """Calculates the maximum number of usable tokens for chunking,
        reserving space for special tokenizer characters.

        Parameters
        ----------
        manual_cap: int|None
            An optional manual override for the maximum chunk token length.

        Returns
        -------
        int
            The maximum number of tokens that can be used for a chunk.
        """

        ROOM_FOR_RESERVED_TOKEN: int = 32 # how many tokens to leave free for special chars added by tokenizer
        model_limit: int = self.max_chunk_tokens # start with maximu model seq_length

        if manual_cap and (manual_cap > ROOM_FOR_RESERVED_TOKEN): model_limit = manual_cap

        usable_tokens: int = int(model_limit - ROOM_FOR_RESERVED_TOKEN)

        logger.info(f"📏 Model max length for chunking: {model_limit}")
        logger.info(f"🧮 Using max_tokens={usable_tokens} for chunking")

        return usable_tokens

    def _chunk_and_format_code(self, example: dict[str, Any], max_tokens: int) -> dict[str, list[str]]:
        """Chunks a code example and formats each chunk into a model-ready prompt.

        This method takes a single data example containing a function's source code
        and a target label and runs a chunking pipeline on a budget. 
        It, then, formats each chunk into a full prompt string using predefined prompt skeletons
        and strategies. The final output is a dictionary suitable for batch processing, where the "text"
        key maps to a list of all generated prompts for the example.

        Parameters
        ----------
        example : dict[str, Any]
            A dictionary representing a single data example. It must contain the
            keys 'func' and 'target'.
        max_tokens : int
            The maximum number of tokens allowed for each code chunk. This is used
            to guide the chunking process.

        Returns
        -------
        dict[str, list[str]]
            A dictionary with a single key, "text", which contains a list of
            fully formatted prompt strings, one for each chunk generated from the
            input `example`.
        """

        chunk_prompts: list[str] = []
        label_str: str = str(example["target"])
        answer: str = "YES" if label_str == "1" else "NO"

        # Perform chunking
        chunks: list[FinalChunk] = generate_code_chunks(
            code=example["func"],
            tsp=TreeSitterParser(language_name=example["language"]),
            label=label_str,
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            trimming_technique=self.trimming_technique,
        )

        # format chunks into prompt in "text" column
        for chunk in chunks:
            # populate user prompt with chunk data
            user_prompt = self.USER_PROMPT_SKELETON.format(
                signature=chunk["function_signature"],
                func_descr=example["function_description"],
                subchunk=chunk["text"].strip(),
                ground_truth=answer,
                cwe_descr=example["vulnerability_description"]
            )
            # create final prompt for selected model
            full_prompt: str = self.prompting_strategy.format(system_prompt=self.SYSTEM_PROMPT, user_prompt=user_prompt)
            # append to formatted prompts
            chunk_prompts.append(full_prompt)

        # important: with `streaming` and `batched=False`, a batching function needs to return a dict[str, list[Any]]
        return {"text": chunk_prompts}

    def chunk(self):
        """Loads project-based splits in streaming mode, applies chunking, and
        formats each chunk into a prompt.

        If chunked data already exists on disk, it loads it instead.
        """

        final_chunked_paths: dict[str, Path] = { "train": self.pth_final_train, "val": self.pth_final_val, "test": self.pth_final_test }

        # do not chunk if already done once
        if all(p.exists() for p in final_chunked_paths.values()):
            logger.info("✅ Final chunked data already exists. Loading directly from disk.")
            self.chunked_data = self._load_all_splits_streaming(split_paths=final_chunked_paths)
            return

        # 2. load the intermediate project-based splits and process them.
        logger.info("Final chunked data not found. Starting the chunking process...")
        project_split_paths = { "train": self.pth_intermediate_train, "val": self.pth_intermediate_val, "test": self.pth_intermediate_test }
        stream_data: dict[str,IterableDataset] = self._load_all_splits_streaming(split_paths=project_split_paths)
        max_tokens: int = self._get_chunking_token_budget()

        self.chunked_data = {
            split: stream_data[split].map(
                self._chunk_and_format_code, batched=False,
                remove_columns=[ "func", "target", "cwe", "size", "commit_id", "message", "language", "parameter_count",
                    "cyclomatic_complexity", "function_description", "vulnerability_description",
                ],
                fn_kwargs={"max_tokens": max_tokens},
            ) for split in ["train", "val", "test"]
        }

        logger.info(msg="🔪 Data succesfully chunked and formatted into prompts. 🔪")

        # <---- serialize splits ---->
        for split_name, dataset in self.chunked_data.items():
            self._save_iterable_dataset_to_jsonl(dataset=dataset, filepath=final_chunked_paths[split_name])
        logger.info("💾 Final chunked data saved to disk.")

        del stream_data
        gc.collect()

    def _tokenize_fn(self, example: dict[str, str]) -> BatchEncoding:
        """Tokenizes a single text example using the class's tokenizer.

        This function is intended to be used with a `map` operation on a dataset.
        It takes a dictionary containing a "text" field, tokenizes the text,
        and returns the resulting `BatchEncoding` which includes `input_ids` and
        `attention_mask`.

        Parameters
        ----------
        example : dict[str, str]
            A dictionary-like object containing the data for one example.
            Must have a "text" key with the string to be tokenized.

        Returns
        -------
        transformers.tokenization_utils_base.BatchEncoding
            An object containing the tokenized `input_ids` and `attention_mask`.
        """

        tokenized: BatchEncoding = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_attention_mask=True,
        )
        return tokenized

    def get_processed_data(self) -> StreamedSplit:
        """Applies tokenization to all splits of the chunked dataset.

        This method iterates through the 'train', 'val', and 'test' splits of
        the `self.chunked_data`, applying the `_tokenize_fn` to each example.
        It removes the original "text" column from the dataset after tokenization
        to save memory. Finally, it deletes the `self.chunked_data` attribute
        and triggers garbage collection to free up resources.

        Returns
        -------
        StreamedSplit
            A dictionary where keys are data splits ('train', 'val', 'test')
            and values are the corresponding tokenized datasets.

        Raises
        ------
        RuntimeError
            If `self.chunked_data` is `None`, indicating that the preceding
            data chunking step has not been completed.
        """

        if self.chunked_data is None:
            raise RuntimeError("Chunked data is not available. Please run DATASET_chunk() first.")

        self.tokenized_data = {
            split: self.chunked_data[split].map(
                self._tokenize_fn,
                batched=False,  # Process one example (one prompt string) at a time
                remove_columns=["text"], # no more need for text, input_ids and attention_mask counts now
            )
            for split in ["train", "val", "test"]
        }

        logger.info("⚙️ Data successfully tokenized for all splits. ⚙️")

        # clean up chunked_data to free memory
        del self.chunked_data
        gc.collect()

        return self.tokenized_data

    def cleanup_intermediate_files(self):
        """Safely removes the intermediate project-split files after processing."""
        logger.info("🧹 Cleaning up intermediate files... 🧹")
        intermediate_dir = self.pth_intermediate_train.parent
        if intermediate_dir.exists():
            try:
                shutil.rmtree(intermediate_dir)
                logger.info(f"Successfully removed intermediate directory: {intermediate_dir}")
            except OSError as e:
                logger.error(f"Error removing intermediate directory {intermediate_dir}: {e}")



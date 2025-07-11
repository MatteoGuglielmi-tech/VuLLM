import gc
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from . import utils
from .chunking import extract_structured_chunks_with_context
from .stdout import logger
from .typedef import StreamedSplit, StreamingDataset, Windows


@dataclass
class DatasetHandler:
    # <---- constructor ---->
    pth_raw_dataset: str
    pth_inline_dataset: str
    tokenizer: PreTrainedTokenizer
    max_chunk_tokens: int = 1024  # as per default for SFTTrainer from HF
    trimming_technique: str = "ast"

    # <---- post init to build prompt skeletons ---->
    def __post_init__(self) -> None:
        self.SHORT_PROMPT_SKELETON = (
            "You are an AI system that analyzes C code for vulnerabilities.\n\n"
            + "Given the following code fragment, determine whether it contains a security vulnerability.\n"
            + "Code is chunked; reassemble by function signature.\n"
            + "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
            + "Function signature:\n{signature}\n\n"
            + "Code Fragment:\n{subchunk}\n\n"
            + "Answer 'YES' if vulnerable, 'NO' otherwise.\n"
            + "Correct answer:\n{ground_truth}"
        ).strip()

        self.PROMPT_SKELETON = (
            "You are an AI system that analyzes C code for vulnerabilities.\n\n"
            + "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
            + "KEY:Code is chunked; reassemble by function signature to obtain full original source code.\n"
            + "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
            + "Function signature:\n{signature}\n\n"
            + "Code Fragment:\n{subchunk}\n\n"
            + "Answer with 'YES' if vulnerable, 'NO' otherwise.\n"
            + "**IMPORTANT**: Strictly respond with only 'YES' or 'NO'.\n\n"
            + "Correct answer:\n{ground_truth}"
        ).strip()

        self.INFERENCE_PROMPT_SKELETON = (
            "You are an AI system that analyzes C code for vulnerabilities.\n\n"
            + "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
            + "KEY:Code is chunked; reassemble by function signature to obtain full original source code.\n"
            + "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
            + "Function signature:\n{signature}\n\n"
            + "Code Fragment:\n{subchunk}\n\n"
            + "Answer 'YES' if vulnerable, 'NO' otherwise."
        ).strip()

        # initialize vars
        self.chunked_data: Optional[StreamedSplit] = None
        self.tokenized_data: Optional[StreamedSplit] = None

        # Paths for saving chunked data
        self.pth_chunked_train = "./data/chunked_train_data.jsonl"
        self.pth_chunked_val = "./data/chunked_val_data.jsonl"
        self.pth_chunked_test = "./data/chunked_test_data.jsonl"

    # <---- load in RAM original preprocessed dataset from disk ---->
    def DATASET_load_raw_dataset(self) -> None:
        """Loads the orignal raw datset into RAM."""

        self.raw_dataset: dict[str, dict[str, str]] = utils.JSON_read_in(
            fp=self.pth_raw_dataset
        )

        logger.info(f"Loaded raw dataset from {self.pth_raw_dataset}")

    # <---- convert dataset to convenient jsonl format for efficiency ---->
    def DATASET_flatten(self) -> None:
        """HuggingFace dataset APIs accept in-line json as input.

        Flatten original JSON and save it as JSONL.
        """

        # ensure the directory exists
        output_dir = os.path.dirname(p=self.pth_inline_dataset)
        if output_dir and not os.path.exists(path=output_dir):
            os.makedirs(name=output_dir)

        # optional: I'm sure to pass jsonl file
        # utils.assert_file_extensions(
        #     filepaths=self.pth_inline_dataset, target_extensions=".jsonl"
        # )

        flattened: list[dict[str, str]] = list(self.raw_dataset.values())
        utils.JSON_serialize(obj=flattened, pth=self.pth_inline_dataset)
        logger.info(f"Flattened dataset saved to {self.pth_inline_dataset}")

    # <---- group dataset entries by group: preparation step for project-based splitting ---->
    def DATASET_group_entries_by_project(
        self,
    ) -> defaultdict[str, list[dict[str, str]]]:
        """Groups dataset entries by their 'project' field."""

        grouped_entries: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for entry in self.raw_dataset.values():
            current_prj: str = entry["project"]
            # create copy to avoid modifying original data
            entry_copy: dict[str, str] = entry.copy()
            del entry_copy["project"]  # remove "project" field from copy
            grouped_entries[current_prj].append(entry_copy)

        logger.info(
            f"Grouped {len(self.raw_dataset)} entries by {len(grouped_entries)} projects."
        )

        return grouped_entries

    # <---- perform project-based splitting ---->
    def DATASET_project_based_split(self) -> None:
        """Performs project-based splitting and saves results as JSONL
        files."""

        train_path: str = "./data/train_data.jsonl"
        val_path: str = "./data/val_data.jsonl"
        test_path: str = "./data/test_data.jsonl"

        # check if all split files already exist
        if (
            os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            logger.info(
                msg="Project-based split JSONL files already exist. Skipping split."
            )
            return

        project_groups: defaultdict[str, list[dict[str, str]]] = (
            self.DATASET_group_entries_by_project()
        )
        project_names: list[str] = list(project_groups.keys())
        # randomly shuffle projects
        random.shuffle(project_names)

        nb_projects: int = len(project_names)
        train_projects = project_names[: int(0.8 * nb_projects)]
        val_projects = project_names[int(0.8 * nb_projects) : int(0.9 * nb_projects)]
        test_projects = project_names[int(0.9 * nb_projects) :]

        train_data: list[dict[str, str]] = []
        val_data: list[dict[str, str]] = []
        test_data: list[dict[str, str]] = []

        for project, entries in project_groups.items():
            if project in train_projects:
                train_data.extend(entries)
            elif project in val_projects:
                val_data.extend(entries)
            elif project in test_projects:
                test_data.extend(entries)

        # ensure 'data' directory exists
        if not os.path.exists("./data"):
            os.makedirs(name="./data")

        # <---- serialize splits ---->
        utils.JSON_serialize(obj=train_data, pth="./data/train_data.jsonl")
        utils.JSON_serialize(obj=val_data, pth="./data/val_data.jsonl")
        utils.JSON_serialize(obj=test_data, pth="./data/test_data.jsonl")

        logger.info(msg="💾 Project-based split JSONL data saved. 💾")
        logger.info(
            msg=f"Train samples: {len(train_data)}"
            f"Val samples: {len(val_data)}"
            f"Test samples: {len(test_data)}"
        )

        # return {"train": train_data, "val": val_data, "test": test_data}

    # <---- load dataset from disk in streaming mode ---->
    def _DATASET_load_in_streaming_mode(
        self, filetype: str = "json", data_files: str = "./data/train_data.jsonl"
    ) -> StreamingDataset:
        """Loads a dataset split from disk in streaming mode."""

        return load_dataset(
            filetype, data_files=data_files, split="train", streaming=True
        )

    # <---- helper to save IterableDataset to JSONL ---->
    def _save_iterable_dataset_to_jsonl(
        self, dataset: StreamingDataset, filepath: str
    ) -> None:
        """Saves an IterableDataset to a JSONL file."""

        output_dir: str = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(name=output_dir)

        with open(file=filepath, mode="w", encoding="utf-8") as f:
            for example in dataset:
                f.write(json.dumps(obj=example) + "\n")

        # logger.info(f"Saved chunked data to {filepath}")

    # <---- load all splits in streaming mode ---->
    def DATASET_load_splits_in_streaming_mode(self, paths: list[str]) -> StreamedSplit:
        """Loads all dataset splits (train, val, test) in streaming mode."""

        return {
            k: self._DATASET_load_in_streaming_mode(data_files=fp)
            for fp, k in zip(paths, ["train", "val", "test"])
        }

    def _get_max_tokens_and_step(self, manual_cap: int | None = None) -> int:
        # how many tokens to leave free for special chars added by tokenizer
        ROOM_FOR_RESERVED_TOKEN: int = 32
        # model_limit = self.tokenizer.model_max_length if not manual_cap else manual_cap
        model_limit = (
            self.max_chunk_tokens
            if not manual_cap or manual_cap < ROOM_FOR_RESERVED_TOKEN
            else manual_cap
        )
        usable = int(model_limit - ROOM_FOR_RESERVED_TOKEN)
        logger.info(f"📏 Model max length for chunking: {model_limit}")
        logger.info(f"🧮 Using max_tokens={usable} for chunking")
        return usable

    # <---- perform chunking of overflowing functions + apply prompt skeleton ---->
    def _chunk_and_format_code(
        self,
        example: dict[str, str],  # or batch: dict[str, list[str]]
        max_tokens: int,
    ) -> dict[str, list[str]]:
        """Chunks a single function example and formats each chunk into a
        prompt.

        Returns a list of formatted prompt dictionaries.
        """

        chunk_prompts: list[str] = []
        label_str: str = str(example["target"])

        # Perform chunking
        chunks: Windows = extract_structured_chunks_with_context(
            code=example["func"],
            label=label_str,
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            trimming_technique=self.trimming_technique,
        )

        # important: when streaming, a batching function needs to return a dict[str, list[Any]] and
        # batched=False

        for chunk in chunks:
            inner_context = chunk.get("inner_context", "").strip()
            text = chunk.get("text", "").strip()
            trailing_context = chunk.get("trailing_context", "").strip()

            # combine them, ensuring newlines for readability if all components exist
            subchunk_parts: list[str] = []
            if inner_context:
                subchunk_parts.append(inner_context)
            if text:
                subchunk_parts.append(text)
            if trailing_context:
                subchunk_parts.append(trailing_context)

            subchunk: str = "\n".join(subchunk_parts).strip()

            # enforcing prompt for each chunk extracted
            answer: str = "YES" if label_str == "1" else "NO"

            full_prompt: str = self.PROMPT_SKELETON.format(
                subchunk=subchunk,
                signature=chunk["function_signature"].strip(),
                ground_truth=answer,
            )

            chunk_prompts.append(full_prompt)

        return {"text": chunk_prompts}

    # <---- wrapper function for loading in streaming mode the project based splits and perform chunking + enforcing prompt skeleton ---->
    def DATASET_chunk(self) -> None:
        """Loads project-based splits in streaming mode, applies chunking, and
        formats each chunk into a prompt.

        If chunked data already exists on disk, it loads it instead.
        """

        chunked_paths = {
            "train": self.pth_chunked_train,
            "val": self.pth_chunked_val,
            "test": self.pth_chunked_test,
        }

        stream_data: StreamedSplit = self.DATASET_load_splits_in_streaming_mode(
            paths=[
                "./data/train_data.jsonl",
                "./data/val_data.jsonl",
                "./data/test_data.jsonl",
            ]
        )

        max_tokens: int = self._get_max_tokens_and_step()

        self.chunked_data = {
            split: stream_data[split].map(
                self._chunk_and_format_code,
                batched=False,  # process one example at a time, makes sense with streaming
                remove_columns=["func", "target", "cwe", "project"],
                fn_kwargs={"max_tokens": max_tokens},
            )
            for split in ["train", "val", "test"]
        }

        logger.info(msg="🔪 Data succesfully chunked and formatted into prompts. 🔪")

        # <---- serialize splits ---->
        # Save the newly chunked data to disk
        for split_name, dataset in self.chunked_data.items():
            self._save_iterable_dataset_to_jsonl(
                dataset=dataset, filepath=chunked_paths[split_name]
            )

        logger.info(msg="💾 Chunked data saved to disk. 💾")

        del stream_data
        gc.collect()

    # <---- perform tokenization ---->
    def DATASET_tokenize_fn(self, example: dict[str, str]) -> BatchEncoding:
        """Tokenizes the formatted prompt."""

        tokenized: BatchEncoding = self.tokenizer(
            example["text"],
            truncation=True,  # Enable truncation here if max_model_length is set on tokenizer
            max_length=self.tokenizer.model_max_length,  # Truncate to model's max length: technically never reached
            return_attention_mask=True,
        )
        return tokenized

    # <---- Apply tokenization to chunked data. ---->
    def DATASET_get_processed_data(self) -> Optional[StreamedSplit]:
        """Apply tokenization to chunked data."""

        if self.chunked_data is None:
            logger.error(
                "Chunked data is not available. Please run DATASET_chunk() first."
            )
            return None

        self.tokenized_data = {
            split: self.chunked_data[split].map(
                self.DATASET_tokenize_fn,
                batched=False,  # Process one example (one prompt string) at a time
                # Remove the original text column after tokenization
                # remove_columns=["text"],
            )
            for split in ["train", "val", "test"]
        }
        logger.info("⚙️ Data successfully tokenized for all splits. ⚙️")

        # clean up chunked_data to free memory
        del self.chunked_data
        gc.collect()

        return self.tokenized_data

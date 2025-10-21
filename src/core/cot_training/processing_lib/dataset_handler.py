import logging
import random

from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from collections import defaultdict


logger = logging.getLogger(name=__name__)


@dataclass
class DatasetHandler:
    """Handles loading, preprocessing, and formatting of the vulnerability dataset."""

    dataset_path: str
    formatted_dataset_path: Path
    tokenizer: PreTrainedTokenizer
    num_cpus: int

    SYSTEM_PROMPT = (
        "You are an expert cybersecurity analyst specializing in C static code analysis. "
        "Your task is to analyze the provided code and produce a step-by-step reasoning "
        "chain explaining whether it contains a vulnerability."
    )

    PROMPT_SKELETON = (
        "**Analysis Instructions:**\n"
        "1.  **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
        "2.  **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
        "3.  **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
        "4.  **Conclude:** State your conclusion based on the analysis.\n\n"
        "**Output Format:**\n"
        "Produce a step-by-step list of your reasoning. After the list, your final answer must be "
        "prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'.\n"
        "--- CODE START ---\n"
        "{func_code}\n"
        "--- CODE END ---\n\n"
        "**Reasoning:**\n"
    ).strip()

    def load_and_split_dataset(
        self, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42
    ) -> DatasetDict:
        """Loads the dataset and performs a stratified, project-based split.

        This method ensures that data from the same project does not appear in
        different splits. It also stratifies the split by project type (mixed-label,
        vulnerable-only, clean-only) to ensure a balanced distribution of labels
        across the train, validation, and test sets, preventing skewed datasets.

        Parameters
        ----------
        test_size: float
            The proportion of projects to allocate to the test set.
        val_size: float
            The proportion of projects to allocate to the validation set.
        seed: int
            A random seed for reproducibility of the splits.

        Returns
        -------
        DatasetDict
            A DatasetDict containing 'train', 'validation', and 'test' splits.
        """

        if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
            raise ValueError(
                "test_size and val_size must be floats between 0 and 1, and their sum must be less than 1."
            )

        logger.info("⚙️ Loading dataset... ⚙️")
        full_dataset = load_dataset("json", data_files=self.dataset_path, split="train")

        logger.info("📊 Analyzing label distribution per project for stratification... 📊")
        project_stats = defaultdict(lambda: {"vulnerable": 0, "clean": 0})
        for example in full_dataset:
            project = example["project"]
            if example["target"] == 1:
                project_stats[project]["vulnerable"] += 1
            else:
                project_stats[project]["clean"] += 1

        mixed_projects, vulnerable_only_projects, clean_only_projects = [], [], []
        for project, stats in project_stats.items():
            if stats["vulnerable"] > 0 and stats["clean"] > 0:
                mixed_projects.append(project)
            elif stats["vulnerable"] > 0:
                vulnerable_only_projects.append(project)
            else:
                clean_only_projects.append(project)

        logger.info(
            f"Found {len(mixed_projects)} mixed, "
            f"{len(vulnerable_only_projects)} vulnerable-only, and "
            f"{len(clean_only_projects)} clean-only projects."
        )

        rng = random.Random(seed)
        rng.shuffle(mixed_projects)
        rng.shuffle(vulnerable_only_projects)
        rng.shuffle(clean_only_projects)

        test_projects, val_projects, train_projects = set(), set(), set()

        def _distribute_projects(project_list: list[str]):
            """Helper to split a list of projects and add to the main sets."""
            num_projects = len(project_list)
            # Ensure at least one project goes to train if list is not empty
            num_test = int(num_projects * test_size)
            num_val = int(num_projects * val_size)

            test_projects.update(project_list[:num_test])
            val_projects.update(project_list[num_test : num_test + num_val])
            train_projects.update(project_list[num_test + num_val :])

        logger.info("✂️ Performing stratified project-based split... ✂️")
        _distribute_projects(mixed_projects)
        _distribute_projects(vulnerable_only_projects)
        _distribute_projects(clean_only_projects)

        train_dataset = full_dataset.filter(lambda example: example["project"] in train_projects)
        val_dataset = full_dataset.filter(lambda example: example["project"] in val_projects)
        test_dataset = full_dataset.filter(lambda example: example["project"] in test_projects)

        logger.info(
            "✅ Stratified project-based split done.\n"
            f"  Train samples: {len(train_dataset)}\n" # type: ignore
            f"  Eval samples: {len(val_dataset)}\n" # type: ignore
            f"  Test samples: {len(test_dataset)}" # type: ignore
        )

        return DatasetDict(
            {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            }
        )

    def format_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Applies CoT prompt formatting to train/validation splits and leaves the
        test split raw for evaluation.

        Parameters
        ----------
        dataset_dict: The DatasetDict containing train, validation, and test splits.

        Returns
        -------
        DatasetDict
            A new DatasetDict whose entires are formatted and tokenized
        """
        logger.info("🥼 Formatting train and validation splits with Chain-of-Thought template... 🥼")

        def formatting_func(example):
            """Formats a single example for Chain-of-Thought fine-tuning."""
            if example["target"] == 1 and example.get("cwe"):
                cwe_string = ", ".join(example["cwe"])
                final_answer = f" YES ({cwe_string})"
            else:
                final_answer = " NO"

            ground_truth = f"{example['reasoning']}\n\nFinal Answer:{final_answer}"
            prompt = self.PROMPT_SKELETON.format(func_code=example["func"])
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ground_truth},
            ]
            return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False)}

        formatted_splits = DatasetDict()

        # Apply formatting to training and validation splits
        for split_name in ["train", "validation"]:
            if split_name in dataset_dict:
                formatted_splits[split_name] = dataset_dict[split_name].map(
                    formatting_func,
                    remove_columns=list(dataset_dict[split_name].features),
                    num_proc=self.num_cpus
                )

        # keep the test split in its original, unformatted state
        if "test" in dataset_dict:
            formatted_splits["test"] = dataset_dict["test"]

        return formatted_splits

    def save_to_disk(self, dataset_dict: DatasetDict, fp: Path):
        dataset_dict.save_to_disk(fp)

    @staticmethod
    def load_from_disk(fp: Path, split: str|None = None) -> DatasetDict|Dataset:
        """Loads a Dataset or DatasetDict from disk.

        Paramters
        ---------
        fp: Path
            The root directory of the saved dataset.
        split: str, default None
            The name of the split to load ("train", "validation", "test").
            If None, loads the entire DatasetDict.

        Returns
        -------
            The specified Dataset split or the entire DatasetDict.
        """

        if split:
            split_path = fp / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split '{split}' not found at {split_path}")
            return load_from_disk(split_path)
        else:
            return load_from_disk(fp)

    def run_pipeline(self) -> DatasetDict:
        dataset_dict: DatasetDict = self.load_and_split_dataset()
        formatted_dataset_dict = self.format_dataset(dataset_dict=dataset_dict)

        self.formatted_dataset_path.mkdir(exist_ok=True)
        self.save_to_disk(formatted_dataset_dict, fp=self.formatted_dataset_path)

        return formatted_dataset_dict

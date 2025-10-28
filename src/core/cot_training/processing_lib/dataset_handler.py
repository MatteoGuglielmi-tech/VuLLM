import logging
import random

from pathlib import Path
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from collections import defaultdict


from ..utilities import rich_table, is_main_process

logger = logging.getLogger(name=__name__)


@dataclass
class DatasetHandler:
    """Handles loading, preprocessing, and formatting of the vulnerability dataset."""

    dataset_path: str
    formatted_dataset_dir: Path
    tokenizer: PreTrainedTokenizer
    num_cpus: int
    debug_mode: bool

    SYSTEM_PROMPT: str = field(
        init=False,
        default=(
            "You are an expert cybersecurity analyst specializing in C static code analysis. "
            "Your task is to analyze the provided code and produce a step-by-step reasoning "
            "chain explaining whether it contains a vulnerability."
        ),
        repr=False,
    )

    PROMPT_SKELETON: str = field(
        init=False,
        default=(
            "**Analysis Instructions:**\n"
            "1. **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
            "2. **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
            "3. **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
            "4. **Conclude:** State your conclusion based on the analysis.\n\n"
            "**Output Format:**\n"
            "Produce a step-by-step list of your reasoning. After the list, your final answer must be "
            "prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'.\n"
            "--- CODE START ---\n"
            "{func_code}\n"
            "--- CODE END ---\n\n"
            "**Reasoning:**\n"
        ).strip(),
        repr=False,
    )

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

        # Calculate total samples
        total_vulnerable = sum(stats["vulnerable"] for stats in project_stats.values())
        total_clean = sum(stats["clean"] for stats in project_stats.values())
        total_samples = total_vulnerable + total_clean

        # Target counts for each split
        target_test_vulnerable = int(total_vulnerable * test_size)
        target_test_clean = int(total_clean * test_size)
        target_val_vulnerable = int(total_vulnerable * val_size)
        target_val_clean = int(total_clean * val_size)
  
        # Create list of (project, vulnerable_count, clean_count) and shuffle
        projects_with_stats = [
            (project, stats["vulnerable"], stats["clean"])
            for project, stats in project_stats.items()
        ]
        rng = random.Random(seed)
        rng.shuffle(projects_with_stats)

        # Greedy assignment to balance labels
        test_projects = set()
        val_projects = set()
        train_projects = set()

        test_vulnerable, test_clean = 0, 0
        val_vulnerable, val_clean = 0, 0

        def _calculate_imbalance(
            current_vuln: int,
            current_clean: int,
            target_vuln: int,
            target_clean: int,
            add_vuln: int,
            add_clean: int,
        ):
            """Calculate how much this assignment would deviate from target proportions."""
            new_vuln = current_vuln + add_vuln
            new_clean = current_clean + add_clean

            # absolute deviation from target
            vuln_diff = abs(new_vuln - target_vuln)
            clean_diff = abs(new_clean - target_clean)

            return vuln_diff + clean_diff

        logger.info("✂️ Performing label-balanced project-based split... ✂️")

        for project, vuln_count, clean_count in projects_with_stats:
            # Calculate imbalance for each split if we add this project
            test_imbalance = _calculate_imbalance(
                current_vuln=test_vulnerable,
                current_clean=test_clean,
                target_vuln=target_test_vulnerable,
                target_clean=target_test_clean,
                add_vuln=vuln_count,
                add_clean=clean_count,
            )
            val_imbalance = _calculate_imbalance(
                current_vuln=val_vulnerable,
                current_clean=val_clean,
                target_vuln=target_val_vulnerable,
                target_clean=target_val_clean,
                add_vuln=vuln_count,
                add_clean=clean_count,
            )

            test_would_exceed = (
                test_vulnerable + vuln_count > target_test_vulnerable * 1.5
                or test_clean + clean_count > target_test_clean * 1.5
            )
            val_would_exceed = (
                val_vulnerable + vuln_count > target_val_vulnerable * 1.5
                or val_clean + clean_count > target_val_clean * 1.5
            )

            # Assign to the split that minimizes imbalance and hasn't exceeded limits
            if test_would_exceed and val_would_exceed:
                # Both exceeded, assign to train
                train_projects.add(project)
            elif test_would_exceed:
                # Test exceeded, choose between val and train
                if val_vulnerable + val_clean < (total_samples * val_size * 1.2):
                    val_projects.add(project)
                    val_vulnerable += vuln_count
                    val_clean += clean_count
                else:
                    train_projects.add(project)
            elif val_would_exceed:
                # Val exceeded, choose between test and train
                if test_vulnerable + test_clean < (total_samples * test_size * 1.2):
                    test_projects.add(project)
                    test_vulnerable += vuln_count
                    test_clean += clean_count
                else:
                    train_projects.add(project)
            else:
                # Neither exceeded, choose split with minimum imbalance
                if test_imbalance <= val_imbalance and test_vulnerable + test_clean < (
                    total_samples * test_size * 1.2
                ):
                    test_projects.add(project)
                    test_vulnerable += vuln_count
                    test_clean += clean_count
                elif val_vulnerable + val_clean < (total_samples * val_size * 1.2):
                    val_projects.add(project)
                    val_vulnerable += vuln_count
                    val_clean += clean_count
                else:
                    train_projects.add(project)

        # Filter the dataset
        train_dataset = full_dataset.filter(
            lambda example: example["project"] in train_projects
        )
        val_dataset = full_dataset.filter(
            lambda example: example["project"] in val_projects
        )
        test_dataset = full_dataset.filter(
            lambda example: example["project"] in test_projects
        )

        # Log actual distributions
        def _count_labels(dataset):
            vuln = sum(1 for ex in dataset if ex["target"] == 1)
            clean = len(dataset) - vuln
            return vuln, clean

        train_vuln, train_clean = _count_labels(train_dataset)
        val_vuln, val_clean = _count_labels(val_dataset)
        test_vuln, test_clean = _count_labels(test_dataset)

        train_split_len = len(train_dataset)
        val_split_len = len(val_dataset)
        test_split_len = len(test_dataset)
        data = {
            "Training": [
                train_split_len, train_vuln, train_clean,
                train_vuln / train_split_len * 100,
                train_clean / train_split_len * 100,
            ],
            "Validation": [
                val_split_len, val_vuln, val_clean,
                val_vuln / val_split_len * 100,
                val_clean / val_split_len * 100,
            ],
            "Test": [
                test_split_len, test_vuln, test_clean,
                test_vuln / test_split_len * 100,
                test_clean / test_split_len * 100,
            ],
        }
        rich_table(
            data=data,
            title="✅ Label-balanced project-based split done.",
            columns=["Split", "Total", "Vulnerable", "Clean", "Vulnerable(%)", "Clean (%)"],
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
            # return {"messages": messages}

        formatted_splits = DatasetDict()

        # Apply formatting to training and validation splits
        for split_name in ["train", "validation"]:
            if split_name in dataset_dict:
                formatted_splits[split_name] = dataset_dict[split_name].map(
                    formatting_func,
                    remove_columns=list(dataset_dict[split_name].features),
                    num_proc=self.num_cpus
                )

        if self.debug_mode and is_main_process():
            print(f"5th training sample: \n {formatted_splits["train"][5]["text"]}")
            print(f"5th validation sample: \n {formatted_splits["validation"][5]["text"]}")

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

        self.formatted_dataset_dir.mkdir(exist_ok=True)
        self.save_to_disk(formatted_dataset_dict, fp=self.formatted_dataset_dir)

        return formatted_dataset_dict

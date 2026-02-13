import json
import logging
import random
import pandas as pd

from typing import TypedDict, cast, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from collections import defaultdict, Counter

from .prompts import VulnPromptFactory
from .datatypes import (
    AssumptionMode,
    PromptPhase,
    VerdictStruct,
    VulnInfo,
    ExpectedModelResponse,
    Message,
    PromptVersion
)
from ..utilities import (
    load_dataset_from_disk,
    rich_rule,
    build_table,
    rich_panel,
    RichColors,
)

logger = logging.getLogger(name=__name__)


class DatasetExample(TypedDict, total=True):
    func: str
    target: int
    project: str
    reasoning: str
    cwe: list[str]
    cwe_desc: list[str]


class SplitStats(TypedDict, total=True):
    projects: int
    percentage: float
    avg_samples_per_project: float
    total_samples: int


class ProjectProfile(NamedTuple):
    """Profile of a project's vulnerability characteristics."""

    name: str
    vulnerable_count: int
    clean_count: int
    cwe_counts: dict[str, int]


@dataclass
class DatasetHandler:
    """Handles loading, preprocessing, and formatting of the vulnerability dataset."""

    dataset_path: str
    formatted_dataset_dir: Path
    tokenizer: PreTrainedTokenizer
    num_cpus: int
    debug_mode: bool

    prompt_phase: PromptPhase
    assumption_mode: AssumptionMode
    add_cwe_guidelines: bool
    prompt_version: PromptVersion

    def __post_init__(self):
        self.prompt_config = VulnPromptFactory.create(
            version=self.prompt_version,
            prompt_phase=self.prompt_phase,
            assumptions_mode=self.assumption_mode,
            add_cwe_guidelines=self.add_cwe_guidelines,
            debug_mode=self.debug_mode,
        )

    def load_and_split_dataset(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        seed: int = 42,
        min_train_cwe_samples: int = 30,
    ) -> DatasetDict:
        """Loads the dataset and performs a CWE-aware, project-based split.

        This method ensures:
        1. Data from the same project does not appear in different splits
        2. Label distribution is balanced across splits
        3. Each CWE has sufficient representation in the training set

        Parameters
        ----------
        test_size: float
            The proportion of projects to allocate to the test set.
        val_size: float
            The proportion of projects to allocate to the validation set.
        seed: int
            A random seed for reproducibility.
        min_train_cwe_samples: int
            Minimum number of samples per CWE required in the training set.
            CWEs that cannot meet this threshold will trigger a warning.

        Returns
        -------
        DatasetDict
            A DatasetDict containing 'train', 'validation', and 'test' splits.
        """

        if not (0 < test_size < 1 and 0 < val_size < 1 and (test_size + val_size) < 1):
            raise ValueError(
                "test_size and val_size must be floats between 0 and 1, "
                "and their sum must be less than 1."
            )

        logger.info("⏬ Loading dataset via pandas (bypassing HF schema inference)...")
        df = pd.read_json(self.dataset_path, lines=True)

        def sanitize_list_field(val):
            if val is None:
                return []
            if isinstance(val, list):
                return [str(v) for v in val if v]
            return []

        df["cwe"] = df["cwe"].apply(sanitize_list_field)
        df["cwe_desc"] = df["cwe_desc"].apply(sanitize_list_field)

        # Keep only required columns
        required_columns = ["project", "func", "target", "cwe", "cwe_desc", "reasoning"]
        df = cast(pd.DataFrame, df[[c for c in required_columns if c in df.columns]])
        full_dataset: Dataset = Dataset.from_pandas(df, preserve_index=False)

        # explicitly cast to ensure correct types
        features = Features(
            {
                "project": Value("string"),
                "func": Value("string"),
                "target": Value("int64"),
                "cwe": Sequence(Value("string")),
                "cwe_desc": Sequence(Value("string")),
                "reasoning": Value("string"),
            }
        )

        full_dataset = full_dataset.cast(features)

        logger.info(f"✅ Dataset loaded: {len(full_dataset)} samples")

        logger.info("📊 Building project profiles with CWE information...")
        project_profiles = self._build_project_profiles(full_dataset)

        global_cwe_counts = self._get_global_cwe_counts(project_profiles)
        logger.info(f"Found {len(global_cwe_counts)} unique CWEs across all projects")

        train_size = 1 - test_size - val_size
        rare_cwes = {
            cwe: count
            for cwe, count in global_cwe_counts.items()
            if count < min_train_cwe_samples / train_size 
        }
        if rare_cwes:
            logger.warning(
                f"⚠️ {len(rare_cwes)} CWEs have very few samples and may not be "
                f"well-represented in training: {list(rare_cwes.keys())}"
            )

        # Perform the CWE-aware split
        logger.info("🔪 Performing CWE-aware, label-balanced, project-based split...")
        train_projects, val_projects, test_projects = self._cwe_aware_split(
            project_profiles=project_profiles,
            global_cwe_counts=global_cwe_counts,
            test_size=test_size,
            val_size=val_size,
            min_train_cwe_samples=min_train_cwe_samples,
            seed=seed,
        )

        train_dataset = full_dataset.filter(lambda ex: ex["project"] in train_projects)
        val_dataset = full_dataset.filter(lambda ex: ex["project"] in val_projects)
        test_dataset = full_dataset.filter(lambda ex: ex["project"] in test_projects)

        # Log distributions and CWE coverage
        self._log_split_statistics(
            train_dataset,
            val_dataset,
            test_dataset,
            project_profiles,
            train_projects,
            val_projects,
            test_projects,
            min_train_cwe_samples,
        )

        dataset_dict = DatasetDict(  # type: ignore[reportCallIssue]
            {  # type: ignore[reportArgumentType]
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset,
            }
        )

        self._log_cwe_coverage_details(
            dataset_dict=dataset_dict,
            min_train_cwe_samples=min_train_cwe_samples,
        )

        return dataset_dict

    def _build_project_profiles(self, dataset) -> dict[str, ProjectProfile]:
        """Build detailed profiles for each project."""
        stats = defaultdict(
            lambda: {"vulnerable": 0, "clean": 0, "cwes": defaultdict(int)}
        )

        for example in dataset:
            project = example["project"]
            if example["target"] == 1:
                stats[project]["vulnerable"] += 1  # type: ignore[reportOperatorIssue]
                for cwe in example.get("cwe", []):
                    if cwe:
                        stats[project]["cwes"][cwe] += 1  # type: ignore[reportOperatorIssue]
            else:
                stats[project]["clean"] += 1  # type: ignore[reportOperatorIssue]

        return {
            project: ProjectProfile(
                name=project,
                vulnerable_count=data["vulnerable"],  # type: ignore[report]
                clean_count=data["clean"],  # type: ignore[reportArgumentType]
                cwe_counts=dict(data["cwes"]),  # type: ignore[reportArgumentType]
            )
            for project, data in stats.items()
        }

    def _get_global_cwe_counts(
        self, profiles: dict[str, ProjectProfile]
    ) -> dict[str, int]:
        """Get total count of each CWE across all projects."""
        cwe_counts = defaultdict(int)
        for profile in profiles.values():
            for cwe, count in profile.cwe_counts.items():
                cwe_counts[cwe] += count
        return dict(cwe_counts)

    def _analyze_cwe_distribution(
        self,
        project_profiles: dict[str, ProjectProfile],
        global_cwe_counts: dict[str, int],
        min_train_cwe_samples: int,
    ) -> dict[str, dict]:
        """
        Analyze CWE distribution to identify problematic cases.

        Returns a dict with risk assessment for each CWE.
        """
        cwe_analysis = {}

        for cwe, total_count in global_cwe_counts.items():
            # Find which projects contain this CWE and how many samples each has
            projects_with_cwe = {
                name: profile.cwe_counts.get(cwe, 0)
                for name, profile in project_profiles.items()
                if cwe in profile.cwe_counts
            }

            num_projects = len(projects_with_cwe)
            max_in_single_project = max(projects_with_cwe.values())
            concentration_ratio = max_in_single_project / total_count

            # Calculate minimum guaranteed training samples
            # (what we'd have if the largest project goes to test/val)
            min_guaranteed_train = total_count - max_in_single_project

            # Risk assessment
            if num_projects == 1:
                risk = "CRITICAL"  # All-or-nothing for training
                risk_reason = (
                    f"Only exists in project '{list(projects_with_cwe.keys())[0]}'"
                )
            elif min_guaranteed_train < min_train_cwe_samples:
                risk = "HIGH"  # Could fall below threshold
                risk_reason = (
                    f"If largest project excluded, only {min_guaranteed_train} "
                    f"samples remain (need {min_train_cwe_samples})"
                )
            elif concentration_ratio > 0.7:
                risk = "MEDIUM"  # Heavily concentrated
                risk_reason = f"{concentration_ratio:.0%} of samples in one project"
            else:
                risk = "LOW"
                risk_reason = f"Well distributed across {num_projects} projects"

            cwe_analysis[cwe] = {
                "total_count": total_count,
                "num_projects": num_projects,
                "projects": projects_with_cwe,
                "concentration_ratio": concentration_ratio,
                "min_guaranteed_train": min_guaranteed_train,
                "risk": risk,
                "risk_reason": risk_reason,
            }

        return cwe_analysis

    def _cwe_aware_split(
        self,
        project_profiles: dict[str, ProjectProfile],
        global_cwe_counts: dict[str, int],
        test_size: float,
        val_size: float,
        min_train_cwe_samples: int,
        seed: int,
    ) -> tuple[set[str], set[str], set[str]]:
        """
        Perform a split that prioritizes CWE coverage in training.

        Strategy:
        1. First, ensure each CWE has minimum representation in training
           by assigning "critical" projects (those with rare CWEs) to training first
        2. Then, distribute remaining projects using label-balanced approach
        """
        rng = random.Random(seed)

        cwe_analysis = self._analyze_cwe_distribution(
            project_profiles, global_cwe_counts, min_train_cwe_samples
        )

        # Log warnings for problematic CWEs
        critical_cwes = [
            c for c, info in cwe_analysis.items() if info["risk"] == "CRITICAL"
        ]
        high_risk_cwes = [
            c for c, info in cwe_analysis.items() if info["risk"] == "HIGH"
        ]

        if critical_cwes:
            logger.warning(
                f"🚨 CRITICAL: {len(critical_cwes)} CWEs exist in only ONE project. "
                f"These projects MUST be in training: {critical_cwes}"
            )
        if high_risk_cwes:
            logger.warning(
                f"⚠️ HIGH RISK: {len(high_risk_cwes)} CWEs are heavily concentrated. "
                f"May not meet minimum training samples: {high_risk_cwes}"
            )

        total_vulnerable = sum(p.vulnerable_count for p in project_profiles.values())
        total_clean = sum(p.clean_count for p in project_profiles.values())

        target_test_vulnerable = int(total_vulnerable * test_size)
        target_test_clean = int(total_clean * test_size)
        target_val_vulnerable = int(total_vulnerable * val_size)
        target_val_clean = int(total_clean * val_size)

        train_projects: set[str] = set()
        val_projects: set[str] = set()
        test_projects: set[str] = set()
        assigned_projects: set[str] = set()

        train_cwe_counts: dict[str, int] = defaultdict(int)

        # Track current label counts
        counts = {
            "train": {"vuln": 0, "clean": 0},
            "val": {"vuln": 0, "clean": 0},
            "test": {"vuln": 0, "clean": 0},
        }

        # Helper to assign a project to training and update all tracking state
        def assign_to_train(project_name: str) -> None:
            profile = project_profiles[project_name]
            train_projects.add(project_name)
            assigned_projects.add(project_name)

            # Update label counts
            counts["train"]["vuln"] += profile.vulnerable_count
            counts["train"]["clean"] += profile.clean_count

            # Update CWE counts
            for cwe, cnt in profile.cwe_counts.items():
                train_cwe_counts[cwe] += cnt

        # Phase 1: FORCE single-project CWEs into training
        for cwe, info in cwe_analysis.items():
            if info["num_projects"] == 1:
                project_name = list(info["projects"].keys())[0]
                if project_name not in assigned_projects:
                    assign_to_train(project_name)
                    logger.info(
                        f"Forced '{project_name}' to training (only source of {cwe})"
                    )

        # Phase 2: Handle high-risk CWEs by prioritizing their largest projects for training
        # Sort by how far below threshold they'd be (most at-risk first)
        high_risk_sorted = sorted(
            [
                (cwe, info)
                for cwe, info in cwe_analysis.items()
                if info["risk"] == "HIGH"
            ],
            key=lambda x: x[1]["min_guaranteed_train"],
        )

        for cwe, info in high_risk_sorted:
            current_train_count = train_cwe_counts[cwe]
            if current_train_count >= min_train_cwe_samples:
                continue

            # Sort projects by contribution to this CWE (largest first)
            candidate_projects = sorted(
                [
                    (p, cnt)
                    for p, cnt in info["projects"].items()
                    if p not in assigned_projects
                ],
                key=lambda x: x[1],
                reverse=True,
            )

            for proj_name, cwe_count in candidate_projects:
                if current_train_count >= min_train_cwe_samples:
                    break

                assign_to_train(proj_name)
                current_train_count += cwe_count

        # Phase 3: Distribute remaining projects using label-balanced approach
        remaining_profiles = [
            profile
            for name, profile in project_profiles.items()
            if name not in assigned_projects
        ]
        rng.shuffle(remaining_profiles)

        for profile in remaining_profiles:
            # Calculate how much each split needs
            test_need = max(0, target_test_vulnerable - counts["test"]["vuln"]) + max(
                0, target_test_clean - counts["test"]["clean"]
            )
            val_need = max(0, target_val_vulnerable - counts["val"]["vuln"]) + max(
                0, target_val_clean - counts["val"]["clean"]
            )

            # Check if splits are "full"
            test_full = (
                counts["test"]["vuln"] >= target_test_vulnerable * 1.2
                and counts["test"]["clean"] >= target_test_clean * 1.2
            )
            val_full = (
                counts["val"]["vuln"] >= target_val_vulnerable * 1.2
                and counts["val"]["clean"] >= target_val_clean * 1.2
            )

            # Decide assignment
            if test_full and val_full:
                split = "train"
            elif test_full:
                split = "val" if not val_full else "train"
            elif val_full:
                split = "test" if not test_full else "train"
            elif test_need >= val_need:
                split = "test"
            else:
                split = "val"

            # Assign
            if split == "train":
                train_projects.add(profile.name)
                # Update CWE counts for training
                for cwe, cwe_count in profile.cwe_counts.items():
                    train_cwe_counts[cwe] += cwe_count
            elif split == "val":
                val_projects.add(profile.name)
            else:
                test_projects.add(profile.name)

            counts[split]["vuln"] += profile.vulnerable_count
            counts[split]["clean"] += profile.clean_count

        return train_projects, val_projects, test_projects

    def _log_split_statistics(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        project_profiles,
        train_projects,
        val_projects,
        test_projects,
        min_train_cwe_samples,
    ):
        """Log comprehensive statistics about the split."""

        def count_labels(dataset):
            vuln = sum(1 for ex in dataset if ex["target"] == 1)
            return vuln, len(dataset) - vuln

        train_vuln, train_clean = count_labels(train_dataset)
        val_vuln, val_clean = count_labels(val_dataset)
        test_vuln, test_clean = count_labels(test_dataset)

        # Log label distribution (your existing table)
        data = {
            "Training": [
                len(train_dataset),
                train_vuln,
                train_clean,
                train_vuln / len(train_dataset) * 100,
                train_clean / len(train_dataset) * 100,
            ],
            "Validation": [
                len(val_dataset),
                val_vuln,
                val_clean,
                val_vuln / len(val_dataset) * 100,
                val_clean / len(val_dataset) * 100,
            ],
            "Test": [
                len(test_dataset),
                test_vuln,
                test_clean,
                test_vuln / len(test_dataset) * 100,
                test_clean / len(test_dataset) * 100,
            ],
        }

        rich_rule()
        global split_stats_tb
        split_stats_tb = build_table(
            data=data,
            title="✅ Split Statistics - Labels",
            columns=[
                "Split",
                "Total",
                "Vulnerable",
                "Clean",
                "Vulnerable(%)",
                "Clean(%)",
            ],
        )

        # Log CWE coverage
        train_cwes = defaultdict(int)
        val_cwes = defaultdict(int)
        test_cwes = defaultdict(int)

        for proj in train_projects:
            for cwe, count in project_profiles[proj].cwe_counts.items():
                train_cwes[cwe] += count
        for proj in val_projects:
            for cwe, count in project_profiles[proj].cwe_counts.items():
                val_cwes[cwe] += count
        for proj in test_projects:
            for cwe, count in project_profiles[proj].cwe_counts.items():
                test_cwes[cwe] += count

        all_cwes = set(train_cwes) | set(val_cwes) | set(test_cwes)

        # Identify problematic CWEs
        missing_in_train = [cwe for cwe in all_cwes if train_cwes[cwe] == 0]
        low_in_train = [
            cwe for cwe in all_cwes if 0 < train_cwes[cwe] < min_train_cwe_samples
        ]

        if missing_in_train:
            logger.warning(
                f"⚠️ {len(missing_in_train)} CWEs have ZERO samples in training: "
                f"{missing_in_train}"
            )
        if low_in_train:
            logger.warning(
                f"⚠️ {len(low_in_train)} CWEs have fewer than {min_train_cwe_samples} "
                f"samples in training: {low_in_train}"
            )

    def _balance_training_set(
        self,
        dataset_dict: DatasetDict,
        target_ratio: float = 0.5,
        seed: int = 42,
    ) -> DatasetDict:
        """
        Balance training set with CWE-stratified oversampling.

        Ensures low-frequency CWEs get proportionally more representation.
        """
        train_dataset: Dataset = dataset_dict["train"]

        # Group indices by CWE
        from collections import defaultdict

        cwe_to_indices: dict[str, list[int]] = defaultdict(list)
        safe_indices: list[int] = []

        for i, ex in enumerate(train_dataset):
            if ex["target"] == 1:  # type: ignore[reportCallIssue, reportArgumentType ]
                for cwe in ex.get("cwe", []):  # type: ignore[reportAttributeAccessIssue]
                    cwe_to_indices[cwe].append(i)
            else:
                safe_indices.append(i)

        n_safe = len(safe_indices)
        n_vulnerable_target = int(n_safe * target_ratio / (1 - target_ratio))

        # Get unique vulnerable indices and current count
        all_vulnerable_indices = set()
        for indices in cwe_to_indices.values():
            all_vulnerable_indices.update(indices)
        n_vulnerable_current = len(all_vulnerable_indices)

        # Store original distribution for comparison
        original_cwe_counts = {cwe: len(indices) for cwe, indices in cwe_to_indices.items()}

        if n_vulnerable_target <= n_vulnerable_current:
            logger.info(
                "✅ No need for upsampling. Vulnerable samples already over desired ratio"
            )
            return dataset_dict

        n_to_add = n_vulnerable_target - n_vulnerable_current

        # Calculate per-CWE sampling weights (inverse frequency)
        cwe_counts = {cwe: len(indices) for cwe, indices in cwe_to_indices.items()}
        total_cwe_samples = sum(cwe_counts.values())

        # Inverse frequency weighting: rare CWEs get higher weight
        cwe_weights = {
            cwe: total_cwe_samples / count for cwe, count in cwe_counts.items()
        }
        total_weight = sum(cwe_weights.values())
        cwe_probs = {cwe: w / total_weight for cwe, w in cwe_weights.items()}

        freq_data: dict[str, list[int|float|str]] = defaultdict(list)
        for cwe, prob in sorted(cwe_probs.items(), key=lambda x: -x[1]):
            freq_data[cwe].extend([f"{prob:.3f}", cwe_counts[cwe]])

        freq_tb = build_table(
            data=freq_data,
            title="📊 CWE sampling weights (inverse frequency)",
            columns=["CWE", "Weight", "Count"],
        )
        # Sample CWEs according to weights, then sample index from that CWE
        rng = random.Random(seed)
        cwes = list(cwe_probs.keys())
        probs = [cwe_probs[c] for c in cwes]

        duplicated_indices = []
        duplicated_cwe_counts: Counter[str] = Counter()
        for _ in range(n_to_add):
            # Pick CWE weighted by inverse frequency
            chosen_cwe = rng.choices(cwes, weights=probs, k=1)[0]
            # Pick random sample from that CWE
            chosen_idx = rng.choice(cwe_to_indices[chosen_cwe])
            duplicated_indices.append(chosen_idx)
            duplicated_cwe_counts[chosen_cwe] += 1

        # Combine and shuffle
        all_indices = safe_indices + list(all_vulnerable_indices) + duplicated_indices
        rng.shuffle(all_indices)

        balanced_train = train_dataset.select(all_indices)

        # =========================================================================
        # LOG CWE DISTRIBUTION BEFORE/AFTER
        # =========================================================================
        final_cwe_counts = {
            cwe: original_cwe_counts[cwe] + duplicated_cwe_counts.get(cwe, 0)
            for cwe in original_cwe_counts
        }

        data: dict[str, list[int|float|str]] = defaultdict(list)
        for cwe in sorted(original_cwe_counts.keys(), key=lambda x: int(x.split("-")[1])):
            before = original_cwe_counts[cwe]
            added = duplicated_cwe_counts.get(cwe, 0)
            after = final_cwe_counts[cwe]
            change_pct = (after - before) / before * 100 if before > 0 else 0
            data[cwe].extend([before, added, after, f"{+round(change_pct,2)}%"])

        data["TOTAL"] = [
            sum(original_cwe_counts.values()),
            sum(duplicated_cwe_counts.values()),
            sum(final_cwe_counts.values()),
            "-",
        ]
        before_after_tb = build_table(
            data=data,
            title="📊 CWE DISTRIBUTION BEFORE/AFTER BALANCING",
            columns=["CWE", "Before", "Added", "After", "Change"],
        )

        rich_panel(
            tables=[freq_tb, before_after_tb],
            panel_title="Training set CWE distribution after balancing",
            border_style=RichColors.STEEL_BLUE,
        )

        total_samples = len(balanced_train)
        total_vulnerable = n_vulnerable_target
        logger.info(
            f"✅ Final balance: {total_vulnerable} vulnerable ({total_vulnerable/total_samples:.1%}), "
            f"{n_safe} safe ({n_safe/total_samples:.1%})"
        )

        return DatasetDict(
            {
                "train": balanced_train,
                "validation": dataset_dict["validation"],
                "test": dataset_dict["test"],
            }
        )

    def _log_cwe_coverage_details(
        self,
        dataset_dict: DatasetDict,
        min_train_cwe_samples: int,
    ) -> None:
        """
        Log detailed CWE coverage across all splits.

        Parameters
        ----------
        dataset_dict : DatasetDict
            Dataset dictionary with 'train', 'validation', 'test' splits.
        min_train_cwe_samples : int
            Minimum samples per CWE required in training. CWEs below this
            threshold will be flagged with a warning.
        """
        from collections import defaultdict

        splits = ["train", "validation", "test"]
        cwe_counts: dict[str, dict[str, int]] = {
            split: defaultdict(int) for split in splits
        }

        for split in splits:
            for ex in dataset_dict[split]:
                if ex["target"] == 1: # type: ignore[reportCallIssue, reportArgumentType]
                    for cwe in ex.get("cwe", []): # type: ignore[reportAttributeAccessIssue]
                        if cwe:
                            cwe_counts[split][cwe] += 1

        # Get all unique CWEs
        all_cwes: set[str] = set()
        for split in splits:
            all_cwes.update(cwe_counts[split].keys())

        # Build data for rich_table
        data: dict[str, list] = {}
        flagged_cwes: list[str] = []

        for cwe in sorted(all_cwes):
            train = cwe_counts["train"].get(cwe, 0)
            val = cwe_counts["validation"].get(cwe, 0)
            test = cwe_counts["test"].get(cwe, 0)
            total = train + val + test

            # Flag if training count is below threshold
            if train < min_train_cwe_samples:
                flagged_cwes.append(cwe)
                cwe_display = f"{cwe} ⚠️"
            else:
                cwe_display = cwe

            data[cwe_display] = [train, val, test, total]


        coverage_details_table = build_table(
            data=data,
            title="📊 CWE Coverage Details",
            columns=["CWE", "Train", "Validation", "Test", "Total"],
        )

        rich_panel(
            tables=[split_stats_tb, coverage_details_table],
            panel_title="Training set original stats",
            border_style=RichColors.STEEL_BLUE,
        )

        # Log summary
        if flagged_cwes:
            logger.warning(
                f"⚠️ {len(flagged_cwes)} CWEs below minimum threshold "
                f"({min_train_cwe_samples}) in training: {flagged_cwes}"
            )
        else:
            logger.info(
                f"✅ All {len(all_cwes)} CWEs meet minimum threshold "
                f"({min_train_cwe_samples}) in training"
            )

    def enrich_reasoning_with_json(self, example: DatasetExample) -> str:
        """
        Add JSON-like structure to create direct visual mapping.
        Teaches model to connect analysis to structured output.
        """
        base_reasoning = example["reasoning"].strip()

        if example["target"] == 0:
            # Safe code
            assessment = {
                "vulnerabilities": [],
                "verdict": {"is_vulnerable": False, "cwe_list": []},
            }
        else:
            cwe_descs = example["cwe_desc"]

            # Build vulnerability entries
            vulnerabilities = []
            cwe_ids = []

            for i, cwe in enumerate(example["cwe"]):
                cwe_id = int(cwe.replace("CWE-", "").strip())
                cwe_ids.append(cwe_id)
                desc = cwe_descs[i] if i < len(cwe_descs) else f"CWE-{cwe_id}"
                vulnerabilities.append({"cwe_id": cwe_id, "description": desc})

            assessment = {
                "vulnerabilities": vulnerabilities,
                "verdict": {"is_vulnerable": True, "cwe_list": cwe_ids},
            }

        # Properly formatted and escaped JSON
        # json_str = json.dumps(assessment, indent=2, ensure_ascii=False)
        json_str = json.dumps(
            assessment,
            indent=None,  # or use separators=(',', ':') for ultra compact
            ensure_ascii=False,
        )
        return (
            base_reasoning + f"\n\n**Structured Assessment:**\n```json\n{json_str}\n```"
        )

    def _formatting_func(self, example: DatasetExample) -> Message:
        """
        Format example (assumes pre-filtering removed invalid examples).
        Should never return None if pre-filtering worked correctly.

        Parameters
        ----------
        example : DatasetExample
            Dataset entry with fileds: 'func', 'reasoning', 'target', 'cwe', 'cwe_desc'

        Returns
        -------
        dict
            Formatted example with 'text' field
        """

        # Enrich reasoning
        enriched_reasoning = self.enrich_reasoning_with_json(example=example)

        vulnerabilities: list[VulnInfo] = []

        if example["target"] == 1 and example["cwe"]:
            cwe_descs: list[str] = example["cwe_desc"]
            for i, cwe in enumerate(example["cwe"]):
                try:
                    cwe_id = int(cwe.replace("CWE-", "").strip())

                    description = (
                        cwe_descs[i]
                        if i < len(cwe_descs)
                        else f"CWE-{cwe_id} vulnerability"
                    )

                    # validate fields and then append
                    vuln_info = VulnInfo(cwe_id=cwe_id, description=description)
                    vulnerabilities.append(vuln_info)
                    # cwe_list.append(cwe_id)

                except ValueError:
                    logger.exception(f"Error parsing for CWE '{cwe}`")
                    continue

        # cwe_list from validated vulnerabilities (guaranteed match!)
        cwe_list: list[int] = [v.cwe_id for v in vulnerabilities]

        # build structured response
        response_data: ExpectedModelResponse = ExpectedModelResponse(
            reasoning=enriched_reasoning,
            vulnerabilities=vulnerabilities,
            verdict=VerdictStruct(
                is_vulnerable=bool(example["target"]), cwe_list=cwe_list
            ),
        )

        # convert to formatted JSON
        messages = self.prompt_config.as_messages(
            func_code=example["func"],
            ground_truth=response_data.model_dump_json(indent=None, ensure_ascii=False),
        )

        return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False)}

    def is_valid_example(self, example: DatasetExample) -> bool:
        """Check if example is valid for training."""

        if example["target"] == 0:  # safe
            return True

        # keep  vulnerable samples with:
        # - filled cwe list
        # - valid vulenrability descrpions
        cwes = example.get("cwe")
        if not cwes:
            return False

        # Check descriptions are valid
        cwe_descs = example.get("cwe_desc", [])
        if not cwe_descs:
            return False

        # Ensure no placeholder descriptions
        return all(
            desc.strip() and desc != "Description not found." for desc in cwe_descs
        )

    def format_dataset(
        self,
        dataset_dict: DatasetDict,
        save_jsonl: bool = True,
        output_dir: Path | None = None,
    ) -> DatasetDict:
        """Applies CoT prompt formatting to train/validation splits and leaves the
        test split raw for evaluation.

        Parameters
        ----------
        dataset_dict: DatasetDict
            Dataset containing train, validation, and test splits.
        save_jsonl : bool, default=True
            Whether to save formatted (pre-tokenization) data to JSONL
        output_dir : Path | None
            Directory to save JSONL files. If None, uses self.output_dir or current directory.

        Returns
        -------
        DatasetDict
            A new DatasetDict whose entires are formatted and tokenized
        """
        logger.info("💱 Formatting train and validation splits...")

        if output_dir is None:
            output_dir = getattr(self, "formatted_dataset_dir", Path.cwd())

        output_dir = Path(output_dir)  # type: ignore
        output_dir.mkdir(parents=True, exist_ok=True)

        formatted_splits = DatasetDict()

        # apply formatting to training and validation splits
        for split_name in ["train", "validation"]:
            if split_name not in dataset_dict:
                continue

            split_data = dataset_dict[split_name]

            filtered = split_data.filter(self.is_valid_example, num_proc=self.num_cpus)

            filtered_count = len(split_data) - len(filtered)
            safe_count = sum(1 for ex in filtered if ex["target"] == 0)  # type: ignore[reportCallIssue, reportArgumentType]
            vuln_count = len(filtered) - safe_count

            logger.info(
                f"{split_name}: Kept {len(filtered)}/{len(split_data)} examples "
                f"(filtered {filtered_count})\n"
                f"  Safe: {safe_count} ({safe_count/len(filtered):.1%})\n"
                f"  Vulnerable: {vuln_count} ({vuln_count/len(filtered):.1%})"
            )

            if save_jsonl:
                op: Path = output_dir / "json"
                op.mkdir(exist_ok=True, parents=True)
                op = op / f"{split_name}_formatted.json"
                self._save_to_jsonl(
                    dataset=filtered,
                    split_name=split_name,
                    output_path=op,
                )

            # format (guaranteed no None)
            formatted = filtered.map(
                self._formatting_func,
                remove_columns=list(filtered.features),
                num_proc=self.num_cpus,
            )

            logger.debug(f"✅ {split_name} split columns after formatting: {formatted.column_names}")

            formatted_splits[split_name] = formatted

        # keep the test split in its original, unformatted state
        if "test" in dataset_dict:
            formatted_splits["test"] = dataset_dict["test"]

        return formatted_splits

    def _save_to_jsonl(
        self, dataset: Dataset, split_name: str, output_path: Path
    ) -> None:
        """
        Save dataset to JSONL file for human inspection.

        Parameters
        ----------
        dataset : Dataset
            Dataset to save
        split_name : str
            Name of the split (train/validation/test)
        output_path : Path
            Path to save the file
        """

        logger.info(f"💾 Saving {split_name} split to {output_path}")

        with open(output_path, "w", encoding="utf-8") as f:
            for example in dataset:
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")

        logger.info(f"✅ Saved {len(dataset)} examples to {output_path}")

    def save_to_disk(self, dataset_dict: DatasetDict, subdir: str = "dataset") -> None:
        """Save formatted dataset.

        Parameters
        ----------
        dataset_dict : DatasetDict
            DatasetDict instance to save
        """

        from ..utilities import save_dataset

        op: Path = self.formatted_dataset_dir / subdir
        op.mkdir(exist_ok=True, parents=True)

        save_dataset(
            dataset=dataset_dict,
            output_location=op,
        )

    def run_pipeline(self, target_vulnerable_ratio: float | None) -> DatasetDict:
        """
        Run the full data processing pipeline.

        Parameters
        ----------
        target_vulnerable_ratio : float|None
            Desired ratio of vulnerable samples in training set.
            Set to None to skip balancing.

        Returns
        -------
        DatasetDict
            Formatted dataset ready for training.
        """
        if self.formatted_dataset_dir.exists() and any(self.formatted_dataset_dir.iterdir()):
            dataset_dir: Path = self.formatted_dataset_dir / "dataset"
            dataset_dict: DatasetDict = load_dataset_from_disk(input_dir=dataset_dir)
            logger.info(f"✅ DatasetDict loaded from disk: {dataset_dir}")
            return dataset_dict

        dataset_dict: DatasetDict = self.load_and_split_dataset()
        self.save_to_disk(dataset_dict=dataset_dict, subdir="pre-formatting")

        if target_vulnerable_ratio is not None:
            dataset_dict = self._balance_training_set(
                dataset_dict=dataset_dict,
                target_ratio=target_vulnerable_ratio,
            )

        self.save_to_disk(dataset_dict=dataset_dict, subdir="pre-formatting-post-balancing")

        formatted_dataset_dict: DatasetDict = self.format_dataset(dataset_dict=dataset_dict)
        self.save_to_disk(dataset_dict=formatted_dataset_dict)

        return formatted_dataset_dict


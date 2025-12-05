import logging

from typing import overload
from pathlib import Path
from datasets import DatasetDict, Dataset, load_from_disk


logger = logging.getLogger(__name__)


def save_dataset(
    dataset: Dataset | DatasetDict,
    output_location: Path,
    split_name: str = "train",
) -> None:
    """Save evaluation results with custom split name.

    Parameters
    ----------
    results_dataset : Dataset
        Dataset instance
    output_dir : Path
        Directory to save to
    split_name : str
        Name of the split (e.g., "train", "test", "validation", etc.)
    """

    already_datsetdict: bool = isinstance(dataset, DatasetDict)

    dataset_dict: DatasetDict = (
        DatasetDict({split_name: dataset})
        if not already_datsetdict
        else dataset
    )
    dataset_dict.save_to_disk(dataset_dict_path=output_location)

    msg = f"✅ Saved {len(dataset)} samples to {output_location}"
    if already_datsetdict:
        msg += f" (split {split_name})"

    logger.info(msg)


@overload
def load_dataset_from_disk(
    input_dir: Path,
    split_name: None = None,
) -> DatasetDict: ...


@overload
def load_dataset_from_disk(
    input_dir: Path,
    split_name: str,
) -> Dataset: ...


def load_dataset_from_disk(
    input_dir: Path, split_name: str | None = None
) -> Dataset | DatasetDict:
    """Load evaluation results from disk.
    Dataset needs to be previously saved via [~utilities.disk.save_dataset].

    Parameters
    ----------
    input_dir : Path
        Directory to load from
    split_name : str, optional (default=None)
        Name of the split to load, when None, it loads all splits

    Returns
    -------
    Dataset|DatasetDict
        Loaded dataset
    """


    if not input_dir.exists():
        raise FileNotFoundError(f"Path '{input_dir}' not found.")

    loaded: Dataset | DatasetDict = load_from_disk(dataset_path=input_dir)

    if isinstance(loaded, Dataset):
        return loaded # Dataset

    if split_name is None:
        return loaded  # DatasetDict
    elif split_name not in loaded:
        raise KeyError(f"`{split_name}` split not found. Available: {list(loaded.keys())}")

    return loaded[split_name]  # Dataset


from typing import overload
from pathlib import Path
from datasets import DatasetDict, Dataset, load_from_disk


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
        return loaded  # Dataset

    if split_name is None:
        return loaded  # DatasetDict
    elif split_name not in loaded:
        raise KeyError(f"`{split_name}` split not found. Available: {list(loaded.keys())}")

    return loaded[split_name]  # Dataset

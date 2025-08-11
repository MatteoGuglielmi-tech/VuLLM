from collections.abc import Generator
import difflib
import json
from typing import Any
from pathlib import Path

from .decorators import ensure_jsonl_extension
from .typedef import JsonlData


import logging
from .stdout import MY_LOGGER_NAME
logger = logging.getLogger(MY_LOGGER_NAME)


@ensure_jsonl_extension
def save_to_jsonl(dataset: JsonlData, filepath: str):
    """Saves a list of dictionaries to a file in JSON Lines format.

    Parameters
    ----------
    dataset : list of dict
        The dataset to be saved. Each element of the list should be a
        JSON-serializable dictionary.
    filepath : str
        The path to the file where the data will be saved.
    """

    with open(file=filepath, mode="w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")


@ensure_jsonl_extension
def load_from_jsonl(filepath: Path) -> JsonlData:
    """Loads data from a JSON Lines file into a list of dictionaries.

    Parameters
    ----------
    filepath : pathlib.Path
        A pathlib.Path object to the .jsonl file.

    Returns:
        A list of dictionaries, where each dictionary represents a line in the file.
    """

    data: JsonlData = []
    with open(file=filepath, mode="r", encoding="utf-8") as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON on line: {line.strip()}. Error: {e}")
                continue
    return data


def diff_two_strings(anchor_str:str, query_str:str) -> Generator[tuple[str,int,int,int,int], None, None]:
    """
    Compares two strings and yields the raw opcode data for each difference.

    Parameters
    ----------
    anchor_str : str
        The base string for comparison.
    query_str : str
        The string to compare against the anchor.

    Yields
    ------
    tuple[str, int, int, int, int]
        A tuple containing the tag, start/end indices for the first string,
        and start/end indices for the second string.

    Notes
    -----
    The `tag` string describes the operation required to transform `anchor_str` into `query_str`:
      - 'replace': The slice `anchor_str[j1:j2]` should be replaced by `query_str[i1:i2]`.
      - 'delete': The slice `anchor_str[j1:j2]` should be deleted (it is not present in `query_str`).
      - 'insert': The slice `query_str[i1:i2]` should be inserted into `anchor_str` at position `j1`.
    """

    s = difflib.SequenceMatcher(None, a=query_str, b=anchor_str, autojunk=False)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag != "equal":
            yield (tag, i1, i2, j1, j2)


def load_json_config(filepath:str) -> dict[str,Any]:
    """Loads a configuration from a standard JSON file.

    This function opens and reads a JSON file, parsing it into a
    Python dictionary.

    Parameters
    ----------
    filepath : str
        The path to the JSON configuration file.

    Returns
    -------
    dict
        A dictionary containing the configuration loaded from the file.
    """

    with open(file=filepath, mode="r") as f:
        return json.load(f)


import json
from typing import TypeAlias, Any
from pathlib import Path

from simple_loader import Loader

JsonlEntry: TypeAlias = dict[str, Any]

def _read_jsonl(input_file_path: Path) -> list[JsonlEntry]:
    jsonl_obj: list[JsonlEntry] = []
    if isinstance(input_file_path, str): input_file_path = Path(input_file_path)
    if not input_file_path.exists(): raise FileNotFoundError(f"{input_file_path} does not exit.")

    with Loader(f"Reading data from '{input_file_path}'"):
        with open(file=input_file_path, mode="r", encoding="utf-8") as f_in:
            for line in f_in:
                try: jsonl_obj.append(json.loads(line))
                except json.JSONDecodeError as e: raise e

        return jsonl_obj

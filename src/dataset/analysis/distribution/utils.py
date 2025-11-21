import json
from typing import TypeAlias, Any
from pathlib import Path

from .ui import rich_status


JsonlEntry: TypeAlias = dict[str, Any]

def read_jsonl(input_file_path: Path) -> list[JsonlEntry]:
    jsonl_obj: list[JsonlEntry] = []
    if isinstance(input_file_path, str):
        input_file_path = Path(input_file_path)
    if not input_file_path.exists():
        raise FileNotFoundError(f"{input_file_path} does not exit.")

    with (
        rich_status(description="Reading sorce file "),
        open(file=input_file_path, mode="r", encoding="utf-8") as f_in
    ):
        for line in f_in:
            try: jsonl_obj.append(json.loads(line))
            except json.JSONDecodeError as e: raise e

    return jsonl_obj

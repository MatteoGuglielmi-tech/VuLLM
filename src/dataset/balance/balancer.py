import json
import random
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias
from collections import defaultdict

from simple_loader import Loader

JsonlEntry: TypeAlias = dict[str,Any]


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(prog="Balance dataset")
    parser.add_argument( "--input_file_path", type=str, help="Absolute path to the raw dataset.")
    parser.add_argument( "--output_file_path", type=str, help="Absolute path to the output location where to save the balanced dataset to.")
    parser.add_argument( "--debug", type=bool, action=argparse.BooleanOptionalAction, help="Activate debug CLI logs")

    return parser

@dataclass
class Balancer:
    """
    Reads a JSONL file, creates a balanced dataset of vulnerable and non-vulnerable
    items, and then groups the result by the 'project' field in the output file.
    """

    input_file_path: Path
    output_file_path: Path
    meta_file_path: Path = field(init=False)

    def __post_init__(self):
        log_file_prefix = self.output_file_path.parent
        metafile = log_file_prefix / "metadata"
        metafile.mkdir(exist_ok=True)
        self.meta_file_path = metafile / "metadata_balancing.txt"

    def _read_jsonl(self, input_file_path:Path) -> list[JsonlEntry]:
        jsonl_obj: list[JsonlEntry] = []
        if not input_file_path.exists(): raise FileNotFoundError(f"{input_file_path} does not exit.")

        with Loader(f"Reading data from '{input_file_path}'"):
            with open(file=input_file_path, mode="r", encoding="utf-8") as f_in:
                for line in f_in:
                    try: jsonl_obj.append(json.loads(line))
                    except json.JSONDecodeError as e: raise e

        return jsonl_obj

    def _write_list_jsonl(self, output_file_path: Path, content: list[JsonlEntry]):
        with Loader(f"Writing file to '{output_file_path}'"):
            with open(file=output_file_path, mode='w', encoding='utf-8') as f_out:
                for item in content:
                    f_out.write(json.dumps(item) + '\n')

    def _write_dict_jsonl(self, output_file_path: Path, content: dict[str,list[JsonlEntry]]):
        log_messages: list[str] = ["--- Data Balancing and Grouping Log ---\n\n"]
        with Loader(f"Writing file to '{output_file_path}'"):
            with open(file=output_file_path, mode='w', encoding='utf-8') as f_out:
                for project, functions in content.items():
                    log_messages.append(f"  - Writing {len(functions)} functions for project '{project}'\n")
                    for item in functions:
                        f_out.write(json.dumps(item) + '\n')

        with open(file=self.meta_file_path, mode="w", encoding="utf-8") as fopen:
            fopen.writelines(log_messages)

    def _write_jsonl(self, output_file_path: Path, content: list[JsonlEntry]|dict[str,list[JsonlEntry]]):
        output_file_path.parent.mkdir(exist_ok=True)
        if isinstance(content, list): self._write_list_jsonl(output_file_path=output_file_path, content=content)
        if isinstance(content, dict): self._write_dict_jsonl(output_file_path=output_file_path, content=content)

    def _group_by_project(self, data: list[JsonlEntry]) -> dict[str, list[JsonlEntry]]:
        with Loader(f"Grouping the data by project..."):
            grouped_by_project = defaultdict(list)
            for item in data:
                project_name = item.get('project')
                if project_name: grouped_by_project[project_name].append(item)

        return grouped_by_project

    def balance_jsonl_data(self):
        jsonl_obj: list[JsonlEntry] = self._read_jsonl(input_file_path=self.input_file_path)
        vulnerable_func_list: list[JsonlEntry] = []
        non_vulnerable_func_list: list[JsonlEntry] = []

        for entry in jsonl_obj:
            if entry.get("target") is not None:
                if entry["target"] == 1:
                    vulnerable_func_list.append(entry)
                else:
                    non_vulnerable_func_list.append(entry)


        print(f"Found {len(vulnerable_func_list)} vulnerable functions (target: 1).")
        print(f"Found {len(non_vulnerable_func_list)} non-vulnerable functions (target: 0).")

        # check balance
        num_vulnerable: int = len(vulnerable_func_list)
        num_non_vulnerable: int = len(non_vulnerable_func_list)

        if num_vulnerable == 0:
            print("No vulnerable functions found.")
            return

        # assess how many non_vulnerable examples to sample
        if num_non_vulnerable < num_vulnerable:
            print("Warning: Not enough non-vulnerable functions to create a balanced dataset.")
            sampled_non_vulnerable_func_list = non_vulnerable_func_list
        else:
            print(f"Randomly sampling {num_vulnerable} non-vulnerable functions...")
            sampled_non_vulnerable_func_list = random.sample(non_vulnerable_func_list, num_vulnerable)

        # full data
        balanced_data: list[JsonlEntry] = vulnerable_func_list + sampled_non_vulnerable_func_list
        # arrange by project
        balanced_by_project: dict[str, list[JsonlEntry]] = self._group_by_project(data=balanced_data)
        print(f"Created a new balanced dataset with {len(balanced_data)} total functions.")
        self._write_jsonl(output_file_path=self.output_file_path, content=balanced_by_project)

        print("Process completed successfully.")


if __name__ == "__main__":
    args = get_parser().parse_args()

    balancer = Balancer(input_file_path=Path(args.input_file_path), output_file_path=Path(args.output_file_path))

    # Run the main function
    balancer.balance_jsonl_data()


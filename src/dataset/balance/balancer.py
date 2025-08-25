import json
import random
import argparse

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias
from collections import defaultdict
from tree_sitter import Tree, Node
from tqdm import tqdm

from simple_loader import Loader
from common.tree_sitter_parser import TreeSitterParser
from dataset.restructure.shared.proc_utils import is_cpp

JsonlEntry: TypeAlias = dict[str,Any]


def get_parser():
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
    tsp: TreeSitterParser = TreeSitterParser(language_name="ext_c")

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

    def remove_comments(self, code: str) -> str:
        tree: Tree = self.tsp.parse(code)
        comments: list[Node] = [node
            for node in self.tsp.traverse_tree(node=tree.root_node)
            if node.type == "comment"]
        for comment_node in sorted(comments, key=lambda c: c.start_byte, reverse=True):
            code = code[:comment_node.start_byte] + code[comment_node.end_byte:]

        return code.lstrip()

    def _filter_cpp_out(self, data: list[JsonlEntry]):
        c_entries: list[JsonlEntry] = []
        for entry in tqdm(iterable=data, desc="🚧 Filtering Cpp functions out 🚧."):
            func_str: str|None = entry.get("func")
            if not func_str: continue
            code: str = self.remove_comments(code=func_str)
            if not code.strip(): continue # empty or comments entry
            if is_cpp(code=code): continue # if cpp, skip

            c_entries.append(entry) # if all checks passed

        return c_entries

    def balance_jsonl_data(self):
        print("🚀 Starting the build process...🚀")

        non_vulnerable_func_list: list[JsonlEntry] = []
        vulnerable_func_list: list[JsonlEntry] = []

        # read in jsonl
        jsonl_obj: list[JsonlEntry] = self._read_jsonl(input_file_path=self.input_file_path)
        # retain only c functions
        c_entries: list[JsonlEntry] = self._filter_cpp_out(data=jsonl_obj)
        # balance
        for entry in c_entries:
            if entry.get("target") is not None:
                if entry["target"] == 1: vulnerable_func_list.append(entry)
                else: non_vulnerable_func_list.append(entry)

        print(f"Found {len(vulnerable_func_list)} vulnerable functions (target: 1).")
        print(f"Found {len(non_vulnerable_func_list)} non-vulnerable functions (target: 0).")

        # check balance
        num_vulnerable: int = len(vulnerable_func_list)
        num_non_vulnerable: int = len(non_vulnerable_func_list)

        if num_vulnerable == 0:
            print("No vulnerable functions found.")
            return
        if num_non_vulnerable < num_vulnerable:
            print("Warning: Not enough non-vulnerable functions to create a balanced dataset.")
            sampled_non_vulnerable_func_list = non_vulnerable_func_list
        else:
            print(f"Randomly sampling {num_vulnerable} non-vulnerable functions...")
            sampled_non_vulnerable_func_list = random.sample(non_vulnerable_func_list, num_vulnerable)

        # full data
        balanced_data: list[JsonlEntry] = vulnerable_func_list + sampled_non_vulnerable_func_list
        print(f"🎉 Created a new balanced dataset with {len(balanced_data)} total functions. 🎉")

        # arrange by project and save it
        balanced_by_project: dict[str, list[JsonlEntry]] = self._group_by_project(data=balanced_data)
        self._write_jsonl(output_file_path=self.output_file_path, content=balanced_by_project)

        print("✅ Process completed successfully! ✅")


if __name__ == "__main__":
    args = get_parser().parse_args()
    balancer = Balancer(input_file_path=Path(args.input_file_path), output_file_path=Path(args.output_file_path))
    balancer.balance_jsonl_data()


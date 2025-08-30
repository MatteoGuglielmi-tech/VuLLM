import json
import argparse
import logging
import pandas as pd
import tiktoken

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast
from collections import defaultdict
from common.common_typedef import Captures
from tree_sitter import Tree, Node
from tqdm import tqdm

from logging_config import setup_logger
from simple_loader import Loader
from common.tree_sitter_parser import TreeSitterParser
from dataset.restructure.shared.proc_utils import is_cpp

JsonlEntry: TypeAlias = dict[str, Any]

setup_logger()
logger = logging.getLogger(name=__name__)


def get_parser():
    parser = argparse.ArgumentParser(prog="Balance dataset")
    parser.add_argument("--input_file_path", type=str, help="Absolute path to the raw dataset.")
    parser.add_argument("--output_file_path", type=str, help="Absolute path to the output location where to save the balanced dataset to.")
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
    df: pd.DataFrame = field(init=False)
    meta_file_path: Path = field(init=False)
    tsp: TreeSitterParser = field(init=False)
    tokenizer: tiktoken.Encoding = field(init=False)

    # defaults
    balance_tolerance: float = .9
    random_state: int = 42
    n_bins: int = 20

    def __post_init__(self):
        log_file_prefix = self.output_file_path.parent
        metafile = log_file_prefix / "metadata"
        metafile.mkdir(exist_ok=True)
        self.meta_file_path = metafile / "metadata_balancing.txt"
        self.tsp: TreeSitterParser = TreeSitterParser(language_name="ext_c")
        # tokenizer like gpt-4's to approximate LLM token count
        self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")

    def _read_jsonl(self, input_file_path: Path) -> list[JsonlEntry]:
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
            with open(file=output_file_path, mode="w", encoding="utf-8") as f_out:
                for item in content:
                    f_out.write(json.dumps(item) + "\n")

    def _write_dict_jsonl(self, output_file_path: Path, content: dict[str, list[JsonlEntry]]):
        log_messages: list[str] = ["--- Data Balancing and Grouping Log ---\n\n"]
        with Loader(f"Writing file to '{output_file_path}'"):
            with open(file=output_file_path, mode="w", encoding="utf-8") as f_out:
                for project, functions in content.items():
                    log_messages.append(f"  - Writing {len(functions)} functions for project '{project}'\n")
                    for item in functions:
                        f_out.write(json.dumps(item) + "\n")

        with open(file=self.meta_file_path, mode="w", encoding="utf-8") as fopen:
            fopen.writelines(log_messages)

    def _write_jsonl(self, output_file_path: Path, content: list[JsonlEntry] | dict[str, list[JsonlEntry]]):
        output_file_path.parent.mkdir(exist_ok=True)
        if isinstance(content, list): self._write_list_jsonl(output_file_path=output_file_path, content=content)
        if isinstance(content, dict): self._write_dict_jsonl(output_file_path=output_file_path, content=content)

    def _group_by_project(self, data: list[JsonlEntry]) -> dict[str, list[JsonlEntry]]:
        with Loader(f"Grouping data by project..."):
            grouped_by_project = defaultdict(list)
            for item in data:
                project_name = item.get("project")
                if project_name: grouped_by_project[project_name].append(item)

        return grouped_by_project

    def _remove_comments(self, code: str) -> str:
        tree: Tree = self.tsp.parse(code)
        comments: list[Node] = [
            node for node in self.tsp.traverse_tree(node=tree.root_node)
            if node.type == "comment"
        ]
        for comment_node in sorted(comments, key=lambda c: c.start_byte, reverse=True):
            code = code[: comment_node.start_byte] + code[comment_node.end_byte :]

        return code.lstrip()

    def _filter_cpp_out(self, data: list[JsonlEntry]):
        c_entries: list[JsonlEntry] = []
        for entry in tqdm(iterable=data, desc="🚧 Filtering Cpp functions out 🚧."):
            func_str: str | None = entry.get("func")
            if not func_str: continue
            code: str = self._remove_comments(code=func_str)
            if not code.strip(): continue  # empty or comments entry
            if is_cpp(code=code): continue  # if cpp, skip
            entry["func"] = code

            c_entries.append(entry)  # if all checks passed

        return c_entries

    def _get_cyclomatic_complexity(self, code_string: str) -> int:
        query_str = """[
            (if_statement)
            (while_statement)
            (do_statement)
            (for_statement)
            (case_statement)

            ; Expression-level decisions
            (conditional_expression)
            (binary_expression operator: "&&" )
            (binary_expression operator: "||" )

            ; custom
            (smartlist_foreach_statement)
            (control_flow_macro_statement)
          ] @decision
        """

        tree = self.tsp.parse(code=code_string)
        captures: Captures = self.tsp.run_query_on_tree(tree=tree, query_str=query_str)
        decisions: list[Node] = captures.get("decision", [])

        if not decisions: return 0
        else: return (1 + len(decisions))

    def _get_token_count(self, code_string: str) -> int:
        return len(self.tokenizer.encode(code_string))

    def _extend_with_metadata(self, code_string: str) -> dict[str, int]:
        complexity: int = self._get_cyclomatic_complexity(code_string=code_string)
        tokens: int = self._get_token_count(code_string=code_string)

        return { "complexity": complexity, "tokens": tokens }

    def _extract_features(self):
        jsonl_obj: list[JsonlEntry] = self._read_jsonl(input_file_path=self.input_file_path)
        c_entries: list[JsonlEntry] = self._filter_cpp_out(data=jsonl_obj) # filtering
        self.df = pd.DataFrame(data=c_entries)
        self.df = self.df.drop(labels=[ "hash", "size", "message"], axis=1)

        # features computation
        tqdm.pandas(desc="Calculating Cyclomatic Complexity")
        self.df['complexity'] = self.df['func'].progress_apply(self._get_cyclomatic_complexity)
        tqdm.pandas(desc="Calculating Token Count")
        self.df['token_count'] = self.df['func'].progress_apply(self._get_token_count)
        logger.info("Feature extraction complete.")

    def _perform_fallback_sampling(self, df_vuln: pd.DataFrame, df_non_vuln: pd.DataFrame):
        logger.warning("\n⚠️  Distribution matching failed. Using fallback: selecting 'easiest' samples.")
        num_to_sample = len(df_vuln)
        df_non_vuln_sorted = df_non_vuln.sort_values(by=['complexity', 'token_count'])
        df_non_vuln_sampled = df_non_vuln_sorted.head(num_to_sample)
        self._assemble_final_df(df_vuln, df_non_vuln_sampled)

    def _assemble_final_df(self, df_vuln: pd.DataFrame, df_non_vuln_sampled: pd.DataFrame):
        combined_df = pd.concat([df_vuln, df_non_vuln_sampled])
        cols_to_drop = ['complexity', 'token_count', 'complexity_bin', 'token_bin']
        existing_cols_to_drop = [col for col in cols_to_drop if col in combined_df.columns]
        self.df_balanced = cast(pd.DataFrame, combined_df.drop(columns=existing_cols_to_drop))
        # self.df_balanced = self.df_balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.df_balanced = self.df_balanced.sort_values(by='project').reset_index(drop=True)

    def _perform_stratified_sampling(self):
        """Performs the core balancing logic."""
        if self.df is None: raise ValueError("Data not loaded. Run _extract_features() first.")

        logger.info("Performing stratified sampling...")
        df_vuln = cast(pd.DataFrame, self.df[self.df['target'] == 1].copy())
        df_non_vuln = cast(pd.DataFrame, self.df[self.df['target'] == 0].copy())

        logger.debug(f"Found {len(df_vuln)} vulnerable functions (target: 1).")
        logger.debug(f"Found {len(df_non_vuln)} non-vulnerable functions (target: 0).")

        try:
            # quantile cut: divide into a set number of equal-sized groups.
            df_vuln['complexity_bin'], complexity_bins = pd.qcut(df_vuln['complexity'], q=self.n_bins, labels=False, retbins=True, duplicates='drop')
            df_vuln['token_bin'], token_bins = pd.qcut(df_vuln['token_count'], q=self.n_bins, labels=False, retbins=True, duplicates='drop')

            # value cut: divides based on pre-defined bins
            # apply same bin edges to non-vulnerable data
            df_non_vuln['complexity_bin'] = pd.cut(df_non_vuln['complexity'], bins=complexity_bins, labels=False, include_lowest=True)
            df_non_vuln['token_bin'] = pd.cut(df_non_vuln['token_count'], bins=token_bins, labels=False, include_lowest=True)

            df_non_vuln = df_non_vuln.dropna(subset=['complexity_bin', 'token_bin']) # type: ignore
            df_non_vuln['complexity_bin'] = df_non_vuln['complexity_bin'].astype(int)
            df_non_vuln['token_bin'] = df_non_vuln['token_bin'].astype(int)
            # count targets in each bin
            target_counts = df_vuln.groupby(['complexity_bin', 'token_bin']).size()

            sampled_indices = []
            for (c_bin, t_bin), count in target_counts.items(): # type: ignore
                non_vuln_in_bin = df_non_vuln[
                    (df_non_vuln["complexity_bin"] == c_bin) & (df_non_vuln["token_bin"] == t_bin)
                ]
                num_to_sample = min(int(count), len(non_vuln_in_bin))
                sampled_indices.extend(non_vuln_in_bin.sample(n=num_to_sample, random_state=self.random_state).index)

            df_non_vuln_sampled = df_non_vuln.loc[sampled_indices]

            # check if really balanced
            num_vuln = len(df_vuln)
            num_sampled = len(df_non_vuln_sampled)
            if num_sampled < (num_vuln * self.balance_tolerance):
                logger.warning(f"\n⚠️  Distribution matching produced a poorly balanced set ({num_sampled}/{num_vuln}).")
                self._perform_fallback_sampling(df_vuln, df_non_vuln)
            else:
                # If the balance is acceptable, proceed as normal.
                self._assemble_final_df(df_vuln=df_vuln, df_non_vuln_sampled=df_non_vuln_sampled)

        except ValueError:
            logger.warning("\n⚠️  Distribution matching failed due to incompatible data for binning.")
            self._perform_fallback_sampling(df_vuln, df_non_vuln)

        logger.info("Balancing complete.")
        if self.df_balanced is not None:
            print("Final balanced dataset distribution:")
            print(self.df_balanced['target'].value_counts())
        else:
            print("Balancing resulted in an empty dataset.")

    def save_balanced_dataset(self, output_path: str):
        """Saves the balanced dataset to a new jsonl file."""
        if self.df_balanced is None: raise ValueError("No balanced dataset to save. Run process() first.")
        self.df_balanced.to_json(output_path, orient='records', lines=True) # to jsonl

        logger.info(f"Balanced dataset saved to {output_path}")

    def process(self):
        """Runs the full pipeline: load, extract features, and balance."""
        self._extract_features()
        self._perform_stratified_sampling()


if __name__ == "__main__":
    args = get_parser().parse_args()
    balancer = Balancer(input_file_path=Path(args.input_file_path), output_file_path=Path(args.output_file_path), n_bins=2)
    balancer.process()


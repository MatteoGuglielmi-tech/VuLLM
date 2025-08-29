import json
import argparse
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias
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


plt.style.use('fivethirtyeight')

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
    meta_file_path: Path = field(init=False)
    tsp: TreeSitterParser = TreeSitterParser(language_name="ext_c")

    def __post_init__(self):
        log_file_prefix = self.output_file_path.parent
        metafile = log_file_prefix / "metadata"
        metafile.mkdir(exist_ok=True)
        self.meta_file_path = metafile / "metadata_balancing.txt"

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
        with Loader(f"Grouping the data by project..."):
            grouped_by_project = defaultdict(list)
            for item in data:
                project_name = item.get("project")
                if project_name: grouped_by_project[project_name].append(item)

        return grouped_by_project

    def remove_comments(self, code: str) -> str:
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
            code: str = self.remove_comments(code=func_str)
            if not code.strip(): continue  # empty or comments entry
            if is_cpp(code=code): continue  # if cpp, skip
            entry["func"] = code

            c_entries.append(entry)  # if all checks passed

        return c_entries

    def _get_cyclomatic_complexity(self, tree: Tree) -> int:
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

        captures: Captures = self.tsp.run_query_on_tree(tree=tree, query_str=query_str)
        decisions: list[Node] = captures.get("decision", [])

        return (1 + len(decisions))

    def _get_token_count(self, tree: Tree) -> int:
        """Counts the number of tokens in the code by counting tree leaves."""

        token_count: int = 0
        for node in self.tsp.traverse_tree(node=tree.root_node):
            if not node.children: token_count += 1

        return token_count

    def _extend_with_metadata(self, tree: Tree) -> dict[str, int]:
        complexity: int = self._get_cyclomatic_complexity(tree=tree)
        tokens: int = self._get_token_count(tree=tree)

        return { "complexity": complexity, "tokens": tokens }

    def analyze_and_select_by_token_count(
        self,
        non_vulnerable_entries: list[dict],
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
    ) -> list[JsonlEntry]:


        entries_with_metadata: list[JsonlEntry] = []
        for entry in non_vulnerable_entries:
            code: str = entry["func"]
            tree: Tree = self.tsp.parse(code=code)
            extended_entry: JsonlEntry = entry.copy() # self._extend_with_metadata(tree)
            extended_entry["metrics"] = self._extend_with_metadata(tree=tree)
            entries_with_metadata.append(extended_entry)

        if not entries_with_metadata: raise ValueError("No entries to analyze")
        token_counts = [e["metrics"]["tokens"] for e in entries_with_metadata]
        s: pd.Series = pd.Series(data=token_counts, dtype=int)

        # -- save collective stats --
        stats_df: pd.DataFrame = s.describe().to_frame().round(3)
        stats_df.columns = ['value'] # Rename column for clarity
        target_location: Path = Path(__file__).parent.parent / "analysis/assets/token_count_distribution.csv"
        for p in target_location.parts[1:-1]: Path(p).mkdir(exist_ok=True) # create "../analysis/assets/"
        stats_df.to_csv(path_or_buf=target_location, encoding="utf-8", lineterminator="\n")

        print("\n--- Distribution Statistics ---")
        print(stats_df)
        print("-----------------------------\n")

        # -- plots --
        df = pd.DataFrame({"Values": s, "Category": "Token count"})

        fig, ax = plt.subplots(figsize=(10,15))
        ax.set_facecolor(color='darkgray')

        sns.boxplot(
            x="Category", y="Values", data=df, ax=ax,
            width=0.5,
            color="#a9d6e5",
            linewidth=1.5,
            showfliers=False # outlayers will be reprsented by stripplot
        )
        sns.stripplot( x="Category", y="Values", data=df, ax=ax,
            color="#456882",  # A darker, contrasting blue
            alpha=.8,  # Use transparency to show density
            jitter=0.2,  # Spread points horizontally
            size=6,
            edgecolor="black",
            linewidth=0.4
        )

        # statistics
        mean_val = df['Values'].mean()
        median_val = df['Values'].median()
        std_val = df['Values'].std()
        max_val = df['Values'].max()
        min_val = df['Values'].min()

        # Plot Mean and Median with distinct, clear markers
        ax.plot([0], [median_val], marker='o', color='#ff6b6b', markersize=10, linestyle='None', label='Median') # type: ignore
        ax.plot([0], [mean_val], marker='D', color='#ff6b6b', markersize=8, linestyle='None', label='Mean') # type: ignore

        # create title and subtitle
        fig.text(0.1, 0.96, 'Distribution of Token Counts in Non-Vulnerable Functions',
                fontsize=13, fontweight='bold', ha='left')
        fig.text(0.1, 0.92, 'Analysis shows a right-skewed distribution with several high-value outliers',
                fontsize=13, ha='left', color='gray')

        stats_text = (f"Mean: {mean_val:.2f}\n"
              f"Median: {median_val:.2f}\n"
              f"Std Dev: {std_val:.2f}\n"
              f"Max / Min: {max_val:.0f} / {min_val:.0f}")

        # place text box
        ax.text(0.95, 0.85, stats_text, transform=ax.transAxes, fontsize=13,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='darkgray', lw=1, alpha=0.9))

        ax.annotate('Mean', xy=(0, mean_val), xycoords='data', # type: ignore
                    xytext=(0.35, mean_val), textcoords='data', # type: ignore
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.1", color='black', lw=1.5),
                    fontsize=10, ha='left', va='center')

        ax.annotate('Median', xy=(0, median_val), xycoords='data', # type: ignore
                    xytext=(0.35, median_val), textcoords='data', # type: ignore
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color='black', lw=1.5),
                    fontsize=10, ha='left', va='center')

        ax.set_xlabel('')
        ax.set_ylabel('Token Count', fontsize=14, fontweight='bold')
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0) # Remove x-axis tick marks
        ax.set_ylim(min_val - (min_val * 0.1), max_val + (max_val * 0.1))

        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_minor_locator(MultipleLocator(50))

        # Add a source line at the bottom
        fig.text(0.1, 0.02, 'Source: Code analysis dataset',
                 fontsize=9, color='gray', ha='left')

        # refine grid and spines
        ax.grid(True, which="minor", linestyle=":", alpha=0.6, axis="y", color="gray")
        ax.grid(True, which="major", linestyle="-", alpha=0.8, axis="y", color="gray")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)

        plt.tight_layout(rect=[0, 0.05, 1, 0.9]) # type: ignore

        target_location = Path(str(target_location).replace(".csv", ".png"))
        plt.savefig(target_location)
        plt.show()

        logger.info("Distribution plot saved to token_distribution.png")

        lower_bound: float = s.quantile(lower_quantile)
        upper_bound: float = s.quantile(upper_quantile)

        logger.info(f"Dynamic lower bound (quantile={lower_quantile}): {lower_bound:.0f} tokens")
        logger.info(f"Dynamic upper bound (quantile={upper_quantile}): {upper_bound:.0f} tokens")

        # filter outlayers
        filtered_entries: list[JsonlEntry] = [e for e in entries_with_metadata if lower_bound <= e["metrics"]["tokens"] <= upper_bound]
        # sort by complexity, token count as tie-breaker
        filtered_entries.sort(key=lambda e: (e["metrics"]["complexity"], e["metrics"]["tokens"]))

        logger.info(f"\nOriginal number of entries: {len(non_vulnerable_entries)}")
        logger.info(f"Filtered number of entries: {len(filtered_entries)}")

        return filtered_entries


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

        logger.info(f"Found {len(vulnerable_func_list)} vulnerable functions (target: 1).")
        logger.info(f"Found {len(non_vulnerable_func_list)} non-vulnerable functions (target: 0).")

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
            sampled_non_vulnerable_func_list = self.analyze_and_select_by_token_count(
                non_vulnerable_entries=non_vulnerable_func_list,
            )

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

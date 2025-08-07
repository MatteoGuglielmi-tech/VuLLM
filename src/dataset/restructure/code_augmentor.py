import os
import json
from dataclasses import dataclass
from tree_sitter import Tree, Node, Query, QueryCursor

from .shared.log import logger
from .shared.animate import Loader
from .shared.tree_sitter_parser import TreeSitterParser
from .llm_clients.base import DescriptionGenerator


@dataclass
class CodeAugmentor:
    """A unified pipeline to pre-process and enrich a code dataset in a single pass."""

    metadata_filepath: str
    description_generator: DescriptionGenerator

    def __post_init__(self):
        """Initializes the Gemini model and loads the metadata map."""

        # self.metadata_map: dict[str,dict[str,str]] = self._load_and_map_metadata()
        self.ts_parsers:dict[str,TreeSitterParser] = { "c": TreeSitterParser("c"), "cpp": TreeSitterParser("cpp") }

    # def _load_and_map_metadata(self) -> dict[str, dict[str,str]]:
    #     """Loads the metadata file and creates a map from commit_id to bug_info."""
    #
    #     if not os.path.exists(path=self.metadata_filepath):
    #         logger.warning(f"Metadata file not found at {self.metadata_filepath}. Proceeding without it.")
    #         return {}
    #
    #     _, ext = os.path.splitext(p=self.metadata_filepath)
    #     if ext != ".jsonl":
    #         raise ValueError("metadata file not in valid JSONL format. Abort.")
    #
    #     metadata_map: dict[str,dict[str,str]] = {}
    #     with Loader(
    #         desc_msg=f"Loading metadata from {self.metadata_filepath}...",
    #         end_msg=f"✅ Done!\nCreated a map for {len(metadata_map)} unique commit IDs.",
    #     ):
    #         with open(file=self.metadata_filepath, mode="r", encoding="utf-8") as f:
    #             for line in f:
    #                 if line.strip():
    #                     entry: dict[str, str] = json.loads(line)
    #                     commit_id: str|None = entry.get("commit_id")
    #                     if commit_id:
    #                         metadata_map[commit_id] = {
    #                             "bug_info": entry.get("bug_info", ""),
    #                             "cwe": entry.get("CWE", ""),
    #                             "cve": entry.get("CVE", ""),
    #                         }
    #
    #     return metadata_map

    def _get_code_metrics(self, code: str, lang: str) -> dict[str, int]:
        """Calculates code metrics for a given function using tree-sitter."""

        ts_parser:TreeSitterParser|None = self.ts_parsers.get(lang)
        if not ts_parser:
            return {"parameter_count": -1, "cyclomatic_complexity": -1}

        tree:Tree = ts_parser.parse(code=code)
        param_count:int = 0

        query:Query = Query(
            ts_parser.language,
            """
            (parameter_list) @params
            [
              (if_statement)
              (while_statement)
              (for_statement)
              (case_statement)
            ] @decision
            """
        )
        captures:dict[str,list[Node]] = QueryCursor(query=query).captures(tree.root_node)

        params:list[Node] = captures.get("params", [])
        decisions:list[Node] = captures.get("decision", [])

        if params:
            param_count:int = len([child for child in params[0].children if child.type == "parameter_declaration"])

        cyclomatic_complexity:int = 1 + len(decisions)

        return { "parameter_count": param_count, "cyclomatic_complexity": cyclomatic_complexity }

    def _gen_batch_func_descr(self, snippets_to_describe: list[str]) -> list[str]:
        try:
            return self.description_generator.generate_batch_descriptions(snippets_to_describe)
        except Exception as e:
            logger.error(f"Batch LLM func generation failed: {e}")
            descriptions = ["N/A"] * len(snippets_to_describe) # fallback
            return descriptions

    def _gen_batch_cwe_descr(self, cwes_to_describe: list[str]) -> list[str]:
        try:
            return self.description_generator.generate_batch_cwe_descriptions(cwes_to_describe)
        except Exception as e:
            logger.error(f"Batch LLM CWE generation failed: {e}")
            descriptions = ["N/A"] * len(cwes_to_describe) # fallback
            return descriptions

    def enrich_dataset_batch(self, entries: list[dict]) -> list[dict]:

        # collect batch
        snippets_to_describe = [ e["func"] for e in entries if "func" in e and not e["func"].startswith(("error:", "skipped:"))]
        cwes_to_describe = [e["cwe"] for e in entries if "cwe" in e and not e["func"].startswith(("error:", "skipped:"))]

        if not snippets_to_describe:
            return entries

        # -- generate all descriptions in a single, efficient batch call. --
        func_descriptions = self._gen_batch_func_descr(snippets_to_describe=snippets_to_describe)
        cwe_descriptions = self._gen_batch_cwe_descr(cwes_to_describe=cwes_to_describe)

        # -- merge --
        desc_iterator = iter(func_descriptions)
        cwe_desc_iterator = iter(cwe_descriptions)
        for entry in entries:
            if "func" in entry and not entry["func"].startswith(("error:", "skipped:")):
                # -- add other enrichments first --
                metrics = self._get_code_metrics(code=entry["func"], lang=entry.get("language", "c"))
                entry.update(metrics)
                # -- add the generated description --
                entry["function_description"] = next(desc_iterator, "N/A")
                # -- add vulnerability description --
                entry["vulnerability_description"] = next(cwe_desc_iterator, "N/A")

        return entries

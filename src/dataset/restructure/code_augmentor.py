import os
import json
from dataclasses import dataclass
from tree_sitter import Tree, Node, Query, QueryCursor

from .log import logger
from .animate import Loader
from .gemini_client import Gemini
from .tree_sitter_parser import TreeSitterParser


@dataclass
class CodeAugmentor:
    """A unified pipeline to pre-process and enrich a code dataset in a single pass."""

    metadata_filepath: str

    def __post_init__(self):
        """Initializes the Gemini model and loads the metadata map."""

        self.gemini: Gemini = Gemini(model_name="gemini-1.5-flash")
        self.metadata_map: dict[str,dict[str,str]] = self._load_and_map_metadata()
        self.ts_parsers:dict[str,TreeSitterParser] = { "c": TreeSitterParser("c"), "cpp": TreeSitterParser("cpp") }

    def _load_and_map_metadata(self) -> dict[str, dict[str,str]]:
        """Loads the metadata file and creates a map from commit_id to bug_info."""

        if not os.path.exists(path=self.metadata_filepath):
            logger.warning(f"Metadata file not found at {self.metadata_filepath}. Proceeding without it.")
            return {}

        _, ext = os.path.splitext(p=self.metadata_filepath)
        if ext != ".jsonl":
            raise ValueError("metadata file not in valid JSONL format. Abort.")

        metadata_map: dict[str,dict[str,str]] = {}
        with Loader(desc_msg=f"Loading metadata from {self.metadata_filepath}..."):
            with open(file=self.metadata_filepath, mode="r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry: dict[str, str] = json.loads(line)
                        commit_id: str|None = entry.get("commit_id")
                        if commit_id:
                            metadata_map[commit_id] = {
                                "bug_info": entry.get("bug_info", ""),
                                "cwe": entry.get("CWE", ""),
                                "cve": entry.get("CVE", ""),
                            }

            logger.info(f"Created a map for {len(metadata_map)} unique commit IDs.")

        return metadata_map

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

    def enrich_entry(self, entry: dict) -> dict[str,str]:
        """Main orchestration method to add all enrichments to a single data entry."""

        func_code:str|None = entry.get("func")
        lang:str = entry.get("language", "c")

        if not func_code or func_code.startswith("error:") or func_code.startswith("skipped:"):
            return entry

        # --- 1. Add Vulnerability Description ---
        commit_id:str|None = entry.get("commit_id")
        if commit_id and commit_id in self.metadata_map:
            entry["vulnerability_description"] = self.metadata_map[commit_id]

        # --- 2. Add Code Metrics ---
        metrics:dict[str,int] = self._get_code_metrics(code=func_code, lang=lang)
        entry.update(metrics)

        # --- 3. Generate Function Description (API Call) ---
        try:
            func_desc:str = self.gemini.generate_description(func_str=func_code)
            entry["function_description"] = func_desc
        except Exception as e:
            logger.error(f"Gemini API failed for an entry: {e}")
            entry["function_description"] = "N/A"

        return entry


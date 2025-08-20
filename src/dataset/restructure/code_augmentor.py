import logging
from dataclasses import dataclass
from tree_sitter import Tree, Node, Query, QueryCursor

from .llm_clients.base import DescriptionGenerator

from ...common.tree_sitter_parser import TreeSitterParser

logger = logging.getLogger(name=__name__)


@dataclass
class CodeAugmentor:
    """A unified pipeline to pre-process and enrich a code dataset in a single pass.

    This class orchestrates the augmentation of a code dataset by generating
    natural language descriptions, calculating code metrics, and adding other
    relevant metadata. It is designed to work on batches of data for
    efficiency, making it suitable for large-scale data processing tasks. The
    primary entry point is the `enrich_dataset_batch` method.

    Attributes
    ----------
    metadata_filepath : str
        The file path to the dataset's metadata. (Note: This attribute is
        declared but not used in the provided methods, suggesting it might be
        used in other parts of the class not shown).
    description_generator : DescriptionGenerator
        An object responsible for communicating with a Language Model (LLM)
        to generate natural language descriptions for code snippets and CWEs.
    ts_parsers : dict[str, TreeSitterParser]
        A dictionary mapping programming language names (e.g., "c", "cpp")
        to their corresponding TreeSitterParser instances. Initialized in
        `__post_init__`.

    Methods
    -------
    enrich_dataset_batch(entries: list[dict]) -> list[dict]
        Enriches a list of data entries with generated descriptions and code
        metrics.
    """

    metadata_filepath: str
    description_generator: DescriptionGenerator

    def __post_init__(self):
        """Initializes TreeSitterParser for C and C++ languages."""

        self.ts_parsers: dict[str, TreeSitterParser] = {
            "c": TreeSitterParser("c"),
            "cpp": TreeSitterParser("cpp"),
            "ext_c": TreeSitterParser("ext_c"),
        }

    def _get_code_metrics(self, code: str, lang: str) -> dict[str, int]:
        """Calculates code metrics for a given function using tree-sitter.

        This method computes the number of parameters and the cyclomatic
        complexity of a given code snippet.

        Parameters
        ----------
        code : str
            The source code of the function to analyze.
        lang : str
            The programming language of the code snippet (e.g., "c", "cpp").

        Returns
        -------
        dict[str, int]
            A dictionary containing the calculated metrics:
            - "parameter_count": The number of function parameters.
            - "cyclomatic_complexity": The cyclomatic complexity score.
            Returns -1 for metrics if the language parser is not available.

        Notes
        -----
        Cyclomatic complexity is calculated as `1 + D`, where `D` is the number
        of decision points (if, while, for, case statements).
        """

        ts_parser: TreeSitterParser | None = self.ts_parsers.get(lang)
        if not ts_parser:
            return {"parameter_count": -1, "cyclomatic_complexity": -1}

        tree: Tree = ts_parser.parse(code=code)
        param_count: int = 0

        query: Query = Query(
            ts_parser.language,
            """
            (parameter_list) @params
            [
              (if_statement)
              (while_statement)
              (for_statement)
              (case_statement)
            ] @decision
            """,
        )
        captures: dict[str,list[Node]] = QueryCursor(query=query).captures(tree.root_node)

        params: list[Node] = captures.get("params", [])
        decisions: list[Node] = captures.get("decision", [])

        if params:
            param_count: int = len([child for child in params[0].children if child.type == "parameter_declaration"])

        cyclomatic_complexity: int = 1 + len(decisions)

        return { "parameter_count": param_count, "cyclomatic_complexity": cyclomatic_complexity }

    def _gen_batch_func_descr(self, snippets_to_describe: list[str]) -> list[str]:
        """Generates function descriptions for a batch of code snippets.

        Wrapper around the DescriptionGenerator to handle batch generation and errors.

        Parameters
        ----------
        snippets_to_describe : list[str]
            A list of code snippets requiring a natural language description.

        Returns
        -------
        list[str]
            A list of generated descriptions. If generation fails, a list
            of "N/A" strings is returned as a fallback.
        """

        try:
            return self.description_generator.generate_batch_descriptions(c_code_batch=snippets_to_describe)
        except Exception as e:
            logger.error(f"Batch LLM func generation failed: {e}")
            descriptions = ["N/A"] * len(snippets_to_describe)  # fallback
            return descriptions

    def _gen_batch_cwe_descr(self, cwes_to_describe: list[str]) -> list[str]:
        """Generates descriptions for a batch of CWE identifiers.

        Wrapper around the DescriptionGenerator to handle batch generation and errors.

        Parameters
        ----------
        cwes_to_describe : list[str]
            A list of CWE identifiers (e.g., "CWE-125") requiring an explanation.

        Returns
        -------
        list[str]
            A list of generated CWE explanations. If generation fails, a list
            of "N/A" strings is returned as a fallback.
        """

        try:
            return self.description_generator.generate_batch_cwe_descriptions(cwe_ids_batch=cwes_to_describe)
        except Exception as e:
            logger.error(f"Batch LLM CWE generation failed: {e}")
            descriptions = ["N/A"] * len(cwes_to_describe)  # fallback
            return descriptions

    def enrich_dataset_batch(self, entries: list[dict]) -> list[dict]:
        """Enriches a batch of data entries with generated descriptions and metrics.

        This is the main processing method of the class. It takes a list of
        dataset entries, extracts code and CWEs, generates descriptions for them
        in a batch, calculates code metrics, and merges all this new information
        back into the original entries.

        Parameters
        ----------
        entries : list[dict]
            A list of dictionaries, where each dictionary represents a data
            point. Each entry is expected to have a "func" key with the code,
            a "cwe" key, and optionally a "language" key.

        Returns
        -------
        list[dict]
            The list of entries, updated in-place with new keys:
            - `parameter_count`: Number of function parameters.
            - `cyclomatic_complexity`: The code's cyclomatic complexity.
            - `function_description`: Natural language description of the function.
            - `vulnerability_description`: Natural language explanation of the CWE.
        """

        # collect batch
        snippets_to_describe = [
            e["func"] for e in entries
            if "func" in e and not e["func"].startswith(("error:", "skipped:"))
        ]
        cwes_to_describe = [
            e["cwe"] for e in entries
            if "cwe" in e and not e["func"].startswith(("error:", "skipped:"))
        ]

        if not snippets_to_describe: return entries

        # -- generate all descriptions in a single, efficient batch call. --
        func_descriptions = self._gen_batch_func_descr(snippets_to_describe=snippets_to_describe)
        cwe_descriptions = self._gen_batch_cwe_descr(cwes_to_describe=cwes_to_describe)

        # -- merge --
        desc_iterator = iter(func_descriptions)
        cwe_desc_iterator = iter(cwe_descriptions)
        keys_to_remove = {"commit_id", "hash", "message", "size"}

        for entry in entries:
            if "func" in entry and not entry["func"].startswith(("error:", "skipped:")):
                # -- add other enrichments first --
                metrics = self._get_code_metrics(code=entry["func"], lang=entry.get("language", "ext_c"))
                entry.update(metrics)
                # -- add the generated description --
                entry["function_description"] = next(desc_iterator, "N/A")
                # -- add vulnerability description --
                entry["vulnerability_description"] = next(cwe_desc_iterator, "N/A")

                for key in keys_to_remove:
                    entry.pop(key, None)

        return entries

import os
import json
from dataclasses import dataclass
from tqdm import tqdm
from tree_sitter import Tree

from .shared.proc_utils import ( load_config, is_cpp,
    get_refactored_code, pause_exec, read_file, read_lines, write2file)
from .code_sanitizer import CodeSanitizer
from .interleaved_block_fixer import InterleavedBlockFixer
from .code_foundry import CodeFoundry
from .code_augmentor import CodeAugmentor

from .shared.tree_sitter_parser import TreeSitterParser, C_LANGUAGE
from .shared.log import logger
from .llm_clients.gemini_describer import GeminiClient
from .llm_clients.llama_describer import LlamaCodeDescriber


@dataclass
class Vulcan:
    """The master pipeline orchestrator. Vulcan forges raw code snippets into
    clean, repaired, and enriched dataset entries.
    """

    USE_LOCAL_MODEL: bool = True

    def __post_init__(self):
        self.processed_data: dict[str,str]= {}

        # adjust paths
        self._setup_config()

        # Initialize the pipeline components
        self.sanitizer = CodeSanitizer()
        self.interleaved_fixer = InterleavedBlockFixer()
        self.foundry = CodeFoundry(ts_lang=C_LANGUAGE)
        self.augmentor = CodeAugmentor(
            metadata_filepath=self.config["default_metadata_path"],
            description_generator=LlamaCodeDescriber() if self.USE_LOCAL_MODEL else GeminiClient(),
        )

        self.tmp_c_file = "./misc/tmp.c"
        self.tmp_cpp_file = "./misc/tmp.cpp"
        os.makedirs(name="./misc", exist_ok=True)

    def _setup_config(self):
        # find the absolute path to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # define the project root 
        project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
        # build the absolute path to the config file.
        config_path = os.path.join(project_root, "config.json")
        # load config with relative paths 
        self.config = load_config(fp=config_path)
        # update paths in config file
        for key, value in self.config.items():
            if key.endswith("_path"):
                self.config[key] = os.path.join(project_root, value)
                # self.config[key] = os.path.abspath(os.path.join(project_root, value))


    def _process_snippet(self, data: dict[str, str]) -> dict[str, str]:
        """Processes a single function snippet through the entire pipeline:
        Sanitize -> Repair -> Format -> Augment
        """

        raw_func_str:str|None = data.get("func")
        if not raw_func_str:
            data["func"] = "error: empty function body"
            return data

        try:
            # --- premature removal of comments to avoid false C++ positives ---
            lang:str="c"
            tsp: TreeSitterParser = TreeSitterParser(language_name=lang)
            code:str = self.sanitizer.remove_comments(code=raw_func_str, tsp=tsp)
            # filter out empty strings or comments only
            if not code.strip():
                data["func"] = "error: non valid function (empty or comments only)"
                return data

            # --- lang check ---
            if is_cpp(code=code):
                data.update({"func": "skipped: c++ function", "language": "cpp"})
                return data

            data["language"] = lang

            # --- PRELIMINARY SANITIZATION ---
            code = self.sanitizer._preprocess_directives(code=code, tsp=tsp)
            code = self.sanitizer._balance_directives(code=code, tsp=tsp)

            # --- STRUCTURAL REPAIR (THE FORGE) ---
            # Step 2a: Fix interleaved blocks with regex first.
            code = self.interleaved_fixer.full_structural_refactor(c_code=code)
            code = self.sanitizer.add_missing_braces(code=code)
            # Step 2b: Run the tree-sitter based multi-pass repair.
            code = self.foundry.run_multi_pass_fix(code=code)

            # --- finish sanitizing ---
            code = self.sanitizer.add_missing_return_types(code=code, tsp=tsp)
            code = self.sanitizer._kr_style_to_ansi(code=code, tsp=tsp)

            # check to ensure a function-like structure exists
            function_query_str = "(function_definition) @function"
            captures = tsp.query(code=code, query_str=function_query_str)
            if not captures:
                data["func"] = "error: non valid function (no function definition found)"
                return data

            # --- FORMAT ---
            code = get_refactored_code(
                code=code, lang_name=lang,
                fp=self.tmp_c_file, clang_format_file_path=self.config["clang_format_path"],
            )
            data["func"] = code

            # --- HEALTH CHECK & MANUAL FIX (Human-in-the-Loop) ---
            final_tree:Tree = tsp.parse(code=code)
            if tsp.is_broken_tree(tree=final_tree):
                manual_fix_path:str = "./misc/manual_fix_required.c"
                write2file(fp=manual_fix_path, content=code)

                logger.warning("="*60)
                logger.warning("🚨 BROKEN AST DETECTED 🚨")
                logger.warning(f"Pipeline paused. Please manually fix the code in:")
                logger.warning(f"==> {os.path.abspath(manual_fix_path)}")
                logger.warning("After saving your changes, type 'continue' and press Enter to resume.")
                logger.warning("="*60)

                pause_exec()

                # Read the manually fixed code back in
                data["func"] = read_file(fp=manual_fix_path, strip=False)
                logger.info("✅ Resuming pipeline with manually fixed code. ✅")

            return data

        except Exception as e:
            error_msg = str(e).replace("\n", " ").strip()
            logger.error(f"Pipeline failed for a snippet. Details: {error_msg}")
            data["func"] = f"error: pipeline failed. Details: {error_msg}"
            data["original_func"] = raw_func_str

            return data

    def run(self):
        """Executes the full preprocessing pipeline."""
        try:
            lines:list[str] = read_lines(fp=self.config["default_input_path"])
        except FileNotFoundError:
            logger.error(f"The source file was not found at {self.config["default_input_path"]}")
            return

        processed_entries: list[dict[str,str]] = []
        # with open(file=self.config["default_output_path"], mode="w", encoding="utf-8") as outfile:
        for i, line in enumerate(tqdm(iterable=lines, desc="Forging Code Snippets")):
            try:
                data:dict[str,str] = json.loads(line)
                processed_data:dict[str,str] = self._process_snippet(data=data)
                processed_entries.append(processed_data)
                # outfile.write(json.dumps(processed_data) + "\n")
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line {i}. Skipping.")
                error_entry:dict[str,str] = { "original_line": str(i), "func": "error: failed to parse line" }
                processed_entries.append(error_entry)
                # outfile.write(json.dumps(error_entry) + "\n")

        # -- run augmentation on accumulated entries at once --
        logger.debug("Start enrichment")
        final_entries = self.augmentor.enrich_dataset_batch(processed_entries)
        # -- write final, fully enriched data to the output file --
        with open(self.config["default_output_path"], "w", encoding="utf-8") as outfile:
            for entry in tqdm(final_entries, desc="Writing Output"):
                outfile.write(json.dumps(entry) + "\n")

        logger.info(f"✅ Successfully forged {len(lines)} snippets. ✅")
        logger.info(f"🗃️ Finished dataset saved to: {self.config["default_output_path"]} 🗃️")

        # clean up temporary files
        if os.path.exists(self.tmp_c_file):
            os.remove(self.tmp_c_file)

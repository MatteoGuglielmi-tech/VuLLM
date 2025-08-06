import os
import json
from dataclasses import dataclass
from tqdm import tqdm
from tree_sitter import Query, QueryCursor

from .proc_utils import load_config, decode_escaped_string, is_cpp, get_refactored_code, pause_exec, read_file, read_lines, write2file
from .tree_sitter_parser import C_LANGUAGE, TreeSitterParser, Tree
from .code_sanitizer import CodeSanitizer
from .interleaved_block_fixer import InterleavedBlockFixer
from .code_foundry import CodeFoundry
from .code_augmentor import CodeAugmentor

from .log import logger


@dataclass
class Vulcan:
    """The master pipeline orchestrator. Vulcan forges raw code snippets into
    clean, repaired, and enriched dataset entries.
    """

    def __post_init__(self):
        self.processed_data: dict[str,str]= {}

        # adjust paths
        self._setup_config()

        # Initialize the pipeline components
        self.sanitizer = CodeSanitizer()
        self.interleaved_fixer = InterleavedBlockFixer()
        self.foundry = CodeFoundry(ts_lang=C_LANGUAGE)
        self.augmentor = CodeAugmentor(metadata_filepath=self.config["default_metadata_path"])

        self.tmp_c_file = "./misc/tmp.c"
        self.tmp_cpp_file = "./misc/tmp.cpp"
        os.makedirs(name="./misc", exist_ok=True)

    def _setup_config(self):
        # find the absolute path to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # define the project root 
        project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
        # build the absolute path to the config file.
        config_path = os.path.join(script_dir, "config.json")
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
            decoded_code:str = decode_escaped_string(raw_string=raw_func_str)
            if is_cpp(code=decoded_code):
                data.update({"func": "skipped: c++ function", "language": "cpp"})
                return data

            lang:str="c"
            data["language"] = lang
            tsp: TreeSitterParser = TreeSitterParser(language_name=lang)

            # --- PRELIMINARY SANITIZATION ---
            code:str = self.sanitizer.remove_comments(code=decoded_code, tsp=tsp)
            # filter out empty strings or comments only
            if not code.strip():
                data["func"] = "error: non valid function (empty or comments only)"
                return data
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
                code=code,
                lang_name=lang,
                fp=self.tmp_c_file,
                clang_format_file_path=self.config["clang_format_path"],
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

            # --- AUGMENT ---
            enriched_data:dict[str,str] = self.augmentor.enrich_entry(entry=data)

            return enriched_data

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

        with open(file=self.config["default_output_path"], mode="w", encoding="utf-8") as outfile:
            for i, line in enumerate(tqdm(iterable=lines, desc="Forging Code Snippets")):
                try:
                    data:dict[str,str] = json.loads(line)
                    processed_data:dict[str,str] = self._process_snippet(data=data)
                    outfile.write(json.dumps(processed_data) + "\n")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line {i}. Skipping.")
                    error_entry:dict[str,str] = { "original_line": str(i), "func": "error: failed to parse line" }
                    outfile.write(json.dumps(error_entry) + "\n")

        logger.info(f"✅ Successfully forged {len(lines)} snippets. ✅")
        logger.info(f"🗃️ Finished dataset saved to: {self.config["default_output_path"]} 🗃️")

        # clean up temporary files
        if os.path.exists(self.tmp_c_file):
            os.remove(self.tmp_c_file)

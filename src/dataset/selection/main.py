import logging

from pathlib import Path
from dotenv import load_dotenv
from rich.traceback import install
from accelerate import Accelerator

from .logging_config import setup_logger
from .cli import get_cli_args
from .utilities import cleanup_resources, cleanup_single_gpu
from .collator import separate_targets, analyze_filter_and_save
from .filter import CWEDatasetFilter

install(show_locals=True)


logger = logging.getLogger(name=__name__)
setup_logger()
load_dotenv()

if __name__ == "__main__":
    logger.debug("🚀 Starting baseline... 🚀")
    args = get_cli_args()

    if args.filter:
        filter_tool = CWEDatasetFilter(jsonl_path=args.dataset)
        filter_tool.exec(output_dir=args.output_dir, filename=args.filename)

    if args.selection:
        separate_targets(
            jsonl_path=args.dataset, output_dir=args.output_dir, force=args.force
        )

        try:
            if args.finetuning:
                from .analyzers import FineTunePromptAnalyzer

                analyzer_type = FineTunePromptAnalyzer
            elif args.assessment:
                from .analyzers import JudgePromptAnalyzer

                analyzer_type = JudgePromptAnalyzer

            vulpath: Path = args.output_dir / "vulnearble.jsonl"
            safepath: Path = args.output_dir / "safe.jsonl"
            analyze_filter_and_save(
                dataset_path=vulpath if args.split == "vul" else safepath,
                analyzer_type=analyzer_type, # type: ignore
                output_dir=args.output_dir,
                filename=args.filename,
                tokenizer_name=args.tokenizer,
                n_samples=args.to_sample,
            )
        finally:
            import torch
            if torch.cuda.device_count() > 1:
                accelerator = Accelerator()
                cleanup_resources(accelerator=accelerator)
            else:
                cleanup_single_gpu()


import logging

from pathlib import Path
from dotenv import load_dotenv
from rich.traceback import install
from accelerate import Accelerator

from .logging_config import setup_logger
from .cli import get_parsed_args
from .utilities import cleanup_resources
from .collator import separate_targets, analyze_filter_and_save

install(show_locals=True)


logger = logging.getLogger(name=__name__)
setup_logger()
load_dotenv()

if __name__ == "__main__":
    logger.debug("🚀 Starting baseline... 🚀")
    args = get_parsed_args()

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
            analyzer_type=analyzer_type,
            output_dir=args.output_dir,
            filename=args.filename,
            tokenizer_name=args.tokenizer,
            n_samples=args.to_sample,
        )
    finally:
        accelerator = Accelerator()
        cleanup_resources(accelerator)

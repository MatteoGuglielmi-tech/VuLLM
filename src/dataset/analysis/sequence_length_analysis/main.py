import logging

from dotenv import load_dotenv
from rich.traceback import install
from accelerate import Accelerator

from .logging_config import setup_logger
from .cli import get_parsed_args
from .analyzer_wrapper import analyze_single_tokenizer, compare_tokenizers
from .utilities import cleanup_resources

install(show_locals=False)


logger = logging.getLogger(name=__name__)

if __name__ == "__main__":
    setup_logger()
    load_dotenv()
    args = get_parsed_args()

    logger.debug("🚀 Starting baseline... 🚀")

    try:
        if args.finetuning:
            if args.version == 1:
                from .analyzers import FineTunePromptAnalyzer

                analyzer_type = FineTunePromptAnalyzer
            elif args.version == 2:
                from .analyzers import FineTunePromptAnalyzerV2

                analyzer_type = FineTunePromptAnalyzerV2
        elif args.assessment:
            from .analyzers import JudgePromptAnalyzer

            analyzer_type = JudgePromptAnalyzer

        if args.tokenizer:
            analyze_single_tokenizer(
                dataset_path=args.dataset,
                analyzer_type=analyzer_type,
                tokenizer_name=args.tokenizer,
                chat_template=args.chat_template,
                output_dir=args.output,
                max_samples=args.max_samples,
            )
        else:
            compare_tokenizers(
                dataset_path=args.dataset,
                analyzer_type=analyzer_type,
                tokenizer_config=args.tokenizers,
                output_dir=args.output,
                max_samples=args.max_samples,
            )

        logger.info(f"🎉 All done! Check the `{args.output}` for results.")
    finally:
        accelerator = Accelerator()
        cleanup_resources(accelerator)

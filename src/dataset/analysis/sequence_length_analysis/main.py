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
                analyzer_version = "v1"
            elif args.version == 2:
                analyzer_version = "v2"
            elif args.version == 3:
                analyzer_version = "v3"
            else:
                raise ValueError(f"Unknown version: {args.version}")
        elif args.assessment:
            analyzer_version = "judge"
        else:
            raise ValueError("Must specify --finetuning or --assessment")

        common_kwargs = {
            "dataset_path": args.dataset,
            "analyzer_version": analyzer_version,
            "output_dir": args.output,
            "max_samples": args.max_samples,
        }

        if analyzer_version in {"v2", "v3"}:
            common_kwargs.update(
                {
                    "assumption_mode": args.assumption_mode,
                    "prompt_phase": args.prompt_mode,
                    "add_hierarchy": args.add_hierarchy,
                }
            )

        if args.tokenizer:
            analyze_single_tokenizer(
                tokenizer_name=args.tokenizer,
                chat_template=args.chat_template,
                **common_kwargs,
            )
        else:
            compare_tokenizers(tokenizer_config=args.tokenizers, **common_kwargs)

        logger.info(f"🎉 All done! Check the `{args.output}` for results.")
    finally:
        accelerator = Accelerator()
        cleanup_resources(accelerator)

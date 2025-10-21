import sys
import logging
import cli

from logging_config import setup_logger
from sequence_length_analyzer import (
    analyze_single_tokenizer,
    compare_tokenizers,
)


logger = logging.getLogger(name=__name__)

if __name__ == "__main__":
    setup_logger()
    args = cli.get_parsed_args()

    try:
        if args.tokenizer:
            analyze_single_tokenizer(
                dataset_path=args.dataset,
                tokenizer_name=args.tokenizer,
                output_dir=args.output,
                max_samples=args.max_samples,
            )
        else:
            compare_tokenizers(
                dataset_path=args.dataset,
                tokenizer_names=args.tokenizers,
                output_dir=args.output,
                max_samples=args.max_samples,
            )

        logger.info(f"🎉 All done! Check the `{args.output}` for results.")

    except KeyboardInterrupt:
        logger.warning("⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

import sys
import argparse
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


# DEFAULT_TOKENIZER_LIST = [
#     "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
#     "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
#     "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
#     "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
# ]


def _get_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        description="Select subset of samples based on token length for representative samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "%(prog)s --dataset data.jsonl --tokenizer unsloth/llama-3-8b-bnb-4bit --output_dir path2output --filename filename.jsonl\n"
        ),
    )

    # ============================================================================
    # COMMON ARGUMENTS (Required for all modes)
    # ============================================================================
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory wherein saving selected samples.",
    )
    parser.add_argument(
        "--filename",
        type=Path,
        required=True,
        help="Output filename wherein saving selected samples.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["vul", "safe"],
        required=True,
        help="Which target to take in analysis.",
    )
    parser.add_argument(
        "--to_sample",
        type=int,
        default=2500,
        help="How many samples to select around the median.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force creation of per-target split.",
    )

    # prompt selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--finetuning",
        action="store_true",
        help="Assess max sequence length using fine-tuning prompt skeleton",
    )
    mode_group.add_argument(
        "--assessment",
        action="store_true",
        help="Assess max sequence length using assessment prompt skeleton",
    )

    return parser


def get_parsed_args() -> argparse.Namespace:
    parser = _get_parser()
    try:
        args = parser.parse_args()
        if args.tokenizer is None:
            logger.warning(
                "[yellow1]"
                "`--tokenizer` argument has not been specfied."
                "Fallback to `unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit`."
                "[/yellow1]"
            )
        logger.info("✅" " [green]CLI arguments parsed successfully[/green]")
        return args
    except argparse.ArgumentError:
        logger.exception("❌ An exception has occured during CLI parsing ❌")
        sys.exit(1)

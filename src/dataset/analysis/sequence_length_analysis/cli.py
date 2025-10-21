import sys
import argparse
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_TOKENIZER_LIST = [
    "unsloth/llama-3.1-8b-instruct-bnb-4bit",
    "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
    "unsloth/Qwen3-32B-unsloth-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
    "unsloth/QwQ-32B-unsloth-bnb-4bit"
]


def _get_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        description="Analyze token distribution in CoT dataset to determine optimal max_seq_length",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "# Single tokenizer analysis\n"
            "%(prog)s --dataset data.jsonl --tokenizer unsloth/llama-3-8b-bnb-4bit\n"
            "# Compare multiple tokenizers\n"
            "%(prog)s --dataset data.jsonl --tokenizers unsloth/llama-3-8b-bnb-4bit mistralai/Mistral-7B-Instruct-v0.2\n"
            "# Analyze subset of data\n"
            "%(prog)s --dataset data.jsonl --tokenizer unsloth/llama-3-8b-bnb-4bit --max-samples 1000\n"
        ),
    )

    parser.add_argument(
        "--dataset", "-d", type=Path, required=True, help="Path to JSONL dataset file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./sequence_analysis"),
        help="Output directory for results (default: ./sequence_analysis)",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Single tokenizer to use (e.g., unsloth/llama-3-8b-bnb-4bit)",
    )

    group.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        default=None,
        help=f"Multiple tokenizers to compare. "
        f"Default is: {', '.join(DEFAULT_TOKENIZER_LIST)}",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)",
    )

    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt (uses default if not provided)",
    )

    parser.add_argument(
        "--prompt-skeleton",
        type=str,
        default=None,
        help="Custom prompt skeleton (uses default if not provided)",
    )

    return parser


def get_parsed_args() -> argparse.Namespace:
    parser = _get_parser()
    args = parser.parse_args()

    try:
        args = parser.parse_args()
        logger.info("Successfully parsed arguments:")
    except SystemExit:
        logger.error("Caught argparse exit due to error.")
        if args.tokenizer and args.tokenizers:
            logger.error("❌ Cannot specify both --tokenizer and --tokenizers")
            sys.exit(1)

    if not args.dataset.exists():
        logger.error(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)

    if args.tokenizer is None and args.tokenizers is None:
        logger.warning("NEITHER --tokenizer nor --tokenizers was given -> using default tokenizer list.")
        args.tokenizers = DEFAULT_TOKENIZER_LIST

    logger.debug(args)

    return args


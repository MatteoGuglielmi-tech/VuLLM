import sys
import argparse
import logging

from pathlib import Path
from .utilities import rich_exception

logger = logging.getLogger(__name__)


DEFAULT_TOKENIZER_LIST = [
    "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
    "microsoft/Phi-4",
    "microsoft/Phi-4-reasoning-plus",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
]


def _get_parser() -> argparse.ArgumentParser:
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        description="Analyze token distribution in a CoT dataset to determine optimal max_seq_length",
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

    # ============================================================================
    # COMMON ARGUMENTS (Required for all modes)
    # ============================================================================
    parser.add_argument(
        "--dataset", "-d", type=Path, required=True, help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path(__file__) / "results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to analyze (default: all)",
    )

    # ============================================================================
    # MODE SELECTION (Mutually Exclusive)
    # ============================================================================
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--tokenizer", type=str, default=None, help="Single tokenizer to use"
    )
    tok_group.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        default=None,
        help=f"Multiple tokenizers to compare. "
        f"Default is: {', '.join(DEFAULT_TOKENIZER_LIST)}",
    )

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
        if args.tokenizer is None and args.tokenizers is None:
            logger.warning("Neither `--tokenizers` nor `--tokenizer` have been specfied. Fallback to `--tokenizers` with default set of tokenizers."
            )
        args.tokenizers = DEFAULT_TOKENIZER_LIST
        logger.info("CLI arguments parsed successfully")
        return args
    except SystemExit:
        rich_exception()
        sys.exit(1)

import sys
import argparse
import logging

from pathlib import Path
from .utilities import rich_exception

logger = logging.getLogger(__name__)


DEFAULT_TOKENIZER_LIST = [
    ("unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit", "qwen-2.5"),
    ("unsloth/Qwen2.5-72B-Instruct-bnb-4bit", "qwen-2.5"),
    ("microsoft/Phi-4", "phi-4"),
    ("microsoft/Phi-4-reasoning-plus", "phi-4"),
    ("unsloth/Llama-3.3-70B-Instruct-bnb-4bit", "llama-3.3"),
    ("unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit", "llama-3.3"),
]


def parse_tokenizer_pair(value: str) -> tuple[str, str]:
    """Parses 'tokenizer:template' into (tokenizer, template)."""
    try:
        tokenizer, template = value.split(":")
        return tokenizer, template
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format '{value}'. Must be 'tokenizer:template'"
        )


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
        "--chat_template",
        type=str,
        default=None,
        help="Chat template to applye (single tokenizer only)",
    )
    tok_group.add_argument(
        "--tokenizers",
        type=parse_tokenizer_pair,
        nargs="+",
        default=None,
        help=f"Multiple tokenizers to compare. "
        f"Default is: {DEFAULT_TOKENIZER_LIST}",
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

    finetune_group = parser.add_argument_group("FineTuning prompt version")
    finetune_group.add_argument("--version", "-v", type=int, choices=[1, 2], help="Prompt version to use.")

    return parser


def get_parsed_args() -> argparse.Namespace:
    parser = _get_parser()
    try:
        args = parser.parse_args()
        if (args.tokenizer or args.tokenizers) is None: # both omitted
            logger.warning(
                "Neither `--tokenizers` nor `--tokenizer` have been specfied. "
                "Fallback to `--tokenizers` with default set of tokenizers."
            )
            args.tokenizers = DEFAULT_TOKENIZER_LIST
        if args.tokenizer and not args.chat_template:
            raise argparse.ArgumentError(
                argument=args.chat_template,
                message="Missing required `--chat_template`",
            )
        if args.finetuning and args.version is None:
            logger.warning(
                "Version argument not specified, but finetuning mode found. "
                "Fallback to v2"
            )
            args.version = 2

        logger.info("CLI arguments parsed successfully")
        return args
    except SystemExit:
        rich_exception()
        sys.exit(1)

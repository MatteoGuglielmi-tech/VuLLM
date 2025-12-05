import sys
import argparse
import logging

from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlagInfo:
    attr_name: str
    flag: str


def check_required_args(args, required: list[FlagInfo]):
    """Check if required arguments are present."""
    for info in required:
        if getattr(args, info.attr_name) is None:
            yield info.flag

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
    base_group = parser.add_argument_group(
        title="Mandatory arguments",
        description="Group of arguments required by all modes",
    )
    base_group.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL dataset file"
    )
    base_group.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory wherein saving selected samples.",
    )
    base_group.add_argument(
        "--filename",
        type=Path,
        required=True,
        help="Output filename wherein saving selected samples.",
    )

    # mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--filter",
        action="store_true",
        help="Peform filtering of low reccurent cwe identifiers."
    )
    mode_group.add_argument(
        "--selection",
        action="store_true",
        help="Perform selection of `--to_sample` examples around token length median.",
    )

    # prompt group (--selection only)
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--finetuning",
        action="store_true",
        help="Assess max sequence length using fine-tuning prompt skeleton",
    )
    prompt_group.add_argument(
        "--assessment",
        action="store_true",
        help="Assess max sequence length using assessment prompt skeleton",
    )

    selection_group = parser.add_argument_group(
        title="Selection mode arguments",
        description="Group of arguments required when `--selection` mode.",
    )
    selection_group.add_argument(
        "--split",
        type=str,
        choices=["vul", "safe"],
        help="Which target to take in analysis.",
    )
    selection_group.add_argument(
        "--to_sample",
        type=int,
        default=2500,
        help="How many samples to select around the median.",
    )
    selection_group.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer to use",
    )
    selection_group.add_argument(
        "--force",
        action="store_true",
        help="Force creation of per-target split.",
    )

    return parser


def validate_args(args: argparse.Namespace):
    """
    Validates that only mode-specific arguments are used with their respective modes.
    Call this after parsing arguments.
    """

    filter_mode_only = {}
    selection_mode_only = {
        "finetuning",
        "assessment",
        "split",
        "to_sample",
        "tokenizer",
        "force"
    }


    required: list[FlagInfo] = []

    if args.filter:
        invalid_args = []
        for arg in selection_mode_only:
            if getattr(args, arg) != get_default_value(arg):
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise argparse.ArgumentTypeError(
                f"Arguments {', '.join(invalid_args)} cannot be used with --filter mode"
            )

    elif args.selection:
        required.extend(
            [
                FlagInfo(attr_name="split", flag="--split"),
                FlagInfo(attr_name="tokenizer", flag="--tokenizer"),
            ]
        )
        missing = list(check_required_args(args=args, required=required))
        mode_required = [
            FlagInfo(attr_name="finetuning", flag="--finetuning"),
            FlagInfo(attr_name="assessment", flag="--assessment"),
        ]
        if missing:
            raise argparse.ArgumentTypeError(
                f"Required for --selection mode: {', '.join(missing)}"
            )
        if not any(mode_required):
            raise argparse.ArgumentTypeError(
                f"A prompt specifier is required for --selection"
            )

        invalid_args = []

        for arg in filter_mode_only:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise argparse.ArgumentTypeError(
                f"Arguments {', '.join(invalid_args)} cannot be used with --selection mode"
            )


def get_default_value(arg_name: str):
    """Returns the default value for a given argument."""
    defaults = {
        "finetuning": False,
        "assessment": False,
        "split": None,
        "to_sample": 2500,
        "tokenizer": None,
        "force": False,
    }
    return defaults.get(arg_name)


def get_cli_args() -> argparse.Namespace:
    parser = _get_parser()
    try:
        args = parser.parse_args()
        validate_args(args)
        logger.info("✅" " [green]CLI arguments parsed successfully[/green]")
        return args
    except argparse.ArgumentError:
        logger.exception("❌ An exception has occured during CLI parsing ❌")
        sys.exit(1)

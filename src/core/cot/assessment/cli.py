import argparse

from argparse import ArgumentTypeError
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Chain of thougths quality assessment",
        description="A jury that judges the quality of CoTs",
    )
    # ============================================================================
    # COMMON ARGUMENTS (Required for all modes)
    # ============================================================================
    path_group = parser.add_argument_group("Path mandatory arguments")
    path_group.add_argument(
        "--input", "-i", type=Path, required=True, help="Path to the source dataset."
    )
    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument(
        "--max_lengths",
        type=int,
        nargs="+",
        default=[8192, 8192, 8192],
        help="List of max sequence lengths for each judge (Order matters!!)."
    )
    model_group.add_argument(
        "--max_new_tokens", "-m",
        type=int, default=256,
        help="Maximum tokens for generation completion.",
    )
    model_group.add_argument(
        "--save_interval", "-s",
        type=int, default=100,
        help="Steps at which serialize the generated content.",
    )
    # ============================================================================
    # MODE SELECTION (Mutually Exclusive)
    # ============================================================================
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--ensemble",
        action="store_true",
        help="Evaluate with an ensemble of judges (full pipeline)",
    )
    mode_group.add_argument(
        "--sequential", action="store_true", help="Evaluate with one judge at a time"
    )
    mode_group.add_argument(
        "--merge", action="store_true", help="Run merge script."
    )
    # ============================================================================
    # SHARED: ENSEMBLE & MERGE ARGUMENTS
    # ============================================================================
    shared_group = parser.add_argument_group("Shared arguments between `ensemble` and `merge` modes")
    shared_group.add_argument("--output", "-o", type=Path, help="Output folder.")
    shared_group.add_argument("--assets", "-p", type=Path, help="Assets folder.")
    shared_group.add_argument(
        "--quality_threshold", "-q",
        type=float, default=0.60,
        help="Minimum average quality (0-1).",
    )
    shared_group.add_argument(
        "--agreement_threshold", "-a",
        type=float, default=0.75,
        help="Minimum judge agreement (0-1, higher=stricter)."
    )
    shared_group.add_argument(
        "--agreement_method", "-t",
        type=str, default="weighted_multidimensional",
        choices=["multidimensional", "weighted_multidimensional"],
        help="Type of agreement to compute",
    )
    # ============================================================================
    # SEQUENTIAL ARGUMENTS
    # ============================================================================
    sequential_group = parser.add_argument_group("Sequential mode only arguments")
    sequential_group.add_argument(
        "--judge", "-j",
        type=str, choices=["qwen-coder", "llama-3.1-70B", "deepseek-qwen"],
        help="Name of the judge model to use.",
    )
    sequential_group.add_argument("--output_path", "-op", type=Path, help="Output filepath.")
    # ============================================================================
    # MERGE ARGUMENTS
    # ============================================================================
    merge_group = parser.add_argument_group("Merge script only arguments")
    merge_group.add_argument(
        "--judge_files",
        type=Path,
        nargs="+",
        default=None,
        help="List of paths leading to evaluation files."
    )
    return parser


def validate_args(args):
    """
    Validates that only mode-specific arguments are used with their respective modes.
    Call this after parsing arguments.
    """

    ensemble_only = {}
    sequential_only = {"output_path", "judge"}
    merge_only = {"judge_files"}
    shared = {
        "output",
        "assets",
        "quality_threshold",
        "agreement_threshold",
        "agreement_method",
    }

    invalid_args = []

    if args.ensemble:
        if args.output is None:
            raise argparse.ArgumentTypeError("--output is required for --ensemble mode")
        if args.assets is None:
            raise argparse.ArgumentTypeError("--assets is required for --ensemble mode")

        prohibited = sequential_only.copy()
        prohibited.update(merge_only)
        for arg in prohibited:
            if getattr(args, arg) != get_default_value(arg):
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise ArgumentTypeError(f"Arguments {', '.join(invalid_args)} cannot be used with --ensemble mode")

    elif args.sequential:
        if args.output_path is None:
            raise argparse.ArgumentTypeError("--output_path is required for --sequential mode")
        if args.judge is None:
            raise ArgumentTypeError("--judge is required for --sequential mode")

        prohibited = shared.copy()
        prohibited.update(merge_only)
        prohibited.update(ensemble_only)
        for arg in prohibited:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise ArgumentTypeError(f"Arguments {', '.join(invalid_args)} cannot be used with --sequential mode")

    elif args.merge:
        if args.output_path is None:
            raise argparse.ArgumentTypeError("--output_path is required for --merge mode")
        if args.assets is None:
            raise argparse.ArgumentTypeError("--assets is required for --assets mode")
        if args.judge_files is None:
            raise ArgumentTypeError("--judge_files is required for --merge mode")

        prohibited = sequential_only.copy()
        prohibited.update(ensemble_only)
        for arg in prohibited:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise ArgumentTypeError(f"Arguments {', '.join(invalid_args)} cannot be used with --inference mode")

    if len(args.max_lengths) > 3:
        raise ValueError("Jury consists in 3 judges!!")

    return args


def get_default_value(arg_name):
    """Returns the default value for a given argument."""

    defaults = {
        # ensemble mode only
        "output": None,
        "assets": None,
        "quality_threshold": 0.6,
        "agreement_threshold": 0.75,
        "agreement_method": "weighted_multidimensional",
        # sequential only
        "judge": None,
        "output_path": None,
        # merge only
        "judge_files": None
    }

    return defaults.get(arg_name)

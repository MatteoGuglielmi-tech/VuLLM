import argparse


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        prog="Dataset PreprocessingPipeline",
        add_help=True,
        allow_abbrev=True,
    )

    parser.add_argument(
        "--clang_format_path",
        "-cfp",
        default=None,
        help="Location of clang-format config file",
    )

    parser.add_argument(
        "--dataset_fp",
        "-fp",
        default=None,
        help="Location of the input dataset.",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        help="Filepath where to save processed dataset to.",
    )

    parser.add_argument(
        "--debug",
        "-d",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Activate debug CLI logs",
    )

    return parser

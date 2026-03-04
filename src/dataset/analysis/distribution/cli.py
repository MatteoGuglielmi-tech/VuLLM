import argparse

from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(prog="🔎 Distribution analysis 🔎")

    # -- Paths --
    parser.add_argument(
        "--source", "-i", type=str, required=True, help="Path to the source dataset."
    )
    parser.add_argument(
        "--target",
        "-o",
        type=str,
        required=True,
        help="Path wherein saving the pipeline outcome.",
    )
    parser.add_argument(
        "--mitre_file",
        "-m",
        type=Path,
        required=True,
        help="Path leading to `cwe_comprehensive_view.csv` from MITRE",
    )

    return parser

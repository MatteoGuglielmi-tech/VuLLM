import argparse


def get_parser():
    parser = argparse.ArgumentParser(prog="🔎 Distribution analysis 🔎")

    # -- Paths --
    parser.add_argument("--source", "-i", type=str, required=True, help="Path to the source dataset.")
    parser.add_argument("--target", "-o", type=str, required=True, help="Path wherein saving the pipeline outcome.")

    return parser

import argparse

parser = argparse.ArgumentParser(
    prog="Dataset fixer",
    usage=None,
    description="This program pre-process and fix the original dataset and save it to a JSON file",
    add_help=True,
    allow_abbrev=True,
)

parser.add_argument(
    "--path",
    "-p",
    type=str,
    required=False,
    default="../../../DiverseVul/diversevul_20230702.json",
    help="Location of the original dataset",
)

parser.add_argument(
    "--file_name",
    "-n",
    required=False,
    default="../../../DiverseVul/DiverseVul.json",
    help="Name of the JSON file to save processed dataset",
)

parser.add_argument(
    "--start_idx",
    "-s",
    type=int,
    required=False,
    default=0,
    help="fn index to start from",
)

parser.add_argument(
    "--clear_json",
    "-c",
    type=bool,
    action=argparse.BooleanOptionalAction,
    help="Clear final JSON file",
)

parser.add_argument(
    "--debug",
    "-d",
    type=bool,
    action=argparse.BooleanOptionalAction,
    help="Activate debug CLI logs",
)

parser.add_argument(
    "--format_config_file",
    "-f",
    type=str,
    required=False,
    default="../../../.clang-format",
    help="Location of clang-format config file",
)

args = parser.parse_args()

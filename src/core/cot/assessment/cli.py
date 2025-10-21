import json
import argparse
from datetime import datetime
from pathlib import Path


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(prog="Chain of thougths assessment", description="A tool for evaluating generated text.")

    # -- Common arguments --
    common_group = parser.add_argument_group("Generic arguments")
    common_group.add_argument("--source", "-i", type=str, required=True, help="Path to the source dataset.")
    common_group.add_argument("--target", "-o", type=str, required=True, help="Directory path wherein saving the pipeline outcome.")
    common_group.add_argument("--max_completion_tokens", "-m", type=int, default=256, help="Maximum tokens for API completions.")
    common_group.add_argument("--sample_size", "-s", type=int, default=500, help="Number of samples to randomly select for assessment.")

    prop_group = parser.add_argument_group("API Arguments")
    prop_group.add_argument("--deployment_name", type=str, default="gpt-4.1", help="Name of the API deployment.")
    prop_group.add_argument("--engine_name", "-e", type=str, required=True, help="Name of the model engine.")

    return parser


def save_running_args(args: argparse.Namespace):
    date: str = datetime.today().strftime("%Y-%m-%d")
    time: str = datetime.now().strftime("%H-%M-%S")
    fp = Path(__file__).parent / f"run/{date}/{time}/cli_args.json"
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(file=fp, mode="w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print("--- Parsed Arguments ---")
    print(f"Mode selected: {args.model_type}")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")

    if args.model_type == 'open_source':
        print(f"Model Name: {args.model_name}")
        print(f"Batch Size: {args.batch_size}")
    elif args.model_type == 'proprietary':
        print(f"Deployment Name: {args.deployment_name}")
        print(f"Engine Name: {args.engine_name}")


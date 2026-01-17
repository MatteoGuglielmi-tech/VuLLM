import json
import argparse
from datetime import datetime
from pathlib import Path


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        prog="Chain of thougths generation",
        description="A tool for generating text using either open-source or proprietary models.",
    )

    # -- Common arguments --
    common_group = parser.add_argument_group("Common arguments")
    # -- Paths --
    common_group.add_argument("--source", "-i", type=str, required=True, help="Path to the source dataset.")
    common_group.add_argument("--target", "-o", type=str, required=True, help="Path wherein saving the pipeline outcome.")
    # -- Generation --
    common_group.add_argument("--max_completion_tokens", "-a", type=int, default=2048, help="Maximum tokens for API completions.")
    common_group.add_argument("--batch_size", "-b", type=int, default=8, help="Number of samples to generate reasoning for in a single pass.")
    # -- Flags ---
    common_group.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Activate debug CLI logs.")

    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Select API type")

    # --- Subparser for Open Source models ---
    parser_os = subparsers.add_parser("open_source", help="Use a local open-source model.")
    os_group = parser_os.add_argument_group("Generation Arguments")
    os_group.add_argument("--model_name", "-n", type=str, default="unsloth/llama-3.1-8b-instruct-bnb-4bit", help="Name of the model to use.")
    os_group.add_argument("--chat_template", "-c", type=str, default="llama-3.2", help="Chat template to use.")
    os_group.add_argument("--max_seq_length", "-m", type=int, default=2048, help="Maximum sequence lenght for deep model.")
    os_group.add_argument("--load_in_4bit", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Load 4-bit quantized model.")

    # --- Subparser for Proprietary models ---
    parser_prop = subparsers.add_parser("proprietary", help="Use a proprietary API-based model.")
    prop_group = parser_prop.add_argument_group("API Arguments")
    prop_group.add_argument("--deployment_name", type=str, default="gpt", help="Name of the API deployment.")
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


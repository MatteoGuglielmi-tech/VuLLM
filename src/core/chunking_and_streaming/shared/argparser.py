import argparse


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(prog="Dataset PreprocessingPipeline")

    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence lenght for deep model.")
    parser.add_argument("--lora_model_dir", type=str, required=True, help="Path to adapters folder.") # this is only for test
    parser.add_argument("--debug", type=bool, action=argparse.BooleanOptionalAction, help="Activate debug CLI logs")

    return parser

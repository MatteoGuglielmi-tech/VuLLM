import argparse
import json
from datetime import datetime
from pathlib import Path


class PathEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts Path objects to strings."""

    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(
        prog="CoT fine-tuning",
        description="Fine-tune, optimize, or run inference with CoT models",
    )

    # ============================================================================
    # COMMON ARGUMENTS (Required for all modes)
    # ============================================================================
    common_group = parser.add_argument_group("Common Arguments")
    common_group.add_argument(
        "--formatted_dataset_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory path wherein saving the formatted dataset (not tokenzied)",
    )
    common_group.add_argument(
        "--chat_template",
        "-c",
        type=str,
        required=True,
        help="Chat template to use",
    )
    common_group.add_argument(
        "--max_seq_length",
        "-m",
        type=int,
        default=4096,
        help="Maximum sequence length for deep model",
    )
    common_group.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=2,
        help="How many samples in a per device batch",
    )
    common_group.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug CLI logs",
    )

    # ============================================================================
    # MODE SELECTION (Mutually Exclusive)
    # ============================================================================
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--finetune", action="store_true", help="Fine-tuning mode")
    mode_group.add_argument("--hpo", action="store_true", help="Hyperparameter optimization mode")
    mode_group.add_argument("--inference", action="store_true", help="Inference mode")

    # ============================================================================
    # SHARED: FINE-TUNING & HPO ARGUMENTS
    # ============================================================================
    shared_training_group = parser.add_argument_group("Shared Training Arguments (Fine-tuning & HPO)")
    shared_training_group.add_argument(
        "--dataset_path",
        "-i",
        type=str,
        # required=True,
        help="Path to the source dataset (required for fine-tuning and HPO)"
    )
    shared_training_group.add_argument(
        "--base_model_name",
        "-n",
        type=str,
        help="Name of the base mo,del to use",
    )
    shared_training_group.add_argument(
        "--epochs", "-e", type=int, default=3, help="Training epochs"
    )
    shared_training_group.add_argument(
        "--grad_acc_steps",
        "-g",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    shared_training_group.add_argument(
        "--strategy",
        type=str,
        choices=["fast", "explore", "balanced"],
        default="explore",
        help="Strategy to use during fine-tuning",
    )
    shared_training_group.add_argument(
        "--logging_steps",
        "-x",
        type=int,
        default=50,
        help="Interval training log is shown at",
    )
    shared_training_group.add_argument(
        "--use_rslora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RsLoRA for stability",
    )
    shared_training_group.add_argument(
        "--use_loftq", action="store_true", help="Use LoftQ"
    )
    shared_training_group.add_argument(
        "--use_weighted_trainer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use WeightedCoTTrainer (fine-tuning only)",
    )
    shared_training_group.add_argument(
        "--deepspeed",
        action="store_true",
        help="Enable DeepSpeed training",
    )

    # ============================================================================
    # FINE-TUNING ARGUMENTS
    # ============================================================================
    finetune_group = parser.add_argument_group("Fine-tuning only arguments")
    finetune_group.add_argument(
        "--learning_rate",
        "-t",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning",
    )
    finetune_group.add_argument(
        "--weight_decay",
        "-w",
        type=float,
        default=0,
        help="Weight decay to apply during fine-tuning",
    )
    finetune_group.add_argument(
        "--lora_rank", "-r", type=int, default=16, help="LoRA rank (fine-tuning only)"
    )
    finetune_group.add_argument(
        "--lora_alpha", "-a", type=int, default=32, help="LoRA alpha (fine-tuning only)"
    )
    finetune_group.add_argument(
        "--lora_dropout", "-d", type=float, default=0, help="LoRa dropout (fine-tuning only)"
    )

    # ============================================================================
    # HPO ONLY ARGUMENTS
    # ============================================================================
    hpo_group = parser.add_argument_group("Hyperparameter Optimization Arguments")
    hpo_group.add_argument(
        "--n_trials",
        "-hr",
        type=int,
        default=5,
        help="How many trials for HPO (HPO only)",
    )
    hpo_group.add_argument(
        "--run_cap",
        "-hc",
        type=int,
        default=100,
        help="Maximum number of runs for HPO (HPO only)",
    )

    # ============================================================================
    # INFERENCE ARGUMENTS
    # ============================================================================
    inference_group = parser.add_argument_group("Inference Arguments")
    inference_group.add_argument(
        "--lora_weights",
        "-l",
        type=str,
        help="Path wherein the LoRA weights have been saved (inference only)",
    )
    inference_group.add_argument(
        "--max_tokens_per_answer",
        "-p",
        type=int,
        default=2048,
        help="Maximum tokens per generated answer (inference only)",
    )
    inference_group.add_argument(
        "--chat_template_inference",
        "-ci",
        type=str,
        help="Chat template of the model used for inference",
    )
    inference_group.add_argument(
        "--assets_dir",
        "-ad",
        type=Path,
        help="Directory where to save the generated plots and reports to (inference only)",
    )
    inference_group.add_argument(
        "--use_batching",
        action="store_true",
        help="Use batching in inference",
    )
    inference_group.add_argument(
        "--include_code_in_reports",
        action="store_true",
        help="Include full function code in misclassification reports",
    )
    inference_group.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Save evaluation artifacts",
    )

    return parser


def validate_args(args):
    """
    Validates that only mode-specific arguments are used with their respective modes.
    Call this after parsing arguments.
    """

    # peft only args
    finetune_only = {
        "learning_rate",
        "weight_decay",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
    }

    # hpo only args
    hpo_only = {"n_trials", "run_cap"}

    # test only args
    inference_only = {
        "lora_weights",
        "max_tokens_per_answer",
        "assets_dir",
        "use_batching",
        "include_code_in_reports",
        "save_artifacts"
    }

    # args shared between fine-tuning and HPO (not allowed/unnecessary in inference)
    training_shared = {
        "dataset_path",
        "base_model_name",
        "epochs",
        "grad_acc_steps",
        "strategy",
        "logging_steps",
        "use_rslora",
        "use_loftq",
        "use_weighted_trainer",
        "deepspeed",
    }

    if args.finetune:
        if args.base_model_name is None:
            raise argparse.ArgumentTypeError("--base_model_name is required for --finetune mode")
        if args.dataset_path is None: # needs dataset path
            raise argparse.ArgumentTypeError("--dataset_path is required for --finetune mode")

        invalid_args = []

        # Check HPO-only arguments
        for arg in hpo_only:
            if getattr(args, arg) != get_default_value(arg):
                invalid_args.append(f"--{arg}")

        # Check inference-only arguments
        for arg in inference_only:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value is not None and value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise argparse.ArgumentTypeError(
                f"Arguments {', '.join(invalid_args)} cannot be used with --finetune mode"
            )

    elif args.hpo:
        if args.base_model_name is None:
            raise argparse.ArgumentTypeError("--base_model_name is required for --hpo mode")
        if args.dataset_path is None:
            raise argparse.ArgumentTypeError("--dataset_path is required for --hpo mode")

        invalid_args = []

        # Check fine-tuning-only arguments
        for arg in finetune_only:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value != default:
                invalid_args.append(f"--{arg}")

        # Check inference-only arguments
        for arg in inference_only:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value is not None and value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise argparse.ArgumentTypeError(
                f"Arguments {', '.join(invalid_args)} cannot be used with --hpo mode"
            )

    elif args.inference:
        if args.lora_weights is None:
            raise argparse.ArgumentTypeError(
                "--lora_weights is required for --inference mode"
            )

        invalid_args = []

        # Check fine-tuning-only arguments
        for arg in finetune_only:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if value != default:
                invalid_args.append(f"--{arg}")

        # Check HPO-only arguments
        for arg in hpo_only:
            if getattr(args, arg) != get_default_value(arg):
                invalid_args.append(f"--{arg}")

        # Check training-shared arguments
        for arg in training_shared:
            value = getattr(args, arg)
            default = get_default_value(arg)
            if arg == "dataset_path":
                if value is not None:
                    invalid_args.append(f"--{arg}")
            if value != default:
                invalid_args.append(f"--{arg}")

        if invalid_args:
            raise argparse.ArgumentTypeError(
                f"Arguments {', '.join(invalid_args)} cannot be used with --inference mode"
            )



def get_default_value(arg_name):
    """Returns the default value for a given argument."""
    defaults = {
        # Shared training arguments
        "dataset_path": None,
        "base_model_name": None,
        "epochs": 3,
        "grad_acc_steps": 4,
        "strategy": "explore",
        "logging_steps": 50,
        "use_rslora": True,
        "use_loftq": False,
        "use_weighted_trainer": False,
        "deepspeed": False,
        # Fine-tuning only
        "learning_rate": 2e-5,
        "weight_decay": 0,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,
        # HPO only
        "n_trials": 5,
        "run_cap": 100,
        # Inference only
        "lora_weights": None,
        "max_tokens_per_answer": 2048,
        "assets_dir": None,
        "use_batching": False,
        "include_code_in_reports": False,
        "save_artifacts": False
    }
    return defaults.get(arg_name)


def save_running_args(args: argparse.Namespace):
    date: str = datetime.today().strftime("%Y-%m-%d")
    time: str = datetime.now().strftime("%H-%M-%S")
    fp = Path(__file__).parent / f"run/{date}/{time}/cli_args.json"
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(file=fp, mode="w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, cls=PathEncoder)

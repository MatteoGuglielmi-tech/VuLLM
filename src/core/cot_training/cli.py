import argparse
import json

from datetime import datetime
from pathlib import Path

from .processing_lib import PromptPhase, AssumptionMode


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
        # required=True,
        help="Directory path wherein saving the formatted dataset (not tokenzied)",
    )
    common_group.add_argument(
        "--chat_template",
        "-c",
        type=str,
        # required=True,
        help=(
            "Chat template to use:\n"
            "  - fine-tuning and hpo modes: it represents the chat_template used to format input samples and extract delimiters for reponse only learning"
            "  - inference mode: it represents the chat_template applied to test samples"
        ),
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
        default=4,
        help=(
            "How many samples in a per device batch:\n"
            "  - fine-tuning and hpo modes: it represents the batch used to teach the model"
            "  - inference mode: it represents the number of samples evaluated per iteration"
        ),
    )
    common_group.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug CLI logs",
    )
    common_group.add_argument(
        "--prompt_mode",
        type=PromptPhase,
        choices=[m.value for m in PromptPhase],
        default="training",
        help="Defines prompt structure to use",
    )
    common_group.add_argument(
        "--assumption_mode",
        type=AssumptionMode,
        choices=[m.value for m in AssumptionMode],
        default="none",
        help="Defines whether the model will be optimistic, pessimistic or neutral",
    )
    common_group.add_argument(
        "--add_hierarchy",
        action="store_true",
        help="Add cwe hierarchy guidelines in system prompt",
    )
    common_group.add_argument(
        "--prompt_version",
        "-v",
        type=str,
        choices=["v1", "v2"],
        help="Prompt version to use.",
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
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Number of epochs to finetune the model for",
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
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient clipping to apply during fine-tuning",
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
    finetune_group.add_argument(
        "--batch_size_eval",
        type=int,
        default=8,
        help="How many samples in a per device batch for evaluation steps",
    )
    finetune_group.add_argument(
        "--target_vulnerable_ratio",
        type=float,
        help="Balance the vulnerable portion of the training set by duplicating the vulnerable samples to reach specified ration",
    )
    finetune_group.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from. If omitted, starts fresh.",
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
        "--assets_dir",
        "-ad",
        type=Path,
        help="Directory where to save the generated plots and reports to (inference only)",
    )
    inference_group.add_argument(
        "--evaluated_test_path",
        type=Path,
        help="Directory where to save the test datset with evalutions to (inference only)",
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
    inference_group.add_argument(
        "--load_test_from_disk",
        action="store_true",
        help="Whether to load a previously evaluated dataset from disk",
    )

    return parser


from dataclasses import dataclass
@dataclass(frozen=True)
class FlagInfo:
    attr_name: str
    flag: str


def check_required_args(args, required: list[FlagInfo]):
    """Check if required arguments are present."""
    for info in required:
        if getattr(args, info.attr_name) is None:
            yield info.flag


def validate_args(args):
    """
    Validates that only mode-specific arguments are used with their respective modes.
    Call this after parsing arguments.
    """

    # peft only args
    finetune_only = {
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "batch_size_eval",
        "target_vulnerable_ratio",
        "resume_from_checkpoint",
    }

    # hpo only args
    hpo_only = {"n_trials", "run_cap"}

    # test only args
    inference_only = {
        "lora_weights",
        "max_tokens_per_answer",
        "evaluated_test_path",
        "assets_dir",
        "use_batching",
        "include_code_in_reports",
        "save_artifacts",
        "load_test_from_disk"
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

    required: list[FlagInfo] = [
        FlagInfo(attr_name="formatted_dataset_dir", flag="--formatted_dataset_dir"),
        FlagInfo(attr_name="chat_template", flag="--chat_template"),

    ]

    if args.finetune:
        required.extend([
            FlagInfo(attr_name="base_model_name", flag="--base_model_name"),
            FlagInfo(attr_name="dataset_path", flag="--dataset_path"),
        ])
        missing = list(check_required_args(args=args, required=required))
        if missing:
            raise argparse.ArgumentTypeError(
                f"Required for --finetune mode: {', '.join(missing)}"
            )

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
        required.extend([
            FlagInfo(attr_name="base_model_name", flag="--base_model_name"),
            FlagInfo(attr_name="dataset_path", flag="--dataset_path"),
        ])
        missing = list(check_required_args(args=args, required=required))
        if missing:
            raise argparse.ArgumentTypeError(
                f"Required for --hpo mode: {', '.join(missing)}"
            )

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
        if not args.load_test_from_disk:
            required.extend([
                FlagInfo(attr_name="lora_weights", flag="--lora_weights"),
            ])
            missing = list(check_required_args(args=args, required=required))
            if missing:
                raise argparse.ArgumentTypeError(
                    f"Required for --inference mode: {', '.join(missing)}"
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
        "epochs": None,
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
        "max_grad_norm": 1.0,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0,
        "batch_size_eval": 8,
        "target_vulnerable_ratio": None,
        "resume_from_checkpoint": None,
        # HPO only
        "n_trials": 5,
        "run_cap": 100,
        # Inference only
        "lora_weights": None,
        "max_tokens_per_answer": 2048,
        "evaluated_test_path": None,
        "assets_dir": None,
        "use_batching": False,
        "include_code_in_reports": False,
        "save_artifacts": False,
        "load_test_from_disk": False
    }
    return defaults.get(arg_name)


def save_running_args(args: argparse.Namespace):
    date: str = datetime.today().strftime("%Y-%m-%d")
    time: str = datetime.now().strftime("%H-%M-%S")
    fp = Path(__file__).parent / f"run/{date}/{time}/cli_args.json"
    fp.parent.mkdir(exist_ok=True, parents=True)
    with open(file=fp, mode="w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, cls=PathEncoder)

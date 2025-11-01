import os
import gc
import argparse
import json
import logging
from pathlib import Path

from accelerate import Accelerator

from .logging_config import setup_logger
from .judges import JudgeConfig, JudgeEnsemble
from .utilities import setup_paths, build_table, rich_panel, rich_exception, rich_rule, is_main_process, cleanup_resources
from .plots import visualize_results

from rich.traceback import install
install(show_locals=True)

setup_logger()
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(
        prog="Chain of thougths quality assessment",
        description="A jury that judges the quality of CoTs",
    )

    path_group = parser.add_argument_group("Path mandatory arguments")
    path_group.add_argument("--input", "-i", type=Path, required=True, help="Path to the source dataset.")
    path_group.add_argument("--output", "-o", type=Path, required=True, help="Output folder.")
    path_group.add_argument("--assets", "-p", type=Path, required=True, help="Assets folder.")

    # -- Generation --
    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument("--max_new_tokens", "-m", type=int, default=512, help="Maximum tokens for generation completion.")

    # -- Assessment --
    assessment_group = parser.add_argument_group("Metrics arguments")
    assessment_group.add_argument(
        "--quality_threshold", "-q",
        type=float, default=0.60,
        help="Minimum average quality (0-1).",
    )
    assessment_group.add_argument(
        "--agreement_threshold", "-a",
        type=float, default=0.75,
        help="Minimum judge agreement (0-1, higher=stricter). Reject if judges differ by more than (1-agreement_threshold)",
    )
    assessment_group.add_argument(
        "--agreement_method", "-t",
        type=str, default="weighted_multidimensional", choices=["multidimensional", "weighted_multidimensional"],
        help="Type of agreement to compute",
    )

    args = parser.parse_args()
    accelerator = Accelerator()
    try:

        cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

        if is_main_process():
            data = {
                "Distributed type": accelerator.distributed_type,
                "Num processes": accelerator.num_processes,
                "Mixed precision": accelerator.mixed_precision,
                "Num SLURM CPUs": cpus_allocated,
            }
            if accelerator.state.deepspeed_plugin is not None:
                data["DEEPSPEED"] = "ENABLED"
                data["ZeRO Stage"] = accelerator.state.deepspeed_plugin.zero_stage
                data["Offload optimizer"]=accelerator.state.deepspeed_plugin.offload_optimizer_device
                data["Offload params"]=accelerator.state.deepspeed_plugin.offload_param_device
            else:
                data["DEEPSPEED"] = "DISABLED"

            table=build_table(data=data, title="", columns=["Parameter", "Value"])
            rich_panel(table, panel_title="Environment configuration", subtitle="", align="center")

            del data, table
            gc.collect()
        rich_rule()

        paths = setup_paths(parser)

        judge_configs = [
            JudgeConfig(
                model_name="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
                chat_template="qwen-2.5",
                specialization="code",
                description="Specialized in C/C++ vulnerability patterns and code analysis"
            ),
            JudgeConfig(
                model_name="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                chat_template="llama-3.1",
                specialization="reasoning",
                description="Deep reasoning model for logical flow and completeness"
            ),
            JudgeConfig(
                model_name="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
                chat_template="qwen-2.5",
                temperature=0.6,
                top_p=0.95,
                min_p=0.05,
                specialization="logic",
                description="Mathematical and logical reasoning specialist"
            ),
        ]

        # =========================================================================
        # Run Filtering
        # =========================================================================
        ensemble = JudgeEnsemble(judge_configs)
        ensemble_info = ensemble.get_ensemble_info()
        with open(file=paths["metadata"], mode="w", encoding="utf-8") as f:
            json.dump(ensemble_info, f, indent=2)

        stats = ensemble.filter_dataset_streaming(
            input_jsonl_path=args.input,
            output_jsonl_path=paths["filtered"],
            rejected_jsonl_path=paths["rejected"],
            stats_json_path=paths["filtering_stats"],
            quality_threshold=args.quality_threshold,
            agreement_threshold=args.agreement_threshold,
            save_interval=15,
            agreement_method=args.agreement_method
        )

        visualize_results(
            stats=stats,
            output_dir=args.assets,
            quality_threshold=args.quality_threshold,
            agreements_threshold=args.agreement_threshold,
        )

    except Exception:
        rich_exception()
    finally:
        cleanup_resources(accelerator=accelerator)


if __name__ == "__main__":
    main()

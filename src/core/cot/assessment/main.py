import os
import gc
import json
import logging

from accelerate import Accelerator

from .cli import get_parser, validate_args
from .logging_config import setup_logger
from .judges import JudgeConfig, JudgeEnsemble, SingleJudgeEvaluator
from .utilities import setup_paths, build_table, rich_panel, rich_exception, rich_rule, is_main_process, cleanup_resources
from .plots import visualize_results

from rich.traceback import install
install(show_locals=True)

setup_logger()
logger = logging.getLogger(__name__)


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)

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

        judge_configs: dict[str, JudgeConfig] = {
            "qwen-coder": JudgeConfig(
                model_name="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
                ref_name="Qwen2.5-Coder-32B",
                chat_template="qwen-2.5",
                specialization="code",
                description="Specialized in C/C++ vulnerability patterns and code analysis"
            ),
            "llama-3.1-70B": JudgeConfig(
                model_name="unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                ref_name="Llama-3.1-70B",
                chat_template="llama-3.1",
                specialization="reasoning",
                description="Deep reasoning model for logical flow and completeness"
            ),
            "deepseek-qwen": JudgeConfig(
                model_name="unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit",
                chat_template="qwen-2.5",
                ref_name="DeepSeek-R1-Distill-Qwen-32B",
                temperature=0.6,
                top_p=0.95,
                min_p=0.05,
                specialization="logic",
                description="Mathematical and logical reasoning specialist"
            ),
        }

        # =========================================================================
        # Run Filtering
        # =========================================================================
        if args.ensemble:
            ensemble = JudgeEnsemble(list(judge_configs.values()))
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
                save_interval=args.save_interval,
                agreement_method=args.agreement_method
            )

            visualize_results(
                stats=stats,
                output_dir=args.assets,
                quality_threshold=args.quality_threshold,
                agreements_threshold=args.agreement_threshold,
            )
        else:
            evaluator = SingleJudgeEvaluator(judge_configs[args.judge])
            evaluator.evaluate_dataset(
                input_jsonl=args.input,
                output_jsonl=args.output_path,
                save_interval=args.save_interval
            )
    except Exception:
        rich_exception()
    finally:
        cleanup_resources(accelerator=accelerator)


if __name__ == "__main__":
    main()

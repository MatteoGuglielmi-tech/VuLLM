import gc
import json
import logging

from .cli import get_parser, validate_args
from .logging_config import setup_logger
from .judges import JudgeConfig, JudgeEnsemble, SingleJudgeEvaluator
from .utilities import (
    setup_paths,
    rich_exception,
    display_env_info,
    init_accelerator,
    cleanup_resources,
    cleanup_single_gpu,
)
from .plots import visualize_results
from .merge_script import merge_and_filter

import torch

from rich.traceback import install
install(show_locals=True)

setup_logger()
logger = logging.getLogger(__name__)


def main():
    parser = get_parser()
    args = parser.parse_args()
    validate_args(args)
    paths = setup_paths(args)

    accelerator = init_accelerator()
    try:

        display_env_info(parser=parser, args=args)
        gc.collect()

        judge_configs: dict[str, JudgeConfig] = {
            "qwen-coder": JudgeConfig(
                model_name="unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
                ref_name="Qwen2.5-Coder-32B",
                chat_template="qwen-2.5",
                max_seq_length=args.max_length if args.sequential else args.max_lengths[0],
                max_new_tokens=args.max_new_tokens,
                specialization="code",
                description="Specialized in C/C++ vulnerability patterns and code analysis"
            ),
            "qwen-72b": JudgeConfig(
                model_name="unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
                ref_name="Qwen2.5-72B",
                chat_template="qwen-2.5",
                max_seq_length=args.max_length if args.sequential else args.max_lengths[1],
                max_new_tokens=args.max_new_tokens,
                specialization="logic",
                description="Logical reasoning and evaluation specialist"
            ),
            "phi-4": JudgeConfig(
                model_name="unsloth/Phi-4-reasoning-plus-unsloth-bnb-4bit",
                ref_name="Phi-4",
                chat_template="phi-4",
                max_seq_length=args.max_length if args.sequential else args.max_lengths[2],
                max_new_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                specialization="reasoning",
                description="Mathematical and logical reasoning specialist"
            ),
            "llama-3.3": JudgeConfig(
                model_name="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
                ref_name="Llama-3.3-70B",
                chat_template="llama-3.3",
                max_seq_length=args.max_length if args.sequential else args.max_lengths[3],
                max_new_tokens=args.max_new_tokens,
                specialization="strong baseline",
                description="Multilinugal model"
            ),
            "deepseek-llama": JudgeConfig(
                model_name="unsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit",
                ref_name="DeepSeek-R1-Distill-Llama",
                chat_template="llama-3.3",
                max_seq_length=args.max_length if args.sequential else args.max_lengths[4],
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                min_p=0.05,
                specialization="reasoning",
                description="Mathematical and logical reasoning specialist"
            ),
        }

        # =========================================================================
        # Run Filtering
        # =========================================================================
        if args.ensemble:
            assert paths is not None
            ensemble = JudgeEnsemble(list(judge_configs.values()))
            ensemble_info = ensemble.get_ensemble_info()
            with open(file=paths["metadata"], mode="w", encoding="utf-8") as f:
                json.dump(ensemble_info, f, indent=2)

            stats = ensemble.filter_dataset_streaming(
                input_jsonl_path=args.input,
                output_kept=paths["filtered"],
                output_rejected=paths["rejected"],
                output_stats=paths["filtering_stats"],
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
        elif args.sequential:
            evaluator = SingleJudgeEvaluator(judge_configs[args.judge])
            evaluator.evaluate_dataset(
                input_jsonl=args.input,
                output_jsonl=args.output_path,
                save_interval=args.save_interval
            )
        else: # merge
            assert paths is not None
            stats = merge_and_filter(
                original_data=args.input_jsonl,
                judge_files=[args.judge1, args.judge2, args.judge3],
                judge_configs=list(judge_configs.values()),
                output_kept=paths["filtered"],
                output_rejected=paths["rejected"],
                stats_json_path=paths["filtering_stats"],
                agreement_method=args.agreement_method,
                agreement_threshold=args.agreement_threshold,
                quality_threshold=args.quality_threshold,
                save_interval=args.save_interval,
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
        if torch.cuda.device_count() > 1:
            cleanup_resources(accelerator=accelerator)
        else:
            cleanup_single_gpu()


if __name__ == "__main__":
    main()

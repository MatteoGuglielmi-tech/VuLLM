import os
import gc
import logging
import warnings

from pathlib import Path
from dotenv import load_dotenv
from typing import cast

from datasets import Dataset
from rich.traceback import install

from .utilities import (
    rich_table,
    rich_exception,
    rich_rule,
    is_main_process,
    cleanup_resources,
    cleanup_single_gpu,
    init_accelerator,
    display_env_info,
)
from .processing_lib import (
    DatasetHandler,
    ModelHandler,
    FineTuningHandler,
    LLMHyperparameterOptimizer,
    TestHandler,
    Evaluator,
)
from .logging_config import setup_logger
from . import cli

import torch

install(show_locals=True)


warnings.filterwarnings(
    "ignore",
    message=".*`num_logits_to_keep` and `logits_to_keep` are set.*",
    category=FutureWarning,
)

logger = logging.getLogger(name=__name__)


if __name__ == "__main__":

    load_dotenv()
    setup_logger()

    parser = cli.get_parser()
    args = parser.parse_args()
    cli.validate_args(args)
    cli.save_running_args(args)

    cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    accelerator = init_accelerator()
    display_env_info(parser=parser, args=args)
    gc.collect()

    try:
        if not args.inference and not args.hpo:
            rich_rule(f"🚀 [bold][italic][green] Starting fine-tuning pipeline [/green][/italic][/bold] 🚀")

            # --- Fine-Tuning ---
            fine_tuner = FineTuningHandler(
                dataset_handler_class=DatasetHandler,
                model_loader_class=ModelHandler,

                # -- dataset --
                dataset_path=args.dataset_path,
                formatted_dataset_dir=Path(args.formatted_dataset_dir),
                num_cpus=cpus_allocated,

                # -- model loading --
                base_model_name=args.base_model_name,
                chat_template=args.chat_template,
                max_seq_length=args.max_seq_length,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_rslora=args.use_rslora,
                use_loftq=args.use_loftq,

                # -- fine-tuning HP --
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_acc_steps,
                weight_decay=args.weight_decay,
                logging_steps=args.logging_steps,
                use_deepspeed=args.deepspeed,

                # -- flags --
                debug=args.debug,
            )
            fine_tuner.fine_tune()
            rich_rule(style="green")

        elif args.hpo:
            rich_rule(f"🚀 [bold][italic][light_salmon1] Starting HPO pipeline [/][/][/] 🚀")

            optimizer = LLMHyperparameterOptimizer(
                # Dataset parameters
                dataset_handler_class=DatasetHandler,
                dataset_path=args.dataset_path,
                formatted_dataset_dir=Path(args.formatted_dataset_dir),
                num_cpus=cpus_allocated,

                # Model parameters
                model_loader_class=ModelHandler,
                base_model_name=args.base_model_name,
                chat_template=args.chat_template,
                max_seq_length=args.max_seq_length,
                use_rslora=args.use_rslora,
                use_loftq=args.use_loftq,

                # HPO parameters
                n_trials=args.n_trials,
                epochs=args.epochs,
                max_steps_per_trial=args.run_cap,
                output_dir="./hpo_results",

                # Training parameters
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.grad_acc_steps,
                logging_steps=args.logging_steps,
                eval_steps=args.eval_steps,
                use_deepspeed=args.deepspeed,
            )

            # Run HPO
            best_params, best_score = optimizer.hpo()

            if is_main_process():
                rich_table(
                    data=best_params,
                    title="🎯 HPO: best hyperparameters found 🎯",
                    columns=["Parameter", "Best value"],
                    post_desc=f"📉 Best validation loss: {best_score:.4f}",
                )

            rich_rule(syle="light_salmon1")
        else:
            rich_rule(f"🚀 [bold][italic][light_sky_blue1] Starting inference pipeline [/][/][/] 🚀")

            test_set = cast(Dataset, DatasetHandler.load_from_disk(fp=args.formatted_dataset_dir, split="test"))
            test_handler = TestHandler(
                lora_model_dir=args.lora_weights,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_tokens_per_answer,
                chat_template=args.chat_template,
            )

            dataset_with_perdictions = test_handler.evaluate_on_test_set(
                test_dataset=test_set,
                batch_size=args.batch_size,
                use_batching=args.use_batching,
            )

            evaluator = Evaluator(output_dir=args.assets_dir, test_dataset=dataset_with_perdictions)
            evaluator.validate_cwe_format() # validate predicted cwe quality
            binary_metrics = evaluator.evaluate_binary_classification(save_artifacts=args.save_artifacts) # address target performance
            cwe_results = evaluator.evaluate_cwe_classification(save_artifacts=args.save_artifacts) # address cwe performance

    except Exception as e:
        rich_exception()
    finally:
        if torch.cuda.device_count() > 1:
            cleanup_resources(accelerator=accelerator)
        else:
            cleanup_single_gpu()

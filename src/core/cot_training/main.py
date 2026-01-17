import os
import gc
import logging
import warnings

from pathlib import Path
from dotenv import load_dotenv

from datasets import Dataset
from rich.traceback import install

from .utilities import (
    rich_table,
    rich_exception,
    rich_rule,
    rich_panel,
    is_main_process,
    cleanup_resources,
    cleanup_single_gpu,
    init_accelerator,
    get_env_info,
    RichColors
)
from .processing_lib import (
    DatasetHandler,
    ModelHandler,
    FineTuningHandler,
    LLMHyperparameterOptimizer,
    TestHandler,
    TestHandlerPlain,
    TypedDataset,
    TestDatasetSchema,
    Evaluator,
)
from .logging_config import setup_logger
from . import cli

import torch

install(show_locals=False)


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
    gc.collect()

    try:
        if not args.inference and not args.hpo:
            rich_rule(
                f"🌩️ [bold][italic][cornflower_blue]Starting fine-tuning pipeline[/cornflower_blue][/italic][/bold] 🌩️",
                style="cornflower_blue",
            )

            # --- Fine-Tuning ---
            fine_tuner = FineTuningHandler(
                dataset_handler_class=DatasetHandler,
                model_loader_class=ModelHandler,

                # -- dataset --
                dataset_path=args.dataset_path,
                formatted_dataset_dir=Path(args.formatted_dataset_dir),
                num_cpus=cpus_allocated,
                target_vulnerable_ratio=args.target_vulnerable_ratio,
                prompt_mode=args.prompt_mode,
                assumption_mode=args.assumption_mode,

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
                per_device_eval_batch_size=args.batch_size_eval,
                gradient_accumulation_steps=args.grad_acc_steps,
                weight_decay=args.weight_decay,
                strategy=args.strategy,
                logging_steps=args.logging_steps,
                use_deepspeed=args.deepspeed,

                # -- flags --
                debug=args.debug,
            )

            run_tb = get_env_info(parser=parser, args=args)
            strategy_tb = fine_tuner._strat_settings(args.strategy)
            run_tb.append(strategy_tb)
            rich_panel(
                tables=run_tb,
                panel_title="Run settings",
                border_style="sea_green3",
            )

            fine_tuner.fine_tune()
            rich_rule(style="cornflower_blue")

        elif args.hpo:
            rich_rule(f"🚀 [bold][italic][light_salmon1] Starting HPO pipeline [/][/][/] 🚀")

            #warning: I don't have enough resources for this
            # TODO: apply strategy changes and eval_batch_size to HPO
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
                # use_deepspeed=args.deepspeed,
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

            rich_rule(style=RichColors.SALMON1)
        else:
            rich_rule(f"🚀 [bold][italic][light_sky_blue1] Starting inference pipeline [/][/][/] 🚀")

            if not args.load_test_from_disk:
                test_handler = TestHandlerPlain(
                # test_handler = TestHandler(
                    lora_model_dir=args.lora_weights,
                    max_seq_length=args.max_seq_length,
                    max_new_tokens=args.max_tokens_per_answer,
                    chat_template=args.chat_template,
                    evaluated_testset_path=args.evaluated_test_path,
                    prompt_mode=args.prompt_mode,
                    assumption_mode=args.assumption_mode
                )
                test_set: Dataset = TestHandler.load_test_dataset(input_dir=args.formatted_dataset_dir)
                pred_testset: TypedDataset[TestDatasetSchema] = test_handler.evaluate_on_test_set(
                    test_dataset=test_set,
                    batch_size=args.batch_size,
                    use_batching=args.use_batching,
                )
                # test_handler.diagnose_model(test_set, n_samples=10)
            else:
                pred_testset: TypedDataset[TestDatasetSchema] = TestHandler.load_test_dataset(
                    input_dir=args.evaluated_test_path, split_name="test", with_eval=True
                )

                evaluator = Evaluator(output_dir=args.assets_dir, test_typeddataset=pred_testset)
                evaluator.validate_cwe_format() # validate predicted cwe quality
                binary_metrics = evaluator.evaluate_binary_classification(save_artifacts=args.save_artifacts) # address target performance
                cwe_results = evaluator.evaluate_cwe_classification(save_artifacts=args.save_artifacts) # address cwe performance

    except (Exception, KeyboardInterrupt) as e:
        import sys

        rich_exception()
        sys.exit(1)
    finally:
        if torch.cuda.device_count() > 1:
            cleanup_resources(accelerator=accelerator)
        else:
            cleanup_single_gpu()

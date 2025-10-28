import os
import logging
import warnings
import argparse

from pathlib import Path
from dotenv import load_dotenv
from typing import cast

from datasets import Dataset
from accelerate import Accelerator
from rich.traceback import install

from .utilities import rich_table, rich_exception, rich_rule, is_main_process, cleanup_resources
from .processing_lib import DatasetHandler, ModelHandler, FineTuningHandler, LLMHyperparameterOptimizer, TestHandler, Evaluator
from .logging_config import setup_logger
from . import cli

install(show_locals=True)


warnings.filterwarnings(
    "ignore",
    message=".*`num_logits_to_keep` and `logits_to_keep` are set.*",
    category=FutureWarning,
)


logger = logging.getLogger(name=__name__)

if __name__ == "__main__":
    # -- loading configs --
    load_dotenv()
    setup_logger()

    parser = cli.get_parser()
    args = parser.parse_args()

    try:
        args = cli.validate_args(args)
        rich_table(data=args, title="✅ Arguments validated successfully! ✅ ", columns=["Argument", "Value"])
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    cli.save_running_args(args)

    cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    accelerator = Accelerator()

    rich_rule("[bold italic]Configuration data")
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

        rich_table(data=data, title="🔍 Diagnostic Information", columns=["Parameter", "Value"])

    try:
        if not args.inference and not args.hpo:
            rich_rule(f"🚀 [bold][italic] Starting fine-tuning pipeline [/][/] 🚀")

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
                eval_steps=args.eval_steps,
                logging_steps=args.logging_steps,
                use_deepspeed=args.deepspeed,

                # -- flags --
                debug=args.debug,
            )
            fine_tuner.fine_tune()
            rich_rule()

        elif args.hpo:
            rich_rule(f"🚀 [bold][italic] Starting HPO pipeline [/][/] 🚀")

            optimizer = LLMHyperparameterOptimizer(
                # Dataset parameters
                dataset_handler_class=DatasetHandler,
                dataset_path=args.dataset_path,
                formatted_dataset_dir=Path(args.tokenized_dataset_dir),
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

            rich_rule()
        else:
            rich_rule(f"🚀 [bold][italic] Starting inference pipeline [/][/] 🚀")

            test_set = cast(Dataset, DatasetHandler.load_from_disk(fp=args.formatted_dataset_dir, split="test"))
            test_handler = TestHandler(
                lora_model_dir=args.lora_weights,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_tokens_per_answer,
                chat_template=args.chat_template_inference
            )
            evaluator = Evaluator(output_dir=args.assets_dir, test_dataset=test_set)

            _, predictions = test_handler.evaluate_on_test_set(test_dataset=test_set, batch_size=args.batch_size, use_batching=args.use_batching)
            binary_results = evaluator.evaluate_binary_classification(predictions=predictions, save_artifacts=True)
            cwe_results = evaluator.evaluate_cwe_classification(predictions=predictions, save_artifacts=True)
            misclass_results = evaluator.analyze_misclassifications(
                predictions=predictions,
                save_artifacts=True,
                include_code=args.include_code_in_reports,
                max_response_length=args.max_tokens_per_answer,
            )

            if is_main_process():
                rich_rule(f" ✅[bold][italic] EVALUATION COMPLETE - SUMMARY [/][/]✅")

                data = {
                    "Accuracy": round(binary_results.accuracy, 3),
                    "F1 (Vulnerable)": round(binary_results.f1_vulnerable, 2),
                    "Valid samples (%)": round((binary_results.valid_samples / binary_results.total_samples) * 100, 2),
                    "Unparsable(%)": round(( binary_results.unparsable_samples / binary_results.total_samples) * 100, 2),
                }
                rich_table(data=data,title="📊 Binary Classification", columns=["Metric", "Value"])

                print("\n")
                data = {
                    "Macro-avg F1": round(cwe_results.macro_avg_f1, 3),
                    "Micro-avg F1": round(cwe_results.micro_avg_f1, 2),
                    "Unique CWEs": len(cwe_results.all_cwes),
                    "Valid samples": cwe_results.valid_samples,
                    "Missing CWEs": cwe_results.samples_missing_cwes,
                }
                rich_table(data=data, title="📊 CWE Classification", columns=["Metric", "Value"])

                print("\n")
                data = {
                    "Total errors": misclass_results.total_errors,
                    "Error rate": round(misclass_results.error_rate, 2),
                    "False Positives": misclass_results.false_positives,
                    "False Negatives": misclass_results.false_negatives,
                }
                rich_table(data=data, title="📊 Misclassifications", columns=["Metric", "Value"])

                logger.info(f"\n📁 All artifacts saved to: {args.assets_dir}")

            rich_rule()

            if args.save_summary:
                evaluator.save_evaluation_summary(
                    output_dir=Path(args.assets_dir),
                    binary_results=binary_results,
                    cwe_results=cwe_results,
                    misclass_results=misclass_results,
                )
    except Exception as e:
        rich_exception()
    finally:
        cleanup_resources(accelerator=accelerator)

from src.core.cot_training.processing_lib.finetuning_handler import FineTuningHandler
from src.core.cot_training.processing_lib.dataset_handler import DatasetHandler
from src.core.cot_training.processing_lib.model_handler import ModelHandler
from src.core.cot_training.processing_lib.inference_handler import TestHandler
from src.core.cot_training.processing_lib.evaluation_handler import Evaluator
from src.core.cot_training.processing_lib.hpo_handler import LLMHyperparameterOptimizer

import os
import logging
import warnings
import argparse

from pathlib import Path
from dotenv import load_dotenv
from typing import cast
from datasets import Dataset
from accelerate import Accelerator

from src.core.cot_training.logging_config import setup_logger
from src.core.cot_training import cli


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
        logger.debug("Arguments validated successfully!")
        logger.debug(f"Mode: {'finetune' if args.finetune else 'hpo' if args.hpo else 'inference'}")
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    cli.save_running_args(args)

    accelerator = Accelerator()
    
    # Print config info only from main process
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("🚀 Training Configuration")
        print("="*50)
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        
        if accelerator.state.deepspeed_plugin is not None:
            print(f"✅ DeepSpeed ENABLED")
            print(f"   ZeRO Stage: {accelerator.state.deepspeed_plugin.zero_stage}")
            print(f"   Offload optimizer: {accelerator.state.deepspeed_plugin.offload_optimizer_device}")
            print(f"   Offload params: {accelerator.state.deepspeed_plugin.offload_param_device}")
        else:
            print("❌ DeepSpeed NOT enabled")
        print("="*50 + "\n")

    logger.info("🚀 Starting baseline... 🚀")
    cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    if not args.inference and not args.hpo:
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
            is_main_process=accelerator.is_main_process
        )
        fine_tuner.fine_tune()
    elif args.hpo:
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

        print(f"\n🎯 Best hyperparameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\n📉 Best validation loss: {best_score:.4f}")
    else:
        test_set = cast(Dataset, DatasetHandler.load_from_disk(fp=args.formatted_dataset_dir, split="test"))
        test_handler = TestHandler(
            lora_model_dir=args.lora_weights,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_tokens_per_answer,
            chat_template=args.chat_template_inference
        )
        evaluator = Evaluator(output_dir=args.assets_dir, test_dataset=test_set)

        _, predictions = test_handler.evaluate_on_test_set(test_dataset=test_set)
        binary_results = evaluator.evaluate_binary_classification(predictions=predictions, save_artifacts=True)
        cwe_results = evaluator.evaluate_cwe_classification(predictions=predictions, save_artifacts=True)
        misclass_results = evaluator.analyze_misclassifications(
            predictions=predictions,
            save_artifacts=True,
            include_code=args.include_code_in_reports,
            max_response_length=args.max_tokens_per_answer,
        )

        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE - SUMMARY")
        print("=" * 80)

        print("\n📊 Binary Classification:")
        print(f"   Accuracy:           {binary_results.accuracy:.2%}")
        print(f"   F1 (Vulnerable):    {binary_results.f1_vulnerable:.2%}")
        print(f"   Valid samples:      {binary_results.valid_samples}/{binary_results.total_samples}")
        print(f"   Unparsable:         {binary_results.unparsable_samples} ({binary_results.unparsable_samples/binary_results.total_samples:.2%})")

        print("\n📊 CWE Classification:")
        print(f"   Macro-avg F1:       {cwe_results.macro_avg_f1:.2%}")
        print(f"   Micro-avg F1:       {cwe_results.micro_avg_f1:.2%}")
        print(f"   Unique CWEs:        {len(cwe_results.all_cwes)}")
        print(f"   Valid samples:      {cwe_results.valid_samples}")
        print(f"   Missing CWEs:       {cwe_results.samples_missing_cwes}")

        print("\n📊 Misclassifications:")
        print(f"   Total errors:       {misclass_results.total_errors} ({misclass_results.error_rate:.2%})")
        print(f"   False Positives:    {misclass_results.false_positives}")
        print(f"   False Negatives:    {misclass_results.false_negatives}")

        print(f"\n📁 All artifacts saved to: {args.assets_dir}")
        print("=" * 80)

        if args.save_summary:
            evaluator.save_evaluation_summary(
                output_dir=Path(args.assets_dir),
                binary_results=binary_results,
                cwe_results=cwe_results,
                misclass_results=misclass_results,
            )

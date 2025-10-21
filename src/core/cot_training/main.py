from src.core.cot_training.processing_lib.finetuning_handler import FineTuningHandler
from src.core.cot_training.processing_lib.dataset_handler import DatasetHandler
from src.core.cot_training.processing_lib.model_handler import ModelHandler
from src.core.cot_training.processing_lib.inference_handler import TestHandler
from src.core.cot_training.processing_lib.hpo_handler import LLMHyperparameterOptimizer

import os
import logging
import warnings
import argparse

from pathlib import Path
from dotenv import load_dotenv
from typing import cast
from datasets import Dataset

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
        print("Arguments validated successfully!")
        print(f"Mode: {'finetune' if args.finetune else 'hpo' if args.hpo else 'inference'}")
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    cli.save_running_args(args)

    logger.info("🚀 Starting baseline... 🚀")
    cpus_allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    if not args.inference and not args.hpo:
        # --- Fine-Tuning ---
        fine_tuner = FineTuningHandler(
            dataset_handler_class=DatasetHandler,
            model_loader_class=ModelHandler,

            # -- dataset --
            dataset_path=args.dataset_path,
            formatted_dataset_path=Path(args.formatted_dataset_path),
            num_cpus=cpus_allocated,

            # -- model loading --
            base_model_name=args.model_name,
            chat_template=args.chat_template,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_rslora=args.use_rslora,
            use_loftq=args.use_loftq,

            # -- fine-tuning HP --
            learning_rate=args.learning_rate,
            training_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            weight_decay=args.weight_decay,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
        )
        fine_tuner.fine_tune()
    elif args.hpo:
        optimizer = LLMHyperparameterOptimizer(
            # Dataset parameters
            dataset_handler_class=DatasetHandler,
            dataset_path=args.dataset_path,
            formatted_dataset_path=Path(args.tokenized_dataset_path),
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
            num_train_epochs=args.epochs,
            max_steps_per_trial=args.run_cap,
            output_dir="./hpo_results",

            # Training parameters
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc_steps,
            warmup_steps=10,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
        )

        # Run HPO
        best_params, best_score = optimizer.hpo()

        print(f"\n🎯 Best hyperparameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        print(f"\n📉 Best validation loss: {best_score:.4f}")
    else:
        test_set = cast(Dataset, DatasetHandler.load_from_disk(fp=Path(args.tokenized_dataset_path), split="test"))
        test_handler = TestHandler(
            lora_model_dir=Path(args.lora_weights),
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_tokens_per_answer,
            chat_template=args.chat_template_inference
        )

        result_dataset = test_handler.evaluate_on_test_set(test_dataset=test_set)

        TestHandler.quantitative_evaluation(all_predictions=result_dataset["model_prediction"], test_dataset=test_set)
        TestHandler.qualitative_evaluation(all_predictions=result_dataset["model_prediction"], test_dataset=test_set)
        TestHandler.analyze_misclassifications(all_predictions=result_dataset["model_prediction"], test_dataset=test_set)

import argparse
import logging

from pathlib import Path
from typing import Any
from logging_config import setup_logger
from binary import BinaryModel
from cwe import CWEModel


setup_logger()
logger = logging.getLogger(__name__)


def get_parser():
    """Creates and returns the argument parser."""

    parser = argparse.ArgumentParser(prog="Random Forest Classifier")

    # -- Paths --
    parser.add_argument("--source", "-i", type=Path, required=True, help="Path to the source dataset.")
    parser.add_argument("--assets", "-o", type=Path, default="./assets", help="Path wherein saving generated images.")

    # -- Generation ---
    parser.add_argument("--test_size", "-t", type=float, default=0.2, help="Size of the test set split.")
    parser.add_argument("--n_tree", "-n", type=int, default=100, help="Number of estimators to generate for the RandomForestClassifier.")
    parser.add_argument("--n_iters", "-g", type=int, default=10, help="Number of models to train in hyperparameters tuning.")
    parser.add_argument("--n_folds", "-f", type=int, default=5, help="How many folds to build for cross-validation when tuning hyperparamters.")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random state seed.")

    # -- Flags ---
    parser.add_argument("--hp_tuning", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Perform hyperparameters tuning.")
    parser.add_argument("--only_target", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Perform classification on binary labels.")
    parser.add_argument("--only_cwe", type=bool, action=argparse.BooleanOptionalAction, default=False, help="Perform classification on cwe.")
    parser.add_argument("--all", type=bool, action=argparse.BooleanOptionalAction, default=True, help="Perform classification on both binary labels and cwe.")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    base_model_params: dict[str, Any] = {
        "source_path": args.source,
        "assets_folder": args.assets,
        "test_size": args.test_size,
        "n_estimators": args.n_tree,
        "seed": args.seed,
        "n_iters": args.n_iters,
        "n_folds": args.n_folds,
        "hp_tuning": args.hp_tuning
    }

    # Run pipeline for the binary target
    if args.all or args.only_target:
        binary_pipeline = BinaryModel(**base_model_params)
        binary_pipeline.run(feature_column="func", target_column="target")

    # Run pipeline for the CWE target
    if args.all or args.only_cwe:
        cwe_pipeline = CWEModel(**base_model_params)
        cwe_pipeline.run(feature_column="func", target_column="cwe")

    logger.info("✅ --- Pipeline Finished --- ✅")

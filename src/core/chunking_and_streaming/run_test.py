import pandas as pd
from argparse import Namespace
from core.chunking_and_streaming.shared.argparser import get_parser
from core.chunking_and_streaming.unsloth_test import UnslothTestPipeline

from core.chunking_and_streaming.dataset import DatasetHandler
from transformers import AutoTokenizer
from .run_fine_tune import BASE_MODEL

from ...common.logging_config import setup_logger


if __name__ == "__main__":
    args: Namespace = get_parser().parse_args()
    setup_logger()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=BASE_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # split data
    data_handler = DatasetHandler(tokenizer=tokenizer, max_chunk_tokens=args.max_seq_length)
    data_handler.execute_base()
    tokenized_train_val_chunks = data_handler.tokenize_chunks_train()

    print("\n--- 🚀 Starting Evaluation Process ---")
    # load the chunked (but not tokenized) test data for evaluation
    df_test = pd.read_json(path_or_buf=data_handler.pth_final_test, lines=True)
    inference_pipeline = UnslothTestPipeline(lora_model_dir=args.lora_model_dir, max_seq_length=args.max_seq_length)
    evaluation_results_df = inference_pipeline.evaluate_on_test_set(df_test_data=df_test)

    UnslothTestPipeline.calculate_and_save_metrics(
        y_true=evaluation_results_df["ground_truth"].to_list(),
        y_pred=evaluation_results_df["predicted_label"].to_list(),
        output_dir=inference_pipeline.output_dir,
    )

    print(f"\n🎉 Pipeline complete! All evaluation artifacts saved in: {inference_pipeline.output_dir}")

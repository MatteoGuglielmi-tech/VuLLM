from argparse import Namespace

from core.chunking_and_streaming.shared.argparser import get_parser
from core.chunking_and_streaming.unsloth_fine_tuner import UnslothFineTunePipeline

from core.chunking_and_streaming.dataset import DatasetHandler
from transformers import AutoTokenizer

from ...common.logging_config import setup_logger


BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


if __name__ == "__main__":
    args: Namespace = get_parser().parse_args()
    setup_logger()

    print("--- 🚀 Starting Data Preparation Process ---")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=BASE_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # split data
    data_handler = DatasetHandler(tokenizer=tokenizer, max_chunk_tokens=args.max_seq_length)

    # --- Caching Logic ---
    if data_handler.pth_tokenized_train.exists() and data_handler.pth_tokenized_val.exists():
        print("✅ Pre-tokenized data found. Loading from disk.")
        tokenized_train_val_chunks = data_handler.load_tokenized_dataset()
    else:
        print("Pre-tokenized data not found. Running full data preparation pipeline...")
        data_handler.execute_base()
        tokenized_train_val_chunks = data_handler.tokenize_chunks_train()
        data_handler.cleanup_intermediate_files()

    if not tokenized_train_val_chunks: raise RuntimeError("Data processing failed. No tokenized data was produced.")

    print("✅ Data preparation complete.")

    print("\n--- 🚀 Starting Training Process ---")
    training_pipeline = UnslothFineTunePipeline(
        hf_train_data=processed_data_splits["train"], # type: ignore
        hf_eval_data=processed_data_splits["val"],  # type: ignore
        base_model_str=BASE_MODEL,
        max_seq_length=args.max_seq_length,
        training_epochs=1,
        # training_steps=100,  # train for 100 steps
    )
    training_pipeline.execute()
    print("\n--- ✅ Training complete. ---")

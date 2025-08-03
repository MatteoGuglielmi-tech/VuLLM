import pandas as pd
import gc

# critical: order of imports here is important for Unsloth to work optimally
# First, import the modules that use Unsloth to apply patches
from core.chunking_and_streaming.unsloth_trainer import UnslothModel
from core.chunking_and_streaming.unsloth_inference import InferencePipeline

# Then, import other modules that use transformers
from core.chunking_and_streaming.dataset import DatasetHandler

import torch
from transformers import AutoTokenizer

# =================================================================================
# MAIN EXECUTION SCRIPT
# =================================================================================
if __name__ == "__main__":
    # <---- 1. Configuration ---->
    # Define all your parameters and paths in one place.
    BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    MAX_SEQ_LENGTH = 2048
    RAW_DATASET_PATH = "../../../Dataset/small_dataset.json"
    INLINE_DATASET_PATH = "./data/inline_dataset.jsonl"

    # <---- 2. Data Preparation Step ---->
    print("--- 🚀 Starting Data Preparation Process ---")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=BASE_MODEL)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Instantiate and run the data handling pipeline
    data_handler = DatasetHandler(
        pth_raw_dataset=RAW_DATASET_PATH,
        pth_inline_dataset=INLINE_DATASET_PATH,
        tokenizer=tokenizer,
        max_chunk_tokens=MAX_SEQ_LENGTH,
    )

    data_handler.DATASET_load_raw_dataset()
    data_handler.DATASET_project_based_split()
    data_handler.DATASET_chunk()

    # Get the final tokenized data for the trainer
    processed_data_splits = data_handler.DATASET_get_processed_data()

    if not processed_data_splits:
        raise RuntimeError("Data processing failed. No tokenized data was produced.")

    print("✅ Data preparation complete.")
    exit()

    # <---- 3. Training Step ---->
    print("\n--- 🚀 Starting Training Process ---")

    training_pipeline = UnslothModel(
        hf_train_data=processed_data_splits["train"],  # type: ignore
        hf_eval_data=processed_data_splits["val"],  # type: ignore
        base_model_str=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        training_steps=100,  # train for 100 steps
    )

    training_pipeline.unsloth_load_base_model()
    training_pipeline.unsloth_patch_model()
    training_pipeline.unsloth_start_training()

    print("\n--- ✅ Training complete. Adapters saved. ---")

    # <---- 4. Evaluation Step ---->
    print("\n--- 🚀 Starting Evaluation Process ---")

    # Get the path to the trained model adapters
    lora_model_path = training_pipeline.lora_model_dir

    # Cleanup memory before starting inference
    del training_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Load the chunked (but not tokenized) test data for evaluation
    # This file was created by data_handler.DATASET_chunk()
    df_test = pd.read_json(path_or_buf=data_handler.pth_chunked_test, lines=True)

    # # Re-create the 'label' column for metrics calculation
    # df_test["label"] = df_test["ground_truth"]

    # Instantiate the dedicated inference pipeline
    inference_pipeline = InferencePipeline(
        lora_model_dir=lora_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Run evaluation on the test set
    evaluation_results_df = inference_pipeline.evaluate_on_test_set(
        df_test_data=df_test
    )

    # Calculate and save all metrics
    InferencePipeline.calculate_and_save_metrics(
        y_true=evaluation_results_df["ground_truth"].to_list(),
        y_pred=evaluation_results_df["predicted_label"].to_list(),
        output_dir=inference_pipeline.output_dir,
    )

    print(
        f"\n🎉 Pipeline complete! All evaluation artifacts saved in: {inference_pipeline.output_dir}"
    )

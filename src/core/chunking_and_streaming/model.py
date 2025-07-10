import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from .stdout import logger


@dataclass
class ModelHandler:
    hf_train_data: Dataset
    hf_eval_data: Dataset
    # `df_test_data` should be the raw test data (e.g., pandas DataFrame) for inference
    # before it's processed into prompts, so we can extract original code/labels for evaluation.
    df_test_data: Optional[pd.DataFrame] = None

    # base model to fine-tune
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    # Max sequence length for the model's tokenizer.
    # This should match the tokenizer's max_length used in DatasetHandler
    max_seq_length: int = 1024

    def __post_init__(self):
        # Determine paths for checkpoints and trainer output
        provider, model_id = self.base_model.split("/")
        date: str = datetime.today().strftime("%Y-%m-%d")
        time: str = datetime.now().strftime("%H-%M-%S")
        common_suffix: str = os.path.join(provider, model_id, date, time)
        self.checkpoint_dir: str = os.path.join("./checkpoints/", common_suffix)
        self.trainer_dir: str = os.path.join("./trainer", common_suffix)

        # Ensure output directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.trainer_dir, exist_ok=True)

        # Initialize model and tokenizer to None, they will be loaded later
        self.model = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        # Fine-tuned model and tokenizer
        self.merged_model = None
        self.ft_tokenizer: Optional[PreTrainedTokenizer] = None

    def WB_init(self) -> None:
        """Initializes Weights & Biases for experiment tracking."""
        try:
            wandb.login()
            wandb.init(
                project=f"Fine-tune {os.path.split(self.base_model)[-1]} for vulnerability detection.",
                job_type="training",
                anonymous="allow",
            )
            logger.info("Weights & Biases initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")

    def BnB_configuration(self) -> BitsAndBytesConfig:
        """Configures BitsAndBytes for 4-bit quantization.

        Loads model with:
        - 4-bit quantization
        - double quantization
        - normalized float (nf4)
        - compute type: brain float 16 (bfloat16)
        """

        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("BitsAndBytes configuration created.")

        return bnb_config

    def MODEL_load(self) -> None:
        """Loads the base model with pre-trained weights and BitsAndBytes
        configuration."""
        logger.info("Initializing model...")
        try:
            model_params: dict = {
                "pretrained_model_name_or_path": self.base_model,
                "torch_dtype": torch.bfloat16,  # "auto"
                "quantization_config": self.BnB_configuration(),
                "device_map": "auto",
                "attn_implementation": "flash_attention_2",  # Use Flash Attention 2 for efficiency
            }

            # load model with pre-trained weights
            self.model = AutoModelForCausalLM.from_pretrained(**model_params)
            if self.model:
                # Disable cache for training
                self.model.config.use_cache = False
                # Tensor parallelism for better performance
                self.model.config.pretraining_tp = 1

            print(f"model type: {type(self.model)}")
            logger.info("Model loaded successfully.")
            # logger.info(f"Model footprint -> {self.model.get_memory_footprint()}") # Uncomment if needed
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def TOKENIZER_load(self) -> None:
        """Loads the pre-trained tokenizer."""

        logger.info("Initializing tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.base_model
            )
            print(f"tokenizer type: {type(self.tokenizer)}")
            if self.tokenizer:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"

            logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            raise

    def LoRA_config(self) -> LoraConfig:
        """Configures LoRA using the target modules, task type, and other
        arguments."""

        if self.model is None:
            raise ValueError("Model must be loaded before configuring LoRA.")

        # Function to find all linear layer names for LoRA
        def find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit  # Assuming 4-bit quantization
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split(".")
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if "lm_head" in lora_module_names:
                lora_module_names.remove(
                    "lm_head"
                )  # lm_head is usually not LoRA-adapted
            return list(lora_module_names)

        modules: list[str] = find_all_linear_names(self.model)
        logger.info(f"Identified LoRA target modules: {modules}")

        peft_config: LoraConfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,  # LoRA attention dimension (rank)
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            inference_mode=False,  # Set to True for inference after training
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            bias="none",  # Bias type for LoRA layers
            target_modules=modules,  # Modules to apply LoRA to
            # modules_to_save=["lm_head"], # Modules to save in addition to LoRA weights
            # This is typically used if you want to fine-tune the lm_head as well,
            # but often it's excluded from target_modules and not explicitly saved
            # unless it's a separate adapter.
        )
        logger.info("LoRA configuration created.")

        return peft_config

    def HF_SFTConfig(self) -> SFTConfig:
        """Configures the SFT (Supervised Fine-Tuning) arguments for the TRL
        Trainer."""

        # Ensure checkpoint_dir and trainer_dir are created
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.trainer_dir, exist_ok=True)

        training_arguments = SFTConfig(
            output_dir=self.checkpoint_dir,  # Directory to save checkpoints
            run_name=f"{os.path.basename(self.base_model)}-sft",  # Unique run name for WandB
            num_train_epochs=1,  # Number of training epochs
            per_device_train_batch_size=4,  # Batch size per device during training
            gradient_accumulation_steps=8,  # Number of steps before performing a backward/update pass
            gradient_checkpointing=True,  # Use gradient checkpointing to save memory
            optim="paged_adamw_8bit",  # Using paged optimizer for memory efficiency (8bit for better compatibility with 4bit quant)
            logging_steps=10,  # Log every 10 steps
            learning_rate=2e-4,  # Learning rate, based on QLoRA paper
            weight_decay=0.001,
            fp16=False,  # Disable fp16 if bf16 is enabled
            bf16=True,  # Enable bfloat16 for mixed precision training
            max_grad_norm=0.3,  # Max gradient norm based on QLoRA paper
            max_steps=-1,  # Train for num_train_epochs
            warmup_ratio=0.03,  # Warmup ratio based on QLoRA paper
            group_by_length=False,  # Set to True to group samples of similar length together
            lr_scheduler_type="cosine",  # Use cosine learning rate scheduler
            report_to="wandb",  # Report metrics to Weights & Biases
            eval_strategy="steps",  # Evaluate every eval_steps
            eval_steps=0.2,  # Evaluate every 20% of the epoch
            packing=False,  # Don't collapse small samples into one (keep as is from DatasetHandler)
            dataset_text_field="text",  # Column header where to find input prompt
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
                "max_length": self.max_seq_length,  # Ensure max_length is passed to SFTTrainer's internal tokenizer
            },
            # max_seq_length=self.max_seq_length,  # Set max sequence length for SFTTrainer
        )
        logger.info("SFTConfig (training arguments) created.")
        return training_arguments

    def HF_SFTTrainer(self) -> SFTTrainer:
        """Initializes and returns the SFTTrainer instance."""
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model and tokenizer must be loaded before initializing SFTTrainer."
            )

        trainer: SFTTrainer = SFTTrainer(
            model=self.model,
            args=self.HF_SFTConfig(),
            train_dataset=self.hf_train_data,
            eval_dataset=self.hf_eval_data,
            peft_config=self.LoRA_config(),
            processing_class=self.tokenizer,
        )
        logger.info("SFTTrainer initialized.")
        return trainer

    def TRAINER_run_training(self) -> None:
        """Runs the fine-tuning training process."""
        logger.info("Fine-tune started...")
        try:
            trainer: SFTTrainer = self.HF_SFTTrainer()
            trainer.train()
            logger.info("Fine-tune finished.")

            # Close run on W&B portal
            logger.info("Closing WandB portal...")
            # wandb.finish()

            # Enable caching after training
            if self.model:
                self.model.config.use_cache = True

            # Save trained model and tokenizer
            logger.info("Saving fine-tuned model and tokenizer...")
            # The SFTTrainer saves the PEFT adapter, not the merged model by default.
            # To save the merged model, we need to load the adapter and merge it.
            
            if self.model and self.tokenizer:
                self.model.save_pretrained(
                    save_directory=os.path.join(self.trainer_dir, "model")
                )
                self.tokenizer.save_pretrained(
                    save_directory=os.path.join(self.trainer_dir, "tokenzier")
                )
            # trainer.save_model(os.path.join(self.trainer_dir, "model"))
            logger.info(f"Model and tokenizer saved to {self.trainer_dir}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def MODEL_load_ft_model(self) -> None:
        """Loads the fine-tuned PEFT model and merges its adapters into the
        base model.

        Also loads the fine-tuned tokenizer.
        """
        logger.info("Loading fine-tuned model and tokenizer...")
        try:
            # Load fine-tuned tokenizer
            self.ft_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.base_model
                # os.path.join(
                #     self.trainer_dir,  "model"#"tokenizer"
                # )
            )
            if self.ft_tokenizer:
                if self.ft_tokenizer.pad_token is None:
                    self.ft_tokenizer.pad_token = self.ft_tokenizer.eos_token
                self.ft_tokenizer.padding_side = "right"

            # Load the PEFT model (adapter)
            peft_model_path = os.path.join(self.trainer_dir, "model")
            peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_path)

            # Merge adapters into the base model
            self.merged_model = peft_model.merge_and_unload(progressbar=True)
            print(f"type of merged model: type(self.merged_model)")

            logger.info(f"Fine-tuned model loaded and merged from {peft_model_path}")
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            raise

    def PEFT_MODEL_infer(self, input_prompts: list[str]) -> list[str]:
        """Performs inference using the merged fine-tuned model.

        Takes a list of prompt strings and returns a list of predicted
        answers ('YES'/'NO').
        """
        if self.merged_model is None or self.ft_tokenizer is None:
            raise ValueError(
                "Fine-tuned model and tokenizer must be loaded before inference."
            )

        logger.info("Starting inference...")
        self.merged_model.eval()  # Set model to evaluation mode

        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.ft_tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        y_pred: list[str] = []
        # labels: list[str] = ["YES", "NO"]  # Expected output labels

        for prompt in tqdm(input_prompts, desc="Generating predictions"):
            try:
                outputs = pipe(
                    inputs=prompt,
                    max_new_tokens=5,  # Allow slightly more tokens for output like 'YES' or 'NO'
                    do_sample=False,  # Use greedy decoding for deterministic answers
                    temperature=0.1,  # Low temperature for less randomness
                    return_full_text=False,  # Only return generated text
                )

                if outputs and isinstance(outputs, list) and len(outputs) > 0:
                    generated_text = outputs[0].get("generated_text", "").strip()
                    # The prompt expects "Answer 'YES' if vulnerable, 'NO' otherwise."
                    # The model will generate "YES" or "NO" after this.
                    # We need to extract the actual answer.

                    # Simple extraction: look for "YES" or "NO"
                    if "YES" in generated_text.upper():
                        y_pred.append("YES")
                    elif "NO" in generated_text.upper():
                        y_pred.append("NO")
                    else:
                        y_pred.append("UNKNOWN")  # Fallback for unexpected output
                        logger.warning(
                            f"Unexpected inference output: '{generated_text}' for prompt: '{prompt[:100]}...'"
                        )
                else:
                    y_pred.append("UNKNOWN")
                    logger.warning(
                        f"No output generated for prompt: '{prompt[:100]}...'"
                    )
            except Exception as e:
                y_pred.append("ERROR")
                logger.error(
                    f"Error during inference for prompt '{prompt[:100]}...': {e}"
                )

        logger.info("Inference completed.")
        return y_pred

    def MODEL_evaluate(self, y_true: list[str], y_pred: list[str]) -> None:
        """Evaluates the model's predictions against true labels.

        Expects 'YES'/'NO' for y_true and y_pred.
        """
        labels_map: dict[str, int] = {
            "NO": 0,
            "YES": 1,
        }  # Mapping for metrics calculations

        # Filter out 'UNKNOWN' or 'ERROR' predictions for evaluation metrics
        filtered_y_true = []
        filtered_y_pred = []
        for true_lbl, pred_lbl in zip(y_true, y_pred):
            if pred_lbl in ["YES", "NO"]:  # Only evaluate if prediction is valid
                filtered_y_true.append(labels_map[true_lbl])
                filtered_y_pred.append(labels_map[pred_lbl])
            else:
                logger.warning(
                    f"Skipping evaluation for invalid prediction: True='{true_lbl}', Pred='{pred_lbl}'"
                )

        if not filtered_y_true:
            logger.warning("No valid predictions to evaluate.")
            return

        y_true_mapped: np.ndarray = np.array(filtered_y_true)
        y_pred_mapped: np.ndarray = np.array(filtered_y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
        logger.info(f"Accuracy: {accuracy:.3f}")

        # Generate classification report
        # Ensure target_names and labels match the mapping
        target_names = ["NO", "YES"]
        report_labels = [0, 1]

        class_report = classification_report(
            y_true=y_true_mapped,
            y_pred=y_pred_mapped,
            target_names=target_names,
            labels=report_labels,
            zero_division="warn",
        )

        logger.info("\nClassification Report:")
        print(class_report)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(
            y_true=y_true_mapped, y_pred=y_pred_mapped, labels=report_labels
        )
        logger.info("\nConfusion Matrix:")
        print(conf_matrix)
        # WARN: headless server doesn't display plot
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        # disp.plot()
        # plt.show()


# Example usage (for demonstration, assuming DatasetHandler is available)
if __name__ == "__main__":

    # Create dummy tokenizer
    tokenizer_for_dummy = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    if tokenizer_for_dummy.pad_token is None:
        tokenizer_for_dummy.pad_token = tokenizer_for_dummy.eos_token
    tokenizer_for_dummy.padding_side = "right"

    # --- Dummy DatasetHandler setup to get processed_data ---
    # This part mimics the execution of DatasetHandler.py
    class DummyDatasetHandler:
        def __init__(self, tokenizer, max_chunk_tokens, trimming_technique):
            self.tokenizer = tokenizer
            self.max_chunk_tokens = max_chunk_tokens
            self.trimming_technique = trimming_technique
            self.pth_chunked_train = "./data/chunked_train_data.jsonl"
            self.pth_chunked_val = "./data/chunked_val_data.jsonl"
            self.pth_chunked_test = "./data/chunked_test_data.jsonl"

        def DATASET_get_processed_data(self):
            # Create dummy tokenized datasets (mimicking output of DatasetHandler.DATASET_get_processed_data)
            # Ensure 'text' column is present as SFTTrainer expects it
            dummy_train_data = Dataset.from_dict(
                {
                    "input_ids": [[1, 2, 3, 4, 5]],
                    "attention_mask": [[1, 1, 1, 1, 1]],
                    "text": ["You are an AI system...Correct answer:\nYES"],
                }
            )
            dummy_eval_data = Dataset.from_dict(
                {
                    "input_ids": [[6, 7, 8, 9, 10]],
                    "attention_mask": [[1, 1, 1, 1, 1]],
                    "text": ["You are an AI system...Correct answer:\nNO"],
                }
            )
            dummy_test_data = Dataset.from_dict(
                {
                    "input_ids": [[11, 12, 13, 14, 15]],
                    "attention_mask": [[1, 1, 1, 1, 1]],
                    "text": ["You are an AI system...Correct answer:\nNO"],
                }
            )
            return {
                "train": dummy_train_data,
                "val": dummy_eval_data,
                "test": dummy_test_data,
            }

    # Instantiate dummy DatasetHandler and get processed data
    dummy_dh = DummyDatasetHandler(
        tokenizer=tokenizer_for_dummy, max_chunk_tokens=100, trimming_technique="line"
    )
    processed_data = dummy_dh.DATASET_get_processed_data()

    # Create dummy test data (raw format, before chunking/prompting, for evaluation)
    dummy_test_df = pd.DataFrame(
        {
            "func": [
                "int main() { return 0; }",
                "void vulnerable() { char buf[10]; gets(buf); }",
            ],
            "target": ["0", "1"],  # Original labels
            "function_signature": ["int main()", "void vulnerable()"],
            "cwe": ["CWE-0", "CWE-1"],
            "project": ["projX", "projY"],
        }
    )

    # Initialize ModelHandler with specific datasets from processed_data
    if processed_data:
        model_handler = ModelHandler(
            hf_train_data=processed_data["train"],  # Pass the actual Dataset object
            hf_eval_data=processed_data["val"],  # Pass the actual Dataset object
            df_test_data=dummy_test_df,
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            max_seq_length=tokenizer_for_dummy.model_max_length,  # Ensure consistency
        )
    else:
        logger.error("Processed data is None, cannot initialize ModelHandler.")
        exit()  # Exit if data is not available

    # --- Pipeline Execution ---
    print("\n--- ModelHandler Pipeline Demonstration ---")

    # 1. Initialize WandB (optional, but good practice)
    # model_handler.WB_init() # Commented out for local dummy run to avoid actual W&B login

    # 2. Load Model and Tokenizer
    model_handler.MODEL_load()
    model_handler.TOKENIZER_load()

    # 3. Run Training (this will also save the model)
    # Note: Actual training requires GPU and proper environment setup.
    # This part will likely fail in a basic local environment without a GPU.
    try:
        logger.info("Attempting to run training. This might fail without a GPU.")
        model_handler.TRAINER_run_training()
    except Exception as e:
        logger.error(f"Training failed (expected in dummy setup without GPU): {e}")

    # For demonstration, let's ensure the dummy saved files exist for `MODEL_load_ft_model`
    dummy_trainer_model_path = os.path.join(model_handler.trainer_dir, "model")
    dummy_trainer_tokenizer_path = os.path.join(model_handler.trainer_dir, "tokenizer")
    os.makedirs(dummy_trainer_model_path, exist_ok=True)
    os.makedirs(dummy_trainer_tokenizer_path, exist_ok=True)
    # Create dummy files to simulate a saved model/tokenizer
    with open(
        os.path.join(dummy_trainer_model_path, "adapter_model.safetensors"), "w"
    ) as f:
        f.write("dummy")
    with open(os.path.join(dummy_trainer_tokenizer_path, "tokenizer.json"), "w") as f:
        f.write("dummy")
    with open(
        os.path.join(dummy_trainer_tokenizer_path, "special_tokens_map.json"), "w"
    ) as f:
        f.write("dummy")
    with open(
        os.path.join(dummy_trainer_tokenizer_path, "tokenizer_config.json"), "w"
    ) as f:
        f.write("dummy")

    # 4. Load Fine-tuned Model for Inference
    # model_handler.MODEL_load_ft_model()
    #
    # # 5. Prepare test prompts for inference
    # # This step would typically involve using DatasetHandler's logic to chunk and format
    # # the df_test_data into the inference prompt skeleton.
    # # For this dummy example, we'll manually create a couple of inference prompts
    # INFERENCE_PROMPT_SKELETON = (
    #     "You are an AI system that analyzes C code for vulnerabilities.\n\n"
    #     + "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
    #     + "KEY:Code is chunked; reassemble by function signature to obtain full original source code.\n"
    #     + "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
    #     + "Function signature:\n{signature}\n\n"
    #     + "Code Fragment:\n{subchunk}\n\n"
    #     + "Answer 'YES' if vulnerable, 'NO' otherwise."
    # ).strip()
    # inference_prompts = [
    #     INFERENCE_PROMPT_SKELETON.format(
    #         signature="int main()", subchunk="int x = 0; // This is safe code."
    #     ),
    #     INFERENCE_PROMPT_SKELETON.format(
    #         signature="void vulnerable()", subchunk="char buffer[10]; gets(buffer);"
    #     ),
    # ]
    #
    # # 6. Perform Inference
    # predictions = model_handler.PEFT_MODEL_infer(input_prompts=inference_prompts)
    # print(f"\nInference Predictions: {predictions}")
    #
    # # 7. Evaluate (using original labels from df_test_data)
    # # Note: This evaluation is simplified for the dummy case.
    # # In a real scenario, you'd align predictions with the original test data's labels.
    # # Here, we'll just use the dummy_test_df's targets mapped to "YES"/"NO".
    # true_labels_for_eval = [
    #     "NO",
    #     "YES",
    # ]  # Corresponds to dummy_test_df targets "0", "1"
    #
    # # Adjust predictions for evaluation if they are not exactly "YES"/"NO"
    # # The PEFT_MODEL_infer now returns "YES"/"NO"/"UNKNOWN"/"ERROR", so we use that directly.
    # model_handler.MODEL_evaluate(y_true=true_labels_for_eval, y_pred=predictions)
    #
    # print("\nModelHandler pipeline demonstration complete.")

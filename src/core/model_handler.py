import os
from dataclasses import dataclass
from datetime import datetime

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, TaskType
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)
from trl import SFTConfig, SFTTrainer

import wandb
# from dataset_handler import DatasetHandler
from log import logger


@dataclass
class ModelHandler:
    hf_train_data: Dataset
    hf_eval_data: Dataset
    df_test_data: pd.DataFrame
    # base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    # base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    def __post_init__(self):
        provider, model_id = self.base_model.split("/")
        date: str = datetime.today().strftime(format="%Y-%m-%d")
        time: str = datetime.now().strftime(format="%H-%M-%S")
        common_suffix: str = os.path.join(provider, model_id, date, time)
        self.checkpoint_dir: str = os.path.join("./checkpoints/", common_suffix)
        self.trainer_dir: str = os.path.join("./trainer", common_suffix)

    def WB_init(self) -> None:
        wandb.login()

        wandb.init(
            project=f"Fine-tune {os.path.split(self.base_model)[-1]} for vulnerability detection.",
            job_type="training",
            anonymous="allow",
        )

    def BnB_configuration(self) -> BitsAndBytesConfig:
        # load model with:
        # - 4-bit quantization
        # - double quantization
        # - normalized float
        # - compt type: brain float 16
        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

        return bnb_config

    def MODEL_load(self) -> None:
        # initialize model parameters
        logger.info(msg="Initializing model")
        model_params: dict = {
            "pretrained_model_name_or_path": self.base_model,
            "torch_dtype": "auto",
            "quantization_config": self.BnB_configuration(),
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
        }

        # load model with pre-trained weights
        self.model = AutoModelForCausalLM.from_pretrained(**model_params)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # print(f"Model footprint -> {self.model.get_memory_footprint()}")

    def TOKENIZER_load(self) -> None:
        # loading pre-trinaed tokenizer
        logger.info(msg="Initializing tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.base_model
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

    def LoRA_config(self) -> LoraConfig:
        # extracting the linear module names
        # configure LoRA using the target modules, task type, and other arguments before setting up training arguments
        def find_all_linear_names(model) -> list[str]:
            cls = bnb.nn.Linear4bit
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split(".")
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if "lm_head" in lora_module_names:
                lora_module_names.remove("lm_head")

            return list(lora_module_names)

        modules: list[str] = find_all_linear_names(self.model)
        # note:
        # good values for rank are 128, 256
        # rank = alpha
        peft_config: LoraConfig = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,  # the higher the rank, the more new content is impactful
            lora_alpha=16,  # scaling parameter, the higher the louder you shouting to the model
            inference_mode=False,  # training mode
            lora_dropout=0.1,  # dropout rate
            bias="none",
            target_modules=modules,  # module to applied LoRa to
            modules_to_save=["lm_head"],  # module to cache
        )

        return peft_config

    # text generation pipeline to predict labels from the “text” column
    def MODEL_predict(self, test_df: pd.DataFrame, model, tokenizer) -> list[str]:
        y_pred: list[str] = []
        labels: list[str] = ["0", "1"]

        for i in tqdm(range(len(test_df))):
            prompt = test_df.iloc[i]["text"]
            pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2,
                temperature=0.1,
            )

            result = pipe(inputs=prompt)
            assert result is not None
            answer = result[0]["generated_text"].split("label:")[-1].strip()

            # Determine the predicted category
            for category in labels:
                if category.lower() in answer.lower():
                    y_pred.append(category)
                    break
            else:
                y_pred.append("none")

        return y_pred

    # y_pred = predict(X_test, model, tokenizer)

    def MODEL_evaluate(self, y_true: list[str], y_pred: list[str]) -> None:
        labels: list[str] = ["0", "1"]
        mapping: dict[str, int] = {label: idx for idx, label in enumerate(labels)}

        def map_func(x):
            # Map to -1 if not found, but should not occur with correct data
            return mapping.get(x, -1)

        y_true_mapped: np.ndarray = np.vectorize(map_func)(y_true)
        y_pred_mapped: np.ndarray = np.vectorize(map_func)(y_pred)

        # Calculate accuracy
        accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
        print(f"Accuracy: {accuracy:.3f}")

        # Generate accuracy report
        unique_labels = set(y_true_mapped)  # Get unique labels

        for label in unique_labels:
            label_indices: list[int] = [
                i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label
            ]
            label_y_true: list[int] = [y_true_mapped[i] for i in label_indices]
            label_y_pred: list[int] = [y_pred_mapped[i] for i in label_indices]
            label_accuracy = accuracy_score(label_y_true, label_y_pred)
            print(f"Accuracy for label {labels[label]}: {label_accuracy:.3f}")

        # Generate classification report
        class_report = classification_report(
            y_true=y_true_mapped,
            y_pred=y_pred_mapped,
            target_names=labels,
            labels=list(range(len(labels))),
        )

        print("\nClassification Report:")
        print(class_report)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(
            y_true=y_true_mapped, y_pred=y_pred_mapped, labels=list(range(len(labels)))
        )
        print("\nConfusion Matrix:")
        print(conf_matrix)
        # WARN: headless server doens't display plot
        # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
        # disp.plot()
        # plt.show()

    # evaluate(y_true, y_pred)

    def HF_SFTConfig(self) -> SFTConfig:
        training_arguments = SFTConfig(
            output_dir=self.checkpoint_dir,  # directory to save and repository id
            run_name=self.base_model,
            num_train_epochs=1,  # number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass, virtually enlarges minibatch
            gradient_checkpointing=True,  # use gradient checkpointing to save memory
            optim="paged_adamw_32bit",  # using paged optimizer to memory efficiency
            logging_steps=1,
            learning_rate=2e-4,  # learning rate, based on QLoRA paper
            weight_decay=0.001,
            fp16=False,  # autocasting to float 16
            bf16=True,  # autocasting to brain float 16
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            max_steps=-1,
            warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
            group_by_length=False,
            lr_scheduler_type="cosine",  # use cosine learning rate scheduler
            report_to="wandb",  # report metrics to w&b
            eval_strategy="steps",  # save checkpoint every epoch
            eval_steps=0.2,
            packing=False,  # dont collapse small samples into one
            dataset_text_field="text",  # column header where to find input prompt
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            # max_seq_length=512,
        )

        return training_arguments

    def HF_SFTTrainer(self) -> SFTTrainer:
        trainer: SFTTrainer = SFTTrainer(
            model=self.model,
            args=self.HF_SFTConfig(),
            train_dataset=self.hf_train_data,
            eval_dataset=self.hf_eval_data,
            peft_config=self.LoRA_config(),
            processing_class=self.tokenizer,
        )

        return trainer

    def TRAINER_run_training(self) -> None:
        logger.info(msg="Fine-tune started")
        trainer: SFTTrainer = self.HF_SFTTrainer()
        trainer.train()
        logger.info(msg="Fine-tune finished")

        # close run on w&b portal
        logger.info(msg="Closing WandB portal")
        wandb.finish()
        # enable caching
        self.model.config.use_cache = True
        # Save trained model and tokenizer
        logger.info(msg="Saving fine-tuned models")
        # trainer.model.save_pretrained(trainer_filepath)
        self.model.save_pretrained(
            save_directory=os.path.join(self.trainer_dir, "model")
        )
        self.tokenizer.save_pretrained(
            save_directory=os.path.join(self.trainer_dir, "tokenzier")
        )

    def MODEL_load_ft_model(self):
        # reload tokenizer and model
        self.ft_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(self.trainer_dir, "tokenzier")
        )
        self.ft_tokenizer.pad_token_id = self.ft_tokenizer.eos_token_id
        self.ft_tokenizer.padding_side = "right"

        # load base model
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(self.trainer_dir, "model")
        )
        # merge adapters
        self.merged_model = peft_model.merge_and_unload()

        # method 2
        # # reload base model
        # base_model = AutoModelForCausalLM.from_pretrained(model_name)
        #
        # # merge base model and fine-tuned model
        # merged_model = PeftModel.from_pretrained(base_model, trainer_filepath)
        # merged_model = merged_model.merge_and_unload()
        #
        # # save merged model
        # merged_model.save_pretrained(merged_model_path)

        # method 3
        # Load the base model
        # base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
        # # Load the LoRA adapter
        # lora_adapter = PeftModel.from_pretrained(base_model, "lora_adapter_path")
        # # Merge the LoRA weights into the base model
        # merged_model = lora_adapter.merge_and_unload()

    def PEFT_MODEL_infer(self, input_prompts: str):
        self.merged_model.eval()
        pipe = pipeline(
            "text-generation",
            model=self.merged_model,
            tokenizer=self.ft_tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        outputs = pipe(
            inputs=input_prompts, max_new_tokens=2, do_sample=True, temperature=0.1
        )

        return outputs

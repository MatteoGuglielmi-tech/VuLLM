from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template

import logging
import torch

from typing import Any
from transformers import BitsAndBytesConfig 

from .base import ReasoningGenerator
from ..loader_config import Loader



logger = logging.getLogger(name=__name__)


class LlamaCoTGenerator(ReasoningGenerator):
    """A class to generate Chain-of-thoughts for a C code snippet."""

    def __init__(
        self,
        model_name: str,
        chat_template: str,
        max_seq_length: int,
        load_in_4bit: bool = True,
    ):
        """Initializes the model, tokenizer, and text-generation pipeline.

        Parameters
        ----------
        model_name: str
            The name of the model to load from Hugging Face.
        chat_template: str
            Chat template to apply when building chat-like structure.
        max_seq_length: int
            Length after which truncation is applied.
        load_in_4bit: bool, default True
            Load model in 4-bit quantization.
        """

        self.max_seq_length = max_seq_length

        with Loader(
            desc_msg=f"Loading model: {model_name}...",
            end_msg="✅ Model and tokenizer loaded successfully.",
            logger=logger,
        ):
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    # load_in_4bit=load_in_4bit,
                    quantization_config=quantization_config,
                    dtype=(torch.bfloat16 if is_bfloat16_supported() else None),
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )

                # Enable native 2x faster inference
                FastLanguageModel.for_inference(model=self.model)
                logger.info("⚡ Unsloth inference mode enabled.")

                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                try:
                    self.tokenizer = get_chat_template(self.tokenizer, chat_template=chat_template)
                except ValueError:
                    raise ValueError(f"available chat templates are: {print(list(CHAT_TEMPLATES.keys()))}")
                except Exception as e:
                    raise Exception(f"Exception occured during tokenizer initialization:\n{e}")

            except torch.cuda.OutOfMemoryError as oom:
                logger.critical("⚠️ OOM Error: Model does not fit entirely on GPU.")
                torch.cuda.empty_cache()
                raise oom

            except Exception as e:
                logger.critical(f"❌ Failed to load model: {e}")
                raise e

    def build_cot_prompt(
        self,
        c_code: str,
        is_vulnerable: bool,
        cwe_ids: list[str] | None = None,
        cwe_descs: list[str] | None = None,
    ) -> dict[str, str]:
        """
        Build the structured prompt for CWE analysis.

        Parameters
        ----------
        c_code : str
            C code to analyze
        is_vulnerable : bool
            Whether the code is known to be vulnerable
        cwe_ids : list[str] | None
            List of CWE identifiers

        Returns
        -------
        dict[str, str]
            Dictionary with 'system' and 'user' prompt content
        """
        messages = self.prompt_template.build_messages(
            func_code=c_code,
            is_vulnerable=is_vulnerable,
            cwe_ids=cwe_ids,
            cwe_descs=cwe_descs,
        )

        return {
            "system": messages[0]["content"],
            "user": messages[1]["content"],
        }

    def generate_reasoning(
        self, mini_batch: list[dict[str, Any]], max_completion_tokens: int
    ) -> list[str]:
        if not mini_batch:
            logger.warning("Empty batch detected")
            return []

        # reasonings: list[str] = []

        # for i in tqdm(range(0, len(mini_batch), batch_size), desc="Generating reasonings"):
            # mini_batch = mini_batch[i: i+batch_size]

        messages_batch = []
        for sample in mini_batch:
            prompt_dict: dict[str, str] = self.build_cot_prompt(
                c_code=sample["func"],
                is_vulnerable=bool(sample["target"]),
                cwe_ids=sample["cwe"],
                cwe_descs=sample["cwe_desc"],
            )

            messages_batch.append([
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]},
                {"role": "assistant", "content": ""},
            ])

        input_text = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            max_length=self.max_seq_length,
            truncation=True,
        ).to(self.model.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_completion_tokens,
                do_sample=True,
                use_cache=True,
                temperature=0.2,
                top_p=0.95,
                min_p=0.1,
            )

            results = self.tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            # reasonings.extend([res.strip() for res in results])
            return [res.strip() for res in results]

        except Exception as e:
            raise Exception(f"Llama: an error occurred during batch generation: {e}")

        # logger.info("✅ Batch generation completed successfully.")
        # return reasonings


# --- Test Case Data ---
sample_entries = [
    {
        "func": "void allocate_buffer(char *input) {\n    char buffer[50];\n    strcpy(buffer, input);\n}",
        "target": 1,
        "cwe": ["CWE-121"],
        "cwe_desc": [
            "Stack-based Buffer Overflow: The program copies a string from an external source into a fixed-size buffer on the stack without checking its length. This can lead to a buffer overflow, allowing an attacker to execute arbitrary code."
        ],
    },
    {
        "func": "int add_ints(int a, int b) {\n    return a + b;\n}",
        "target": 0,
        "cwe": [],
        "cwe_desc": [],
    },
]

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        generator = LlamaCoTGenerator(
            model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
            chat_template="llama-3.1",
            max_seq_length=4096,
            load_in_4bit=True
        )
        reasoning_results = generator.generate_reasoning(mini_batch=sample_entries, max_completion_tokens=512)

        print("\n" + "=" * 20 + " TEST RESULTS " + "=" * 20)
        for entry, result in zip(sample_entries, reasoning_results):
            print(f"\n--- Analyzing Function ---\n{entry['func']}\n")
            print(f"--- Generated Reasoning ---\n{result}\n")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Test script failed: {e}")

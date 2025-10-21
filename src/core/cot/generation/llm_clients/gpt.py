import logging

from typing import Any
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam
from src.core.cot.loader_config import Loader
from src.core.cot.generation.llm_clients.base import ReasoningGenerator

load_dotenv()

logger = logging.getLogger(name=__name__)


class AzureCoTGenerator(ReasoningGenerator):
    def __init__(self, deployment_name: str):
        """Initializes the Azure OpenAI client."""

        self.deployment_name = deployment_name

        try:
            with Loader(
                desc_msg=f"Loading model: GPT-4.1",
                end_msg="✅ Azure OpenAI client initialized and credentials validated.",
                logger=logger,
            ):
                # params in private file `.env`
                self.client = AzureOpenAI()
                # test call to verify credentials
                self.client.with_options(max_retries=1).models.list()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI client. Check your credentials and endpoint. Error: {e}")
            raise

    def generate_reasoning(self, mini_batch: list[dict[str, Any]], max_completion_tokens: int) -> list[str]:
        """Generates CoT reasoning for a batch of entries by making API calls to Azure OpenAI."""

        results = []
        if not mini_batch:
            logger.warning("Empty batch detected.")
            return results

        for sample in mini_batch:
            prompt_dict = self.build_cot_prompt(
                c_code=sample["func"],
                is_vulnerable=bool(sample["target"]),
                cwe_ids=sample["cwe"],
                cwe_descriptions=sample["cwe_desc"],
            )

            messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]},
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=0.1,
                    top_p=1.0,
                    # set penalties to 0 to avoid distorting the technical language
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                reasoning = response.choices[0].message.content
                results.append(reasoning.strip() if reasoning else "")
            except Exception as e:
                logger.error(f"An error occurred during an API call: {e}")
                results.append("")

        return results


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
    {
        "func": "void random_name(char *user_input, char *format_string) {\n char buffer[100];\n strcpy(buffer, user_input);\n printf(format_string);\n}",
        "target": 1,
        "cwe": ["CWE-121", "CWE-134"],
        "cwe_desc": [
            "Stack-based Buffer Overflow: The program copies a string from an external source into a fixed-size buffer on the stack without checking its length. This can lead to a buffer overflow, allowing an attacker to execute arbitrary code.",
            "The product uses a function that accepts a format string as an argument, but the format string originates from an external source.",
        ],
    },
]

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        generator = AzureCoTGenerator(
            deployment_name="gpt-4.1",
        )
        reasoning_results = generator.generate_reasoning(
            mini_batch=sample_entries, max_completion_tokens=2046
        )

        print("\n" + "=" * 20 + " TEST RESULTS " + "=" * 20)
        for entry, result in zip(sample_entries, reasoning_results):
            print(f"\n--- Analyzing Function ---\n{entry['func']}\n")
            print(f"--- Generated Reasoning ---\n{result}\n")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Test script failed: {e}")

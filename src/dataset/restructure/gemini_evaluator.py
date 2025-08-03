import os
import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.protos import Schema


@dataclass
class GeminiEvaluator:
    """A class to evaluate a dataset by sending questions to a Gemini model,
    and comparing the model's predictions with ground truth answers.

    This class streams the input dataset to handle very large files efficiently
    without loading the entire dataset into memory.
    """

    model_name: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        """Initializes the evaluator with the Gemini API key and model name."""

        self.gemini_model: GenerativeModel | None = None
        self._setup_()

    def _setup_(self) -> None:
        """Loads API keys and configures the generative model."""

        load_dotenv()
        API_KEY: str | None = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        if not API_KEY:
            raise ValueError("GOOGLE_AI_STUDIO_API_KEY or .env file not found.")

        genai.configure(api_key=API_KEY)  # type: ignore

        output_schema = Schema(
            type="OBJECT",
            properties={"classification": Schema(type="STRING", enum=["YES", "NO"])},
            required=[ "classification" ],
        )

        gemini_generation_config = genai.GenerationConfig(  # type: ignore
            candidate_count=1,  # force one response
            temperature=0.2,  # I don't want the model to be creative
            response_mime_type="application/json",  # candidates as JSON
            response_schema=output_schema,
        )

        try:
            self.gemini_model = genai.GenerativeModel(   # type: ignore
                model_name=self.model_name, generation_config=gemini_generation_config
            )
            print(f"Successfully initialized model: {self.model_name} in JSON mode for YES/NO classification.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def _make_api_call(self, prompt: str, max_retries: int = 5, initial_delay: int = 2) -> str:
        """Makes a call to the Gemini API with exponential backoff for retries.

        Args:
            prompt: str
                Prompt (query/question) to submit to the model
            max_retries: int
                The maximum number of times to retry the API call.
            initial_delay: int
                The initial delay in seconds for the first retry.
        Returns:
            str: The model's generated text, or an error message if all retries fail.
        """

        assert self.gemini_model is not None, "Model loading failed!"

        delay = initial_delay
        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(prompt)
                if response.parts:
                    # The model's response is a JSON string, so we parse it.
                    response_json = json.loads(response.text)
                    return response_json.get("classification", "Error: 'classification' key not found in response.")
                else:
                    return ("Error: No content generated. The prompt may have been blocked.")

            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}/{max_retries}. Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print("All retry attempts failed.")
                    return f"Error: Failed to get a response after {max_retries} attempts."

        return "Error: Max retries reached without a successful API call."

    def evaluate_dataset(self, input_filepath: str, output_filepath: str):
        """Loads a dataset from a .jsonl file, queries the model for each entry,
        and saves the question, ground truth, and prediction to a new .jsonl file.

        Args:
            input_filepath: str
                The path to the input .jsonl dataset.
                Each line should be a JSON object with 'question' and 'ground_truth' keys.
            output_filepath: str
                The path where the output .jsonl file will be saved.
        """

        print(f"Starting evaluation of dataset: {input_filepath}")

        if not os.path.exists(input_filepath):
            print(f"Error: Input file not found at {input_filepath}")
            return

        with open(input_filepath, 'r', encoding='utf-8') as f:
            total_lines: int = sum(1 for _ in f)
        print(f"Found {total_lines} entries to process.")

        try:
            with open(file=input_filepath, mode="r", encoding="utf-8") as infile, \
                open(file=output_filepath, mode="w", encoding="utf-8") as outfile:

                for i, line in enumerate(infile):
                    try:
                        data: dict[str,str] = json.loads(line.strip())
                        question_prompt: str|None = data.get("question")
                        ground_truth: str|None = data.get("ground_truth")

                        if not question_prompt or not ground_truth:
                            print(f"Skipping line {i+1}: missing 'question' or 'ground_truth'.")
                            continue

                        # get the model's prediction
                        print(f"Processing entry {i+1}/{total_lines}...")
                        prediction: str = self._make_api_call(prompt=question_prompt)

                        # prepare the output data
                        output_entry: dict[str,str] = {
                            "question": question_prompt,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                        }

                        # write the result to the output file immediately
                        outfile.write(json.dumps(output_entry) + "\n")

                    except json.JSONDecodeError:
                        print(f"Skipping line {i+1}: Invalid JSON format.")
                    except Exception as e:
                        print(f"An unexpected error occurred at line {i+1}: {e}")

        except IOError as e:
            print(f"Error handling files: {e}")
            return

        print(f"\nEvaluation complete. Results saved to: {output_filepath}")


def create_dummy_dataset(filepath: str, num_entries: int):
    """Creates a dummy .jsonl file for testing purposes.
    Parameters:
        filepath: str
            target location for the resulting dataset
        num_entries: int
            number of entries to replicate
    """

    print(f"Creating dummy dataset at: {filepath}")
 
    # This function now creates prompts that look more like your use case.
    def create_prompt(code_fragment: str, signature: str):
        return (
            "You are an AI system that analyzes C code for vulnerabilities.\n\n"
            "**TASK**: Given the following code fragment, determine whether it contains a security vulnerability.\n"
            "KEY: Code is chunked; reassemble by function signature to obtain full original source code.\n"
            "Note: input chunk may not be a valid C code. This is intended, the merge of them (removing the overlap due to contex) is valid.\n"
            f"Function signature:\n{signature}\n\n"
            f"Code fragment:\n{code_fragment}\n\n"
            "Answer with 'YES' if vulnerable, 'NO' otherwise.\n"
            "**IMPORTANT**: Strictly respond with only 'YES' or 'NO'.\n\n"
        ).strip()

    dummy_data = [
        {
            "question": create_prompt("char buf[10]; strcpy(buf, input);", "main(char* input)"),
            "ground_truth": "YES"
        },
        {
            "question": create_prompt("printf(\"Hello, World!\");", "main()"),
            "ground_truth": "NO"
        },
        {
            "question": create_prompt("sprintf(cmd, \"/bin/sh -c %s\", user_input); system(cmd);", "handle_request(char* user_input)"),
            "ground_truth": "YES"
        }
    ]
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(num_entries):
            # Cycle through the dummy data
            entry = dummy_data[i % len(dummy_data)]
            f.write(json.dumps(entry) + '\n')
    print("Dummy dataset created.")

def _display_file_content(file: str):
    with open(file=file, mode='r', encoding='utf-8') as f:
        for line in f:
            print(line.strip())

if __name__ == '__main__':
    INPUT_FILE = "my_dataset.jsonl"
    OUTPUT_FILE = "predictions.jsonl"

    # --- Main Execution ---
    # 1. Create a dummy dataset for demonstration
    create_dummy_dataset(filepath=INPUT_FILE, num_entries=5)

    # 2. Initialize the evaluator class
    evaluator = GeminiEvaluator()
    # 3. Run the evaluation process
    evaluator.evaluate_dataset(input_filepath=INPUT_FILE, output_filepath=OUTPUT_FILE)


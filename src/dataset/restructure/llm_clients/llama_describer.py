import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline # type: ignore
from transformers.utils.quantization_config import BitsAndBytesConfig

from dataset.restructure.shared.log import logger
from .base import DescriptionGenerator


class LlamaCodeDescriber(DescriptionGenerator):
    """
    A class to generate natural language descriptions of C code snippets
    using a local, pre-trained Llama model from Hugging Face.
    """

    def __init__(self, model_name: str = "unsloth/llama-3.1-8b-instruct-bnb-4bit"):
        """
        Initializes the model, tokenizer, and text-generation pipeline.

        Params:
            model_name: str
                The name of the model to load from Hugging Face.
                Defaults to a 4-bit quantized Llama 3.1 8B model for memory efficiency.
        """

        print(f"Loading model: {model_name}...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.desc_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

        print("✅ Model loaded successfully.")

    def _clean_response(self, prompt: str, generated_text: str) -> str:
        assistant_token: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_token in generated_text:
            response: str = generated_text.split(assistant_token)[-1].strip() # get generated text
            response = os.linesep.join([line for line in response.splitlines() if line]) # remove emtpy lines
            return response
        return generated_text[len(prompt):].strip()

    def _create_func_prompt(self, c_code: str) -> list[dict[str,str]]:
        blocks: dict[str,str] = self._build_c_function_prompt(c_code=c_code)
        return [{"role": "system", "content": blocks["system"]}, {"role": "user", "content": blocks["user"]}]

    def _create_cwe_prompt(self, cwe_id: str) -> list[dict[str,str]]:
        blocks: dict[str,str] = self._build_cwe_prompt(cwe_id=cwe_id)
        return [{"role": "system", "content": blocks["system"]}, {"role": "user", "content": blocks["user"]}]

    def generate_batch_descriptions(self, c_code_batch: list[str]) -> list[str]:
        if not c_code_batch: return []

        prompts = [
            self.tokenizer.apply_chat_template(
                self._create_func_prompt(c_code=code),
                tokenize=False,
                add_generation_prompt=True,
            )
            for code in c_code_batch
        ]

        try:
            batch_outputs = self.desc_pipeline(
                text_inputs=prompts,
                max_new_tokens=200,  # Max length of the generated description
                temperature=0.2,     # Lower temperature for more factual, less creative output
                do_sample=True,      # Required for temperature to have an effect
                batch_size=8,
            )
            # -- parse each output in the batch to extract the clean description --
            return [self._clean_response(prompts[i], out[0]['generated_text']) for i, out in enumerate(batch_outputs)]

        except Exception as e:
            print(f"Llama: an error occurred during batch generation: {e}")
            return ["N/A"] * len(c_code_batch)

    def generate_batch_cwe_descriptions(self, cwe_ids_batch: list[str]) -> list[str]:
        if not cwe_ids_batch: return []

        # 1. Identify valid CWE IDs and their original positions
        valid_requests = {
            index: cwe_id for index, cwe_id in enumerate(cwe_ids_batch)
            if cwe_id and ((isinstance(cwe_id, str) and cwe_id.strip()) or (isinstance(cwe_id, list) and cwe_id != []))
        }
        if not valid_requests:
            return ["N/A"] * len(cwe_ids_batch)

        prompts = [
            self.tokenizer.apply_chat_template(self._create_cwe_prompt(cwe_id=cid), tokenize=False, add_generation_prompt=True)
            for cid in valid_requests.values()
        ]
        try:
            batch_outputs = self.desc_pipeline(prompts, max_new_tokens=100, temperature=0.2, do_sample=True, batch_size=8)
            raw_responses = [self._clean_response(prompts[i], out[0]['generated_text']) for i, out in enumerate(batch_outputs)]
            cleaned_responses = [self._remove_leading_cwe(description=resp).strip() for resp in raw_responses]
        except Exception as e:
            logger.error(f"Llama CWE batch generation failed: {e}")
            cleaned_responses = ["N/A"] * len(cwe_ids_batch)

        final_descriptions = ["N/A"] * len(cwe_ids_batch)
        response_iter = iter(cleaned_responses)
        for index in valid_requests.keys():
            final_descriptions[index] = next(response_iter)

        return final_descriptions


# --- Example Usage ---
if __name__ == "__main__":

    sample_c_function = """
    int _gnutls_ciphertext2compressed(gnutls_session_t session, opaque * compress_data, int compress_size, gnutls_datum_t ciphertext, uint8 type) {
        uint8 MAC[MAX_HASH_SIZE];
        uint16 c_length;
        int length;
        mac_hd_t td;
        int ret;
        if (memcmp(MAC, &ciphertext.data[length], hash_size) != 0) {
            gnutls_assert();
            return GNUTLS_E_DECRYPTION_FAILED;
        }
        if (compress_size < length) {
            gnutls_assert();
            return GNUTLS_E_INTERNAL_ERROR;
        }
        memcpy(compress_data, ciphertext.data, length);
        return length;
    }
    """

    describer = LlamaCodeDescriber()

    cwe_batch_to_test = [
        "CWE-79",   # Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')
        "CWE-89",   # Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')
        "CWE-125",  # Out-of-bounds Read
        "CWE-416",  # Use After Free
        "CWE-22",   # Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')
        "CWE-306",  # Missing Authentication for Critical Function
        "CWE-787",  # Out-of-bounds Write
        "CWE-502",  # Deserialization of Untrusted Data
        "CWE-20",   # Improper Input Validation
        "CWE-918",  # Server-Side Request Forgery (SSRF)
    ]

    # Generate the description
    print("\n--- Generating CWE Description ---")
    descriptions = describer.generate_batch_cwe_descriptions(cwe_ids_batch=cwe_batch_to_test)

    # Print the result
    print("\n--- Generated CWE Description ---")
    for d, c in zip(descriptions, cwe_batch_to_test):
        print(f"\n--- Generated Description for {c} ---")
        print(f"{d}\n")


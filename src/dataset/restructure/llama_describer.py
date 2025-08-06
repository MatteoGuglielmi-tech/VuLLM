import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig


class LlamaCodeDescriber:
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

        self.description_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=150,       # Max length of the generated description
            temperature=0.2,          # Lower temperature for more factual, less creative output
            do_sample=True,           # Required for temperature to have an effect
        )
        print("Model loaded successfully.")

    def generate_description(self, c_code: str) -> str:
        """Generates a description for a given C function string.

        Params:
            c_code: str
                A string containing the C function to be described.

        Returns:
            str: The generated natural language description.
        """

        if not self.model or not self.tokenizer:
            raise RuntimeError("Llama model initialization failed.")
        if not c_code:
            return "Error: Input code is empty."

        role_content:str = (
            "You are an expert C code analyst and programmer."
            "Your task is to provide a concise, high-level functional description for the provided C function.\n"
            "KEY: The description must be in plain English prose."
        )

        user_content = (
            "Please generate a description for the following C function.\n\n"
            "**Instructions**:\n"
            "- The description must be a single, complete paragraph.\n"
            "- Focus only on the function's primary goal.\n"
            "- **DO NOT** mention specific variable names, parameters, or return types.\n"
            "- **DO NOT** use markdown, non-ASCII characters or any special formatting in your response.\n\n"
            "--- CODE START ---\n"
            f"{c_code}\n"
            "--- CODE END ---\n\n"
            "**Description:**"
        ).strip()

        messages: list[dict[str, str]] = [
            { "role": "system", "content": role_content },
            { "role": "user", "content": user_content },
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        try:
            outputs = self.description_pipeline(prompt)
            generated_text:str = outputs[0]['generated_text']

            # parse output to extract assistant's response.
            assistant_token:str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if assistant_token in generated_text:
                response:str = generated_text.split(assistant_token)[-1]
                return response.strip()
            else:
                return generated_text[len(prompt):].strip()

        except Exception as e:
            print(f"An error occurred during generation: {e}")
            return "N/A"


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

    # Generate the description
    print("\n--- Generating Description ---")
    description = describer.generate_description(sample_c_function)

    # Print the result
    print("\n--- Generated Description ---")
    print(description)



import os
import time
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types.generation_types import GenerationConfig

from dataset.restructure.shared.log import logger
from .base import DescriptionGenerator


@dataclass
class GeminiClient(DescriptionGenerator):
    """A client for interacting with the Google Gemini API.

    This class handles the setup, configuration, and API calls to the
    Google Gemini model for:
    - generating code descriptions.

    Attributes
    ----------
    model_name : str, optional
        The name of the Gemini model to use, by default "gemini-1.5-flash".
    gemini_model : GenerativeModel, optional
        The initialized generative model instance from the Google AI SDK.
    gemini_generation_config : GenerationConfig, optional
        The configuration object for controlling generation parameters.
    """

    model_name: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        """Initializes and sets up the Gemini client after dataclass construction."""

        self.gemini_model: GenerativeModel|None = None
        self.gemini_generation_config: GenerationConfig|None = None
        self._setup_()

    def _setup_(self) -> None:
        """Loads API keys and configures the generative model.

        Loads the Google AI Studio API key from environment variables (or a .env file)
        and configures the `genai` library. It then initializes the `GenerativeModel`
        and sets the default `GenerationConfig`.

        Raises
        ------
        ValueError
            If the 'GOOGLE_AI_STUDIO_API_KEY' is not found in the environment.
        """

        load_dotenv()
        API_KEY: str|None = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        if not API_KEY:
            raise ValueError("GOOGLE_AI_STUDIO_API_KEY not found in environment or .env file.")

        genai.configure(api_key=API_KEY)  # type: ignore
        self.gemini_model = GenerativeModel(model_name=self.model_name)
        self.gemini_base_config: dict[str, Any] = {
            "candidate_count": 1,                       # force one response
            "stop_sequences": None,                     # no stop sequences
            "temperature": 0.2,                         # I don't want the model to be creative
            "response_mime_type": "application/json",   # candidates as JSON
            "response_schema": str,                     # in JSON I want strings
        }

    def _create_func_prompt(self, c_code: str) -> str:
        """Creates the full prompt for a single code snippet.

        Parameters
        ----------
        c_code : str
            The C/C++ code snippet to be described.

        Returns
        -------
        str
            A fully formatted prompt string ready to be sent to the API.
        """

        blocks: dict[str,str] = self._build_c_function_prompt(c_code=c_code)
        prompt: str = f"{blocks["system"]}\n{blocks["user"]}"

        return prompt

    def _create_cwe_prompt(self, cwe_id: str) -> str:

        blocks: dict[str,str] = self._build_cwe_prompt(cwe_id=cwe_id)
        prompt: str = f"{blocks["system"]}\n{blocks["user"]}"

        return prompt

    def generate_content(self, prompt: str, max_tokens: int) -> str:

        if not self.gemini_model:
            raise RuntimeError("Gemini model is not initialized. Call _setup_ first.")

        try:
            gemini_gen_config = GenerationConfig(**self.gemini_base_config, max_output_tokens=max_tokens)
            response = self.gemini_model.generate_content(contents=prompt, generation_config=gemini_gen_config)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed for a snippet: {e}")
            return "N/A"

    def generate_batch_descriptions(self, c_code_batch: list[str]) -> list[str]:

        if not c_code_batch: return []

        descriptions: list[str] = []
        for c_code in c_code_batch:
            if c_code.strip():
                prompt:str = self._create_func_prompt(c_code=c_code)
                descriptions.append(self.generate_content(prompt=prompt, max_tokens=150))
                time.sleep(1) # delay to avoid requests-per-minute rate limits error

        return descriptions

    def generate_batch_cwe_descriptions(self, cwe_ids_batch: list[str]) -> list[str]:

        if not cwe_ids_batch: return []
        descriptions: list[str] = []
        for cwe_id in cwe_ids_batch:
            if cwe_id and ((isinstance(cwe_id, str) and cwe_id.strip()) or (isinstance(cwe_id, list) and cwe_id != [])):
                prompt:str = self._create_cwe_prompt(cwe_id=cwe_id)
                response:str = self.generate_content(prompt=prompt, max_tokens=100)
                response = self._remove_leading_cwe(description=response)
                descriptions.append(response)
                time.sleep(1) # delay to avoid requests-per-minute rate limits error
            else:
                descriptions.append("N/A")

        return descriptions


import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

import google.generativeai as genai
from google.generativeai.generative_models import GenerativeModel
from google.generativeai.types.generation_types import GenerationConfig


@dataclass
class Gemini:
    """A client for interacting with the Google Gemini API."""

    model_name: str = "gemini-2.5-flash"

    def __post_init__(self) -> None:
        """Initializes attributes."""

        self.gemini_model: Optional[GenerativeModel] = None
        self.gemini_generation_config: Optional[GenerationConfig] = None
        self._setup_()

    def _setup_(self) -> None:
        """Loads API keys and configures the generative model."""

        load_dotenv()
        API_KEY: str | None = os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        if not API_KEY:
            raise ValueError(
                "GOOGLE_AI_STUDIO_API_KEY not found in environment or .env file."
            )

        genai.configure(api_key=API_KEY)  # type: ignore
        self.gemini_model = genai.GenerativeModel(model_name=self.model_name)  # type: ignore
        self.gemini_generation_config = genai.GenerationConfig(  # type: ignore
            candidate_count=1,  # force one response
            stop_sequences=None,  # no stop sequences
            max_output_tokens=400,  # ~ 100 words
            temperature=0.2,  # I don't want the model to be creative
            response_mime_type="application/json",  # candidates as JSON
            response_schema=str,  # in JSON I want strings
        )

    def generate_description(self, func_str: str) -> str:
        """Generates a functional description for a given C/C++ code snippet."""

        if not self.gemini_model or not self.gemini_generation_config:
            raise RuntimeError("Gemini model is not initialized. Call _setup_ first.")

        prompt_skeleton = (
            "You are an expert C/C++ code analyst. Your task is to provide a concise, "
            "high-level functional description for the provided code snippet.\n\n"
            "**Instructions**:\n"
            "- Describe the function's primary purpose in a single paragraph.\n"
            "- The description must be in plain English prose.\n"
            "- Do NOT mention function parameters, return types, or implementation details.\n"
            "- Do NOT use markdown, code formatting, or non-ASCII characters.\n\n"
            "**Code**:\n"
            "{code_snippet}\n\n"
            "**Description**:\n"
        ).strip()

        full_prompt: str = prompt_skeleton.format(code_snippet=func_str)

        synth_desc: str = self.gemini_model.generate_content(
            contents=full_prompt,
            generation_config=self.gemini_generation_config,
        ).text

        return synth_desc

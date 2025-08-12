from abc import ABC, abstractmethod

class PromptingStrategy(ABC):
    """An abstract base class for different prompt formatting strategies.

    This class defines the interface for all prompting strategies. Subclasses
    must implement the `format` method, which takes system and user prompts
    and returns a single, model-ready string.
    """

    @abstractmethod
    def format(self, system_prompt: str, user_prompt: str) -> str:
        """Formats the system and user prompts into a final string.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction or context for the model.
        user_prompt : str
            The user-provided query or instruction.

        Returns
        -------
        str
            The fully formatted prompt string ready for model consumption.
        """
        pass

class GenericStrategy(PromptingStrategy):
    """A generic strategy that simply combines the system and user prompts.

    This strategy concatenates the system and user prompts, separated by a
    newline character. It serves as a basic, model-agnostic formatting
    approach.
    """

    def format(self, system_prompt: str, user_prompt: str) -> str:
        """Combines system and user prompts with a newline separator."""

        return f"{system_prompt}\n{user_prompt}"

class Llama3Strategy(PromptingStrategy):
    """A strategy that formats prompts for Llama 3 instruct models.

    This strategy constructs a prompt using the specific special tokens and
    structure required by Llama 3 instruct models. It parses the `user_prompt`
    to separate the user's query from the expected answer, using
    "Correct answer:" as a delimiter.
    """
    def format(self, system_prompt: str, user_prompt: str) -> str:
        """Formats prompts using the Llama 3 instruct model structure."""

        parts:list[str] = user_prompt.split("Correct answer:")
        user_content:str = parts[0].strip()
        assistant_content:str = parts[1].strip() if len(parts) > 1 else ""

        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_content}<|eot_id|>"
        )

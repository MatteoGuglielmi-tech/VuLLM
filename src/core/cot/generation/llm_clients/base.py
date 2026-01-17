from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod

from ..prompts import PromptTemplate
from ..parsers import ParsedSample

T = TypeVar("T", bound=PromptTemplate)


class ReasoningGenerator(ABC, Generic[T]):
    """
    Abstract base class for generating Chain-of-Thought vulnerability analysis.

    Type Parameters
    ---------------
    T : PromptTemplate
        The specific prompt template type this generator uses
    """

    def __init__(self, prompt_template: T):
        """
        Initialize with a prompt template.

        Parameters
        ----------
        prompt_template : T
            The prompt template instance to use
        """
        self.prompt_template = prompt_template

    @abstractmethod
    def generate_reasoning(
        self, mini_batch: list[dict[str, Any]], max_completion_tokens: int
    ) -> list[ParsedSample]:
        """Generate reasoning for a batch of samples."""
        pass

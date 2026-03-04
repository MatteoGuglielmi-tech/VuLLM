import warnings
from typing import Literal
from transformers import PreTrainedTokenizer

from .base import BaseSequenceLengthAnalyzer
from .finetuningv2 import FineTunePromptAnalyzerV2
from .finetuningv3 import FineTunePromptAnalyzerV3
from .assessment import JudgePromptAnalyzer

from ..datatypes import AssumptionMode, PromptPhase

AnalyzerVersion = Literal["v1", "v2", "v3", "judge"]


class AnalyzerFactory:
    """Factory for creating analyzers with appropriate parameters."""

    @staticmethod
    def create(
        version: AnalyzerVersion,
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        # V2-specific parameters (optional, with defaults)
        assumption_mode: AssumptionMode | None = None,
        prompt_phase: PromptPhase | None = None,
        add_hierarchy: bool = False,
    ) -> BaseSequenceLengthAnalyzer:
        """Create analyzer based on version.

        Parameters
        ----------
        version : AnalyzerVersion
            Which analyzer to create ("v1", "v2", or "judge")
        tokenizer : PreTrainedTokenizer
            Tokenizer instance
        chat_template : str
            Chat template to use
        assumption_mode : str, optional
            For V2: assumption mode ("with_context" or "without_context")
        prompt_mode : str, optional
            For V2: prompt mode ("training" or "inference")
        add_hierarchy : bool
            For V2: whether to add CWE hierarchy
        include_reasoning : bool
            For judge: whether to request reasoning

        Returns
        -------
        BaseSequenceLengthAnalyzer
            Configured analyzer instance
        """

        match version:
            case "v2":
                # Validate V2-specific parameters
                if assumption_mode is None or prompt_phase is None:
                    raise ValueError("V2 analyzer requires 'assumption_mode' and 'prompt_mode'")

                return FineTunePromptAnalyzerV2(
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    assumption_mode=assumption_mode,
                    prompt_phase=prompt_phase,
                    add_hierarchy=add_hierarchy,
                )
            case "v1"| "v3":

                if version == "v1":
                    warnings.warn(
                        "This prompt version is deprecated, use `v2` or `v3` instead. Defaulting to v3" ,
                        DeprecationWarning,
                        stacklevel=2,
                    )

                # Validate V3-specific parameters
                if assumption_mode is None or prompt_phase is None:
                    raise ValueError("V3 analyzer requires 'assumption_mode' and 'prompt_mode'")

                return FineTunePromptAnalyzerV3(
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                    assumption_mode=assumption_mode,
                    prompt_phase=prompt_phase,
                    add_hierarchy=add_hierarchy,
                )
            case "judge":
                return JudgePromptAnalyzer(
                    tokenizer=tokenizer,
                    chat_template=chat_template,
                )
            case _:
                raise ValueError(f"Unknown analyzer version: {version}")

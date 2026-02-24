from .base import BaseVulnPrompt
from .v1 import VulnPromptV1
from .v2 import VulnPromptV2

from ..datatypes import AssumptionMode, PromptPhase, PromptVersion


class VulnPromptFactory:
    """Factory for creating analyzers with appropriate parameters."""

    @staticmethod
    def create(
        version: PromptVersion,
        prompt_phase: PromptPhase,
        assumptions_mode: AssumptionMode,
        add_cwe_guidelines: bool,
        debug_mode: bool = False,
        # specific parameters
        # ...
    ) -> BaseVulnPrompt:
        """Create analyzer based on version.

        Parameters
        ----------
        version : PromptVersion
            Which prompt to use ("v1", "v2")
        assumption_mode : str
            Assumptions to use
        prompt_mode : str
            Fine-tuning stage, influences prompt and chat template structures
        add_cwe_guidelines: str, optional (default=False)
            Add CWE hierarchy guidelines in system prompt

        Returns
        -------
        BaseVulnPrompt
            Configured model prompt instance
        """

        match version:
            case "v1":
                return VulnPromptV1(
                    phase=prompt_phase,
                    assumptions_mode=assumptions_mode,
                    add_cwe_guidelines=add_cwe_guidelines,
                    debug_mode=debug_mode,
                )
            case "v2":
                return VulnPromptV2(
                    phase=prompt_phase,
                    assumptions_mode=assumptions_mode,
                    add_cwe_guidelines=add_cwe_guidelines,
                    debug_mode=debug_mode,
                )
            case _:
                raise ValueError(f"Unknown prompt version: {version}")

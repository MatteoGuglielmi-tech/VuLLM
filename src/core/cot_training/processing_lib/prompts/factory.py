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


if __name__ == "__main__":
    from ...utilities import rich_rule
    from .base import BaseVulnPrompt
    from pathlib import Path
    import argparse

    parser = argparse.ArgumentParser(
        prog="CoT fine-tuning",
        description="Fine-tune, optimize, or run inference with CoT models",
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptPhase,
        choices=[m.value for m in PromptPhase],
        default="training",
        help="Defines prompt structure to use",
    )
    parser.add_argument(
        "--assumption_mode",
        type=AssumptionMode,
        choices=[m.value for m in AssumptionMode],
        default="none",
        help="Defines whether the model will be optimistic, pessimistic or neutral",
    )
    parser.add_argument(
        "--add_hierarchy",
        action="store_true",
        help="Add cwe hierarchy guidelines in system prompt",
    )
    parser.add_argument(
        "--prompt_version",
        "-v",
        type=str,
        choices=["v1", "v2"],
        help="Prompt version to use.",
    )
    args = parser.parse_args()

    def _serialize_in_use_prompt(
        sys_prompt: str,
        user_prompt: str,
        phase: PromptPhase,
        assumptions_mode: AssumptionMode,
        add_cwe_guidelines: bool,
        ground_truth: str | None,
        dst: str = "prompt",
    ):

        parent_dir = Path(dst)
        parent_dir.mkdir(parents=True, exist_ok=True)
        filename: str = (
            f"{phase}_{assumptions_mode}{"_hierarchy" if add_cwe_guidelines else ""}.txt"
        )
        with open(file=parent_dir / filename, mode="w") as text_file:
            text_file.write(50 * "=" + " System prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(sys_prompt)
            text_file.write("\n\n")
            text_file.write(50 * "=" + " User prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(user_prompt)
            text_file.write("\n\n")
            if ground_truth is not None:
                text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
                text_file.write("\n")
                text_file.write(ground_truth)

    fact = VulnPromptFactory()
    parent_dir = f"text_prompts/{args.prompt_version}"

    rich_rule(title="TRAINING MODE PESSIMISTIC")
    config = fact.create(
        version=args.prompt_version,
        prompt_phase=args.prompt_mode,
        assumptions_mode=args.assumption_mode,
        add_cwe_guidelines=args.add_hierarchy
    )
    training_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
    )
    _serialize_in_use_prompt(
        phase=args.prompt_mode,
        assumptions_mode=args.assumption_mode,
        add_cwe_guidelines=args.add_hierarchy,
        sys_prompt=training_messages[0]["content"],
        user_prompt=training_messages[1]["content"],
        ground_truth=training_messages[2]["content"],
        dst=parent_dir
    )

    print(f"Mode:")
    print(f"phase={args.prompt_mode}")
    print(f"mode={args.assumption_mode}")
    print(f"add_cwe_guidelines={args.add_hierarchy}")
    print(f"Serialized to: {parent_dir}")
    rich_rule()

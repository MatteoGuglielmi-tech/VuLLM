from typing import Any
from abc import ABC, abstractmethod


BatchOutput = list[list[dict[str,str]]]


class ReasoningGenerator(ABC):
    """An abstract base class for generating Chain-of-Thought (CoT) vulnerability analysis reasoning for code."""

    def build_cot_prompt(
        self,
        c_code: str,
        is_vulnerable: bool,
        cwe_ids: list[str],
        cwe_descriptions: list[str],
    ) -> dict[str, str]:
        """Builds the structured prompt for CoT vulnerability analysis, handling multiple CWEs."""

        system_prompt = (
            "You are an expert cybersecurity analyst specializing in C/C++ static code analysis. "
            "Your task is to analyze the provided code for all specified vulnerabilities and produce a "
            "step-by-step reasoning chain that logically leads to the known outcomes."
        )

        conclusion_instruction = "Address all specified CWEs in your reasoning, then state the final answer."

        # Build the "Known Outcome" text based on the list of CWEs
        if is_vulnerable and cwe_ids:
            outcome_parts = ["This function is known to be VULNERABLE with the following weaknesses:"]
            for id, desc in zip(cwe_ids, cwe_descriptions):
                outcome_parts.append(f"- **{id}**: {desc}")
            outcome_text = "\n".join(outcome_parts)
            outcome_text += "\n\nEnsure your reasoning logically leads to this known outcome."
        elif is_vulnerable:
            outcome_text = "This function is known to be VULNERABLE but no weaknesses have been specified."
            conclusion_instruction = "Attempt to identify the potential weakness, then state the final answer."
        else:
            outcome_text = "This function is considered NOT VULNERABLE."

        output_format = "Produce a step-by-step list of your reasoning. Afterwards, your final answer must be a one-liner prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'."

        user_prompt = (
            "**Analysis Instructions:**\n"
            "1. **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
            "2. **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
            "3. **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
            f"4. **Conclude:** {conclusion_instruction}\n\n"
            "**Output Format:**\n"
            f"{output_format}\n"
            # "Explicitly mention: 'the function is VULNERABLE' or 'the function is NOT VULNERABLE'.\n"
            "--- CODE START ---\n"
            f"{c_code}\n"
            "--- CODE END ---\n\n"
            f"**Known Outcome:**\n{outcome_text}\n\n"
            "**Reasoning:**"
        ).strip()

        return { "system": system_prompt, "user": user_prompt }

    @abstractmethod
    def generate_reasoning(self, mini_batch: list[dict[str, Any]], max_completion_tokens: int) -> list[str]:
        """generates descriptions for a batch of c functions."""
        pass


from .factory import PromptTemplateFactory
from .base import PromptTemplate


# this doesn't follow the structure of other prompts since it's here only for backtracking
@PromptTemplateFactory.register("cwe_old_gen")
class CWEPromptTemplateOldGen(PromptTemplate):
    """
    Prompt template for generating Chain-of-Thought reasoning
    for vulnerability analysis fine-tuning data.
    """

    SYSTEM_PROMPT = (
        "You are an expert cybersecurity analyst specializing in C/C++ static code analysis. "
        "Your task is to analyze the provided code for all specified vulnerabilities and produce a "
        "step-by-step reasoning chain that logically leads to the known outcomes."
    )


    USER_PROMPT = (
        "**Analysis Instructions:**\n"
        "1. **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
        "2. **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
        "3. **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
        "4. **Conclude:** {conclusion_instruction}\n\n"
        "**Output Format:**\n"
        "{output_format}\n"
        # "Explicitly mention: 'the function is VULNERABLE' or 'the function is NOT VULNERABLE'.\n"
        "--- CODE START ---\n"
        "{c_code}\n"
        "--- CODE END ---\n\n"
        "**Known Outcome:**\n{outcome_text}\n\n"
        "**Reasoning:**"
    ).strip()


    def build_cot_prompt(
        self,
        c_code: str,
        is_vulnerable: bool,
        cwe_ids: list[str],
        cwe_descriptions: list[str],
    ) -> dict[str, str]:
        """Builds the structured prompt for CoT vulnerability analysis, handling multiple CWEs."""

        conclusion_instruction = "Address all specified CWEs in your reasoning, then state the final answer."

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

        self.user.format(
            conclusion_instruction=conclusion_instruction,
            output_format=output_format,
            c_code=c_code,
            outcome_text=outcome_text
        )

        return {"system": self.system, "user": self.user}

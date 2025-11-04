"""
Sequence Length Analyzer for fine-tuning.
Analyzes token distribution in CoT dataset to determine optimal max_seq_length.

Features:
- Memory-efficient streaming (handles large JSONL files)
- Multiple tokenizer support
- Comprehensive statistics and visualizations
- Truncation impact analysis
"""

from transformers import PreTrainedTokenizer

from .base import BaseSequenceLengthAnalyzer
from ..datatypes import ReasoningSample


class FineTunePromptAnalyzer(BaseSequenceLengthAnalyzer):
    """Analyzes token distribution in CoT dataset to determine optimal max_seq_length."""

    SYSTEM_PROMPT = (
        "You are an expert cybersecurity analyst specializing in C static code analysis. "
        "Your task is to analyze the provided code and produce a step-by-step reasoning "
        "chain explaining whether it contains a vulnerability."
    )

    PROMPT_SKELETON = (
        "**Analysis Instructions:**\n"
        "1. **Trace Data Flow:** Analyze the flow of any external or user-controlled input.\n"
        "2. **Pinpoint Dangerous Functions:** Identify the use of functions known to be risky (e.g., `strcpy`, `gets`, `sprintf`, `memcpy`) for each specified weakness.\n"
        "3. **Check for Safeguards:** Look for any bounds checking, sanitization, or defensive programming that might mitigate risks.\n"
        "4. **Conclude:** State your conclusion based on the analysis.\n\n"
        "**Output Format:**\n"
        "Produce a step-by-step list of your reasoning. After the list, your final answer must be "
        "prefixed with 'Final Answer:' and be in the format 'YES (CWE-XXX, ...)' or 'NO'.\n"
        "--- CODE START ---\n"
        "{func_code}\n"
        "--- CODE END ---\n\n"
        "**Reasoning:**\n"
    )

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)

    def format_sample(self, sample: ReasoningSample) -> tuple[str, str, str]:
        user_content = self.PROMPT_SKELETON.format(func_code=sample.func)
        assistant_content = f"{sample.reasoning}"

        return self.SYSTEM_PROMPT, user_content, assistant_content

    def count_tokens_for_sample(self, sample: ReasoningSample) -> int:
        system_content, user_content, assistant_content = self.format_sample(sample)

        full_convo = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted = self.apply_template(messages=full_convo)

        return self._encode_and_count(text=full_formatted)


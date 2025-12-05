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
from ..datatypes import ReasoningSample, TokensStats


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

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: str):
        super().__init__(tokenizer, chat_template)

    def format_sample(self, sample: ReasoningSample) -> tuple[str, str, str]:
        user_content = self.PROMPT_SKELETON.format(func_code=sample.func)
        assistant_content = f"{sample.reasoning}"

        return self.SYSTEM_PROMPT, user_content, assistant_content

    def count_tokens_for_sample(self, sample: ReasoningSample) -> TokensStats:
        system_content, user_content, assistant_content = self.format_sample(sample)

        # Count individual components
        # System
        system_messages = [{"role": "system", "content": system_content}]
        system_formatted = self.tokenizer.apply_chat_template(
            system_messages, tokenize=False, add_generation_prompt=False
        )
        system_tokens = len(self.tokenizer.encode(system_formatted, add_special_tokens=False))  # type: ignore

        # System + User (to get user contribution)
        system_user_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        system_user_formatted = self.tokenizer.apply_chat_template(
            system_user_messages, tokenize=False, add_generation_prompt=False
        )
        system_user_tokens = len(self.tokenizer.encode(system_user_formatted, add_special_tokens=True))  # type: ignore
        user_tokens = system_user_tokens - system_tokens

        # Full conversation (system + user + assistant)
        full_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        total_tokens = len(self.tokenizer.encode(full_formatted, add_special_tokens=True))  # type: ignore

        # Assistant tokens = total - (system + user)
        assistant_tokens = total_tokens - system_user_tokens

        # Split assistant into reasoning and answer
        if sample.target == 1 and sample.cwe:
            cwe_string = ", ".join(sample.cwe)
            final_answer_str = f"\n\nFinal Answer: YES ({cwe_string})"
        else:
            final_answer_str = "\n\nFinal Answer: NO"

        # approximation: can't easily separate comps post-tokenization
        answer_tokens_approx = self._encode_and_count(final_answer_str)
        reasoning_tokens_approx = assistant_tokens - answer_tokens_approx

        total_approx = reasoning_tokens_approx + answer_tokens_approx
        if total_approx > 0:
            answer_tokens = int(
                assistant_tokens * (answer_tokens_approx / total_approx)
            )
            reasoning_tokens = assistant_tokens - answer_tokens
        else:
            reasoning_tokens = assistant_tokens
            answer_tokens = 0

        return TokensStats(
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            assistant_tokens=assistant_tokens,
            total_tokens=total_tokens,
        )

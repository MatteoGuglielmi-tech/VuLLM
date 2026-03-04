"""
Sequence Length Analyzer for assessment task.
Analyses the token distribution when applying the prompt to evaluate CoT quality.
"""

from transformers import PreTrainedTokenizer

from .base import BaseSequenceLengthAnalyzer
from ..datatypes import Messages, ReasoningSample, TokensStats


class JudgePromptAnalyzer(BaseSequenceLengthAnalyzer):
    """Analyzes token distribution in CoT dataset to determine optimal max_seq_length."""

    SYSTEM_PROMPT: str = (
        "You are an expert code security analyst evaluating reasoning quality for vulnerability detection."
        "Task: Rate the quality of the reasoning provided for identifying vulnerabilities in the given C code."
    )

    PROMPT_SKELETON: str = (
        "**Metadata information:**\n"
        "Source project: {project}\n"
        "Ground truth label: {target} (0=safe, 1=vulnerable)\n\n"

        "Relevant CWEs:\n{cwe_info}\n\n"

        "**C Function of reference:**\n"
        "--- CODE START ---\n"
        "```c\n{func}\n```\n\n"
        "--- CODE END ---\n\n"

        "**Reasoning to evaluate:**\n"
        "{reasoning}\n\n"

        "**Evaluation Criteria**:\n"
        "Evaluate the reasoning across the following dimensions (each scored 0-1):\n\n"
        "1. **Correctness** (0-1): Does the reasoning correctly identify vulnerabilities and match the ground truth label?\n"
        "   - 0.0-0.3: Incorrect conclusion or misses critical vulnerabilities\n"
        "   - 0.4-0.6: Partially correct but with significant errors (including vulnerbilities not explicitly present)\n"
        "   - 0.7-0.9: Mostly correct with minor issues\n"
        "   - 1.0: Fully correct identification and conclusion\n\n"
        "2. **Completeness** (0-1): Are all relevant security issues and CWEs covered?\n"
        "   - 0.0-0.3: Major vulnerabilities or CWEs missing\n"
        "   - 0.4-0.6: Some issues covered but incomplete or detected vulnerabilities that are not clearly present\n"
        "   - 0.7-0.9: Most issues covered with minor omissions\n"
        "   - 1.0: Comprehensive coverage of all relevant issues\n\n"
        "3. **Clarity** (0-1): Is the reasoning clear, well-structured, and easy to follow?\n"
        "   - 0.0-0.3: Confusing, poorly structured, hard to understand\n"
        "   - 0.4-0.6: Understandable but could be clearer\n"
        "   - 0.7-0.9: Clear and well-organized\n"
        "   - 1.0: Exceptionally clear and well-structured\n\n"
        "4. **Technical Accuracy** (0-1): Are technical details, vulnerability patterns, and references accurate?\n"
        "   - 0.0-0.3: Contains significant technical errors\n"
        "   - 0.4-0.6: Mostly accurate but with some mistakes\n"
        "   - 0.7-0.9: Accurate with minor issues\n"
        "   - 1.0: Technically flawless\n\n"
        "5. **Logical Flow** (0-1): Does the reasoning follow a logical progression from analysis to conclusion?\n"
        "   - 0.0-0.3: Disjointed or illogical progression\n"
        "   - 0.4-0.6: Somewhat logical but with gaps\n"
        "   - 0.7-0.9: Good logical flow with minor issues\n"
        "   - 1.0: Perfect logical progression\n\n"

        "**Output Format:**\n"
        "Provide your evaluation in the following JSON format:\n\n"
        "```json\n"
        "{{\n"
        '  "quality_score": <float 0-1>,  // OVERALL quality score (weighted combination of criteria above)\n'
        '  "correctness": <float 0-1>,  // Criterion 1 score\n'
        '  "completeness": <float 0-1>,  // Criterion 2 score\n'
        '  "clarity": <float 0-1>,  // Criterion 3 score\n'
        '  "technical_accuracy": <float 0-1>,  // Criterion 4 score\n'
        '  "logical_flow": <float 0-1>,  // Criterion 5 score\n'
        '  "confidence": <float 0-1>,  // How confident are you in this evaluation? (0=not confident, 1=very confident)\n'
        '  "justification": "<string>"  // Brief explanation (2-3 sentences) justifying the quality_score\n'
        "}}\n"
        "```\n\n"

        "**Important Notes:**\n"
        "- quality_score should be a weighted combination reflecting overall quality (not just an average)\n"
        "- confidence reflects your certainty in the evaluation (low if reasoning is ambiguous)\n"
        "- justification should be concise and factual, highlighting key strengths/weaknesses\n"
        "- Output ONLY valid JSON, no additional text before or after"
    ).strip()

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: str):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    def format_sample(self, sample: ReasoningSample) -> Messages:
        """Create prompt for judging reasoning quality"""

        cwe_info = (
            "\n".join([f"- {cwe}: {desc}" for cwe, desc in zip(sample.cwe, sample.cwe_desc)])
            if bool(sample.target)
            else "None"
        )

        prompt = self.PROMPT_SKELETON.format(
            project=sample.project,
            target=sample.target,
            cwe_info=cwe_info,
            func=sample.func,
            reasoning=sample.reasoning,
        )

        # no_answer_field_here: str = ""
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def count_individual_components(self, sample: ReasoningSample) -> TokensStats:
        messages = self.format_sample(sample)

        system_messages = messages[0]
        system_formatted = self.apply_template(messages=[system_messages])
        system_tokens = self._encode_and_count(text=system_formatted)

        system_user_formatted = self.apply_template(messages=messages)
        total_tokens = self._encode_and_count(text=system_user_formatted)
        user_tokens = total_tokens - system_tokens
        assistant_tokens = 0

        return TokensStats(
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            # reasoning_tokens=self._encode_and_count(sample.reasoning),
            # answer_tokens=0,
            assistant_tokens=assistant_tokens,
            total_tokens=total_tokens,
        )

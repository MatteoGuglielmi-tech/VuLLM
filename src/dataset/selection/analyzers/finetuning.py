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
from ..datatypes import Sample


class FineTunePromptAnalyzer(BaseSequenceLengthAnalyzer):
    """Analyzes token distribution in CoT dataset to determine optimal max_seq_length."""

    SYSTEM_PROMPT: str = (
        # Persona
        "You are a security expert specialized in C static code analysis. "

        # Task
        "Your task is to analyze C code and produce clear, pedagogical reasoning "
        "that explains the security assessment step-by-step.\n\n"

        # Analysis Guidelines
        "## Analysis Framework\n"
        "Follow these steps systematically in your reasoning:\n\n"
        "1. **Trace Data Flow**: Identify input sources and analyze how external data flows through the function\n"
        "2. **Identify Dangerous Patterns**: Look for unsafe functions (strcpy, gets, sprintf, malloc without bounds, etc.), "
        "unchecked operations, and potential overflows\n"
        "3. **Check Security Controls**: Assess bounds checking, input validation, error handling, and sanitization\n"
        "4. **Vulnerability Classification**: Map any identified issues to specific CWE categories\n\n"

        # Response Format
        "## Output Format\n"
        "Provide your analysis in this EXACT structure:\n\n"

        "**REASONING:**\n"
        "Write 2-4 concise sentences explaining the security analysis. "
        "Cover what the code does, what makes it dangerous or safe, and which CWEs apply if vulnerable.\n\n"

        "**VERDICT:**\n"
        "```json\n"
        "{{\n"
        '  "verdict": {{\n'
        '    "is_vulnerable": true/false,\n'
        '    "cwe_list": [list of CWE numbers as integers],\n'
        "  }}\n"
        "}}\n"
        "```\n\n"

        # Optional Example
        "Example output format:\n"
        "**REASONING:**\n"
        '"The function uses strcpy() to copy user input into a fixed-size buffer without bounds checking. '
        "This allows an attacker to overflow the buffer, potentially overwriting adjacent memory. "
        'This pattern matches CWE-119 (buffer overflow) and CWE-120 (classic buffer overflow)."\n\n'

        "**VERDICT:**\n"
        "```json\n"
        "{{\n"
        '  "verdict": {{\n'
        '    "is_vulnerable": true,\n'
        '    "cwe_list": [119, 120],\n'
        "  }}\n"
        "}}\n"
        "```\n\n"

        # Field Definitions
        "## Field Definitions\n"
        "- **is_vulnerable**: Boolean indicating whether the code is vulnerable (true/false)\n"
        "- **cwe_list**: CWEs with clear, traceable evidence in the code (as integer list)\n"

        # Requirements
        "## Requirements\n"
        "- Keep reasoning concise: 2-4 sentences, maximum 100 words, as a single text block\n"
        "- Write in natural, narrative prose (no bullet points in reasoning)\n"
        "- Verdict must be valid JSON in the exact structure above\n"
        '- CWE numbers as integers without prefix (e.g., 119 not "CWE-119")\n'
        "- If safe (no vulnerabilities): both 'cwe_list' and 'unsubstantiated_cwes' must be empty"
    )

    USER_PROMPT: str = (
        "Analyze this C code for security vulnerabilities.\n"

        "```c\n"
        "{func_code}\n"
        "```\n\n"
    ).strip()

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer)

    def format_sample(self, sample: Sample) -> tuple[str, str, str]:
        user_content = self.USER_PROMPT.format(func_code=sample["func"])
        assistant_content = f'{sample.get("reasoning", "")}'

        return self.SYSTEM_PROMPT, user_content, assistant_content

    def count_tokens_for_sample(self, sample: Sample) -> int:
        system_content, user_content, assistant_content = self.format_sample(sample)

        full_convo = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted = self.apply_template(messages=full_convo)

        return self._encode_and_count(text=full_formatted)


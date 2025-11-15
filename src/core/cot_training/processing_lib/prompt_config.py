from typing import Any
from dataclasses import dataclass, field

Messages = list[dict[str,str]]


@dataclass
class VulnerabilityPromptConfig:
    SYSTEM_PROMPT: str = field(
        init=False,
        default=(
            "You are an expert cybersecurity analyst specializing in C static code analysis. "
            "Your task is to analyze C code for security vulnerabilities using systematic reasoning "
            "and produce structured JSON output with your findings."
        ),
        repr=False,
    )

    USER_PROMPT: str = field(
        init=False,
        default=(
            "**Task:** Analyze the following C code for security vulnerabilities.\n\n"

            "**Code to Analyze:**\n"
            "--- CODE START ---\n"
            "```c\n"
            "{func_code}\n"
            "```\n\n"
            "--- CODE END ---\n\n"

            "**Analysis Framework:**\n"
            "Follow these steps systematically:\n\n"
            "1. **Data Flow Analysis:** Identify input sources and trace how data flows through the function\n"
            "2. **Dangerous Pattern Detection:** Identify unsafe functions (strcpy, gets, sprintf, etc.), unchecked operations, and potential overflows\n"
            "3. **Security Controls Assessment:** Check for bounds checking, input validation, error handling, and sanitization\n"
            "4. **Vulnerability Classification:** Map any identified issues to specific CWE categories\n\n"

            "**Output Format:**\n"
            "Provide your analysis as valid JSON in the following exact structure:\n\n"
            "```json\n"
            "{{\n"
            '  "reasoning": {{\n'
            '    "data_flow": "<string: describe input sources and data flow>",\n'
            '    "dangerous_patterns": "<string: list unsafe functions/operations found>",\n'
            '    "security_controls": "<string: describe protections present or missing>",\n'
            '    "classification": "<string: CWE mapping and severity assessment>"\n'
            "  }},\n"
            '  "vulnerabilities": [\n'
            "    {{\n"
            '      "description": "<string: what is the vulnerability>",\n'
            '      "location": "<string: where in the code (line/function)>",\n'
            '      "cwe_id": <int: CWE number, e.g., 119>,\n'
            '      "severity": "<string: low|medium|high|critical>"\n'
            "    }}\n"
            "  ],\n"
            '  "verdict": {{\n'
            '    "is_vulnerable": <boolean: true if vulnerabilities found, false otherwise>,\n'
            '    "cwe_list": [<int: list of CWE numbers, e.g., [119, 120]>],\n'
            '    "confidence": <float: 0.0-1.0, your confidence in this analysis>,\n'
            '    "summary": "<string: 2-3 sentence summary of findings>"\n'
            "  }}\n"
            "}}\n"
            "```\n\n"

            "**Critical Requirements:**\n"
            "- Output ONLY valid JSON in the exact structure shown above\n"
            "- Do not include any text before or after the JSON\n"
            "- Ensure all strings are properly quoted and escaped\n"
            "- The vulnerabilities array should be empty [] if no vulnerabilities are found\n"
            "- All fields are required - do not omit any\n"
            "- CWE IDs must be integers without the 'CWE-' prefix\n"
            "- DO NOT use '>' or '<' to encapsulate field values"
        ).strip(),
        repr=False,
    )

    def __iter__(self):
        """Allow tuple unpacking: system, user = VulnerabilityPromptConfig()"""
        return iter([self.SYSTEM_PROMPT, self.USER_PROMPT])
    
    @property
    def system(self) -> str:
        """System prompt."""
        return self.SYSTEM_PROMPT
    
    @property
    def user(self) -> str:
        """User prompt template."""
        return self.USER_PROMPT
    
    def format_user_prompt(self, func_code: str) -> str:
        """Format user prompt with function code."""
        return self.USER_PROMPT.format(func_code=func_code)
    
    def as_messages(self, func_code: str, ground_truth: str|None = None) -> Messages:
        """
        Format as messages list for chat models.

        Parameters
        ----------
        func_code: str
            C function source code

        Returns
        -------
            List of message dicts for chat API
        """

        scheleton = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.format_user_prompt(func_code)},
        ]

        if ground_truth is not None:
            scheleton.append({"role": "assistant", "content": ground_truth})

        return scheleton


@dataclass
class ParsedResponse:
    """Container for json response"""

    reasoning: dict[str, str]
    vulnerabilities: list[dict[str, str|int]]
    verdict: dict[str, Any]
    parse_error: bool = False

    def to_dict(self) -> dict:
        return {
            "reasoning": self.reasoning,
            "vulnerabilities": self.vulnerabilities,
            "verdict": self.verdict,
            "parse_error": self.parse_error,
        }

    @property
    def reasoning_info(self):
        return self.reasoning

    @property
    def vul_info(self):
        return self.vulnerabilities

    @property
    def verdict_info(self):
        return self.verdict

    @property
    def is_unable_to_parse(self):
        return self.parse_error


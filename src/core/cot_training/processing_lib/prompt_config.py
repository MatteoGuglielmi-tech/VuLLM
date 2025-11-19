from typing import Any, NotRequired, Optional, TypedDict
from dataclasses import dataclass, field
from transformers import BatchEncoding


Message = dict[str, str | list[int] | list[str] | list[list[int]] | BatchEncoding]
Messages = list[dict[str,str]]


class VulnInfo(TypedDict, total=True):
    cwe_id: int
    description: str


class VerdictStruct(TypedDict):
    # NotRequired allows empty verdict in case
    # of exception during response parsing
    is_vulnerable: NotRequired[bool]
    cwe_list: NotRequired[list[int]]


class ResponseStruct(TypedDict, total=True):
    reasoning: str
    vulnerabilities: list[VulnInfo]
    verdict: VerdictStruct


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
            "Follow these steps systematically in your reasoning:\n\n"
            "1. **Trace Data Flow:** Identify input sources and trace/analyze how external data flows through the function\n"
            "2. **Identify Dangerous Patterns:** Look for unsafe functions (strcpy, gets, sprintf, etc.), unchecked operations, potential overflows and possible dangerous steps\n"
            "3. **Check Security Controls:** Assess bounds checking, input validation, error handling, and sanitization\n"
            "4. **Vulnerability Classification:** Map any identified issues to specific CWE categories\n\n"

            "**Output Format:**\n"
            "Provide your analysis as valid JSON:\n\n"
            "```json\n"
            "{{\n"
            '  "reasoning": "<string: your step-by-step security analysis>",\n'
            '  "vulnerabilities": [\n'
            "    {{\n"
            '      "cwe_id": <int: CWE number without prefix, e.g., 119>,\n'
            '      "description": "<string: brief description of this specific vulnerability and what it is>"\n'
            "    }}\n"
            "  ],\n"
            '  "verdict": {{\n'
            '    "is_vulnerable": <boolean: true if vulnerabilities found, false otherwise>,\n'
            '    "cwe_list": [<int: list of CWE numbers, e.g., [119, 120]>]\n'
            "  }}\n"
            "}}\n"
            "```\n\n"

            "**Critical Requirements:**\n"
            "- Output ONLY valid JSON in the exact structure above\n"
            "- Do not include any text before or after the JSON\n"
            "- The 'reasoning' field must contain your complete analysis as a single text block\n"
            "- Include all four analysis steps in your reasoning (data flow, patterns, controls, classification)\n"
            "- The 'vulnerabilities' array should list each identified CWE with its description\n"
            "- The 'cwe_list' in verdict should mirror the CWE IDs from vulnerabilities array\n"
            "- If no vulnerabilities: set 'vulnerabilities' to [] and 'is_vulnerable' to false\n"
            "- CWE IDs must be integers (e.g., 119, not 'CWE-119')\n"
            "- Ensure proper JSON escaping"
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

    def as_messages(
        self, func_code: str, ground_truth: Optional[str] = None
    ) -> Messages:
        """
        Create chat messages for training.

        Parameters
        ----------
        func_code : str
            The C function code to analyze
        ground_truth : str, optional, default=None
            The expected JSON response (already formatted)

        Returns
        -------
        list[dict[str, str]]
            Messages in chat format [{"role": "...", "content": "..."}]
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.format_user_prompt(func_code=func_code)},
        ]

        if ground_truth is not None:
            messages.append({"role": "assistant", "content": ground_truth})

        return messages


@dataclass
class ParsedResponse:
    """Container for json response"""

    reasoning: str
    vulnerabilities: list[VulnInfo]
    verdict: VerdictStruct
    parse_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning": self.reasoning,
            "vulnerabilities": self.vulnerabilities,
            "verdict": self.verdict,
            "parse_error": self.parse_error,
        }

    @property
    def reasoning_info(self) -> str:
        return self.reasoning

    @property
    def vul_info(self) -> list[VulnInfo]:
        return self.vulnerabilities

    @property
    def verdict_info(self) -> VerdictStruct:
        return self.verdict

    @property
    def is_vulnerable(self) -> Optional[bool]:
        if self.parse_error:
            return None
        return self.verdict.get("is_vulnerable")

    @property
    def cwe_list(self) -> list[int]:
        """Extract CWE list from verdict, handling parse errors."""
        if self.parse_error:
            return []
        return self.verdict.get("cwe_list", [])

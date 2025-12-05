from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator
from transformers import BatchEncoding


Message = dict[str, str | list[int] | list[str] | list[list[int]] | BatchEncoding]
Messages = list[dict[str,str]]


@dataclass
class VulnerabilityPromptConfig:
    SYSTEM_PROMPT: str = field(
        init=False,
        default=(
            # Persona
            "You are a security expert specialized in C static code analysis. "

            # Task
            "Your task is to analyze C code and produce clear, pedagogical reasoning "
            "that explains the security assessment step-by-step.\n\n"

            # Analysis Guidelines
            "## Analysis Framework\n"
            "Follow these steps systematically in your reasoning:\n\n"
            "1. **Trace Data Flow:** Identify input sources and trace/analyze how external data flows through the function\n"
            "2. **Identify unknown APIs:** List all API/function calls unknown in the function context\n"
            "3. **Identify Dangerous Patterns:** Look for unsafe functions (strcpy, gets, sprintf, etc.), unchecked operations, potential overflows and possible dangerous steps\n"
            "4. **Check Security Controls:** Assess bounds checking, input validation, error handling, and sanitization\n"
            "5. **Vulnerability Classification:** Map any identified issues to specific CWE categories\n\n"

            # Make vulnerabilities based only on clear evidence
            "CRITICAL ASSUMPTIONS (MUST FOLLOW):\n"
            "1. All unknown/external function calls are SAFE and behave correctly according to their implied contract\n"
            "2. Return values from unknown functions are VALID and PROPERLY BOUNDED\n"
            "3. Memory returned by unknown allocators is PROPERLY SIZED and INITIALIZED\n"
            "4. Strings returned by unknown functions are PROPERLY NULL-TERMINATED\n"
            "5. Length values returned alongside buffers ACCURATELY REFLECT the buffer size\n"

            "DO NOT flag vulnerabilities based on:\n"
            "- Speculation about how unknown functions MIGHT fail\n"
            "- Missing null-termination checks for strings from unknown APIs\n"
            "- Missing bounds checks when lengths are provided by unknown APIs\n"
            "- Theoretical edge cases that require unknown functions to violate their implied contract\n"

            "ONLY flag vulnerabilities where:\n"
            "- The vulnerable pattern is ENTIRELY within the visible code\n"
            "- The bug would occur EVEN IF all unknown functions behave correctly\n"
            "- There is CONCRETE evidence of misuse (e.g., wrong format specifier, obvious off-by-one in visible loop etc)\n"
            "- If context is explicitly provided and it indicates an API may be unsafe or unreliable, "
            "you MAY flag vulnerabilities that depend on that API's behavior."

            "EXAMPLES OF NON-VULNERABILITIES (DO NOT FLAG):\n"

            "Example 1 - Trusting API return values:\n"
            "```c\n"
            "char *buf = get_buffer(&len);  // Unknown API\n"
            "memcpy(dest, buf, len);        // SAFE: len is trusted from API\n"
            "```\n"
            "This is SAFE because we trust get_buffer() returns correct len.\n"

            "Example 2 - Trusting null-termination:\n"
            "```c\n"
            "char *str = get_string();      // Unknown API\n"
            "int n = strlen(str);           // SAFE: str is trusted to be null-terminated\n"
            "```\n"
            "This is SAFE because we trust get_string() returns valid string.\n"

            "Example 3 - Trusting bounded values:\n"
            "```c\n"
            "parse_input(input, &offset, &length);  // Unknown API\n"
            "memcpy(dest, input + offset, length);  // SAFE: offset/length trusted\n"
            "```\n"
            "This is SAFE because we trust parse_input() returns valid bounds.\n"

            # Response Format
            "## Output Format\n"
            "Provide your analysis in this EXACT structure:\n"

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

            "Example output format:\n"
            "```json\n"
            "{{\n"
            '  "reasoning": "The function uses strcpy() to copy user input into a fixed-size buffer without bounds checking. This allows an attacker to overflow the buffer, potentially overwriting adjacent memory. This pattern matches CWE-119 (buffer overflow) and CWE-120 (classic buffer overflow).",\n'
            '  "vulnerabilities": [\n'
            "    {{\n"
            '      "cwe_id": 119,\n'
            '      "description": "The function uses `strcpy` to copy an input string into a fixed-size buffer without checking if the input exceeds the available buffer size, leading to buffer overflow."\n'
            "    }}\n"
            "  ],\n"
            '  "verdict": {{\n'
            '    "is_vulnerable": true,\n'
            '    "cwe_list": [119],\n'
            "  }}\n"
            "}}\n"
            "```\n\n"

            # Field Definitions
            "## Field Definitions\n"
            "- **reasoning**: 2 to 4 concise sentences explaining the security analysis. "
            "It covers what the code does, what makes it dangerous or safe, and which CWEs apply if vulnerable.\n"
            "- **vulnerabilities**: array of objects reporting all vulnerability information.\n"
            "- **verdict**: object with following properties:"
            "  - **is_vulnerable**: Boolean indicating whether the code is vulnerable (true/false)\n"
            "  - **cwe_list**: CWEs with clear, traceable evidence in the code (as integer list)\n"

            "## Critical Requirements\n"
            "- Output ONLY valid JSON in the exact structure above. "
            "Do not include any text before or after the JSON\n"

            "- Keep reasoning concise: 2-4 sentences, maximum 100 words, as a single text block. "
                "It must contain your complete analysis as a single text block\n"
            "- Include all 5 analysis steps in your reasoning (data flow, unknown calls, patterns, controls, classification)\n"
            "- Write in natural, narrative prose (no bullet points in reasoning)\n"

            "- The 'vulnerabilities' array should list each identified CWE with its description\n"
            "- The 'cwe_list' in verdict should mirror the CWE IDs from vulnerabilities array\n"

            "- If no vulnerabilities: set 'vulnerabilities' to [] and 'is_vulnerable' to false\n"
            "- CWE IDs must be integers without prefix (e.g., 119, not 'CWE-119')\n"
            "- Ensure proper JSON escaping"
            "- DO NOT use '>' or '<' to encapsulate field values"
        ),
        repr=False,
    )

    USER_PROMPT: str = field(
        init=False,
        default=(
            "## Code to Analyze\n"
            "```c\n"
            "{func_code}\n"
            "```\n\n"

            "Remember:\n"
            "During reasoning, bare in mind to ask yourself:\n" 
            "If I told the maintainer about this bug, would they say: \"that's impossible, our API guarantees X\"? "
            "If yes → NOT A VULNERABILITY.\n"
            "Focus solely on clear and solid evidence."
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


class VulnInfo(BaseModel):
    """Vulnerability information."""

    cwe_id: int
    description: str


class VerdictStruct(BaseModel):
    """Verdict structure."""

    is_vulnerable: bool
    cwe_list: list[int] = Field(default_factory=list)


class ExpectedModelResponse(BaseModel):
    """
    Container for parsed JSON response.

    Use model_validate_json() to parse from JSON string.
    Validation errors indicate parse failures.
    """

    reasoning: str = Field(..., description="Reasoning for the verdict")
    vulnerabilities: list[VulnInfo] = Field(
        default_factory=list, description="List of vulnerabilities"
    )
    verdict: VerdictStruct = Field(..., description="Final verdict")

    @property
    def is_vulnerable(self) -> bool:
        """Extract vulnerability status from verdict."""
        return self.verdict.is_vulnerable

    @property
    def cwe_list(self) -> list[int]:
        """Extract CWE list from verdict."""
        return list(set(self.verdict.cwe_list))

    @field_validator("reasoning")
    @classmethod
    def reasoning_not_empty(cls, v: str) -> str:
        """Validate reasoning is not just whitespace."""
        if not v.strip():
            raise ValueError("Reasoning cannot be empty or whitespace")
        return v.strip()

    @field_validator("vulnerabilities")
    @classmethod
    def check_vulnerabilities_match_verdict(
        cls, v: list[VulnInfo], info
    ) -> list[VulnInfo]:
        """Ensure vulnerabilities list matches verdict."""

        verdict = info.data.get("verdict")
        if verdict and verdict.get("is_vulnerable") and len(v) == 0:
            raise ValueError("If vulnerable, must have at least one vulnerability")
        return v

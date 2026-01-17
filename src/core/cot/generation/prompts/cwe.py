import logging

from typing import Optional, overload

from .typedefs import Messages, CweId, CWEDescription
from .factory import PromptTemplateFactory
from .base import PromptTemplate

logger = logging.getLogger(__name__)


@PromptTemplateFactory.register("cwe")
class CWEPromptTemplate(PromptTemplate):
    """
    Prompt template for generating Chain-of-Thought reasoning
    for vulnerability analysis fine-tuning data.

    Examples
    --------
    >>> template = CWEPromptTemplate()

    >>> # Simple usage
    >>> messages = template.build_messages(
    ...     func_code="void unsafe() { char buf[10]; gets(buf); }",
    ...     is_vulnerable=True,
    ...     cwe_ids=["CWE-119", "CWE-120"],
    ...     ground_truth_response="..."
    ... )

    >>> # With descriptions
    >>> messages = template.build_messages(
    ...     func_code="...",
    ...     is_vulnerable=True,
    ...     cwe_ids=["CWE-119"],
    ...     cwe_descs=["Improper Restriction of Operations within Memory Bounds"],
    ... )
    """

    SYSTEM_PROMPT = (
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
        '    "unsubstantiated_cwes": [CWEs from the known outcome that lack clear evidence]\n'
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
        '    "unsubstantiated_cwes": []\n'
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
        "  - **unsubstantiated_cwes**: CWEs mentioned in the known outcome but lacking clear code-level evidence (as integer list).\n\n"

        "## Critical Requirements\n"
        "- **Be honest**: if you cannot trace a CWE to specific code, list it as unsubstantiated;\n"
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
    )

    USER_PROMPT: str = (
        "Analyze this C code for security vulnerabilities.\n"
        "```c\n"
        "{func_code}\n"
        "```\n"

        "Known outcome:\n"
        "{known_outcome}"

        "During generation and reasoning, bare in mind to ask yourself:\n" 
        "If I told the maintainer about this bug, would they say: \"that's impossible, our API guarantees X\"? "
        "If yes → NOT A VULNERABILITY.\n"
        "Focus solely on clear and solid evidence."
    )

    def _build_cwe_info(
        self, cwe_ids: list[str], cwe_descs: Optional[list[str]] = None
    ) -> str:

        if cwe_descs:
            if len(cwe_ids) != len(cwe_descs):
                raise ValueError(f"Length mismatch: {len(cwe_ids)} CWE IDs vs {len(cwe_descs)} descriptions")

            outcome_parts: list[str] = [
                f"- {id}: {desc}" for id, desc in zip(cwe_ids, cwe_descs)
            ]
        else:
            outcome_parts: list[str] = [f"- {id}" for id in cwe_ids]

        return "\n".join(outcome_parts)

    def _build_known_outcome(
        self,
        is_vulnerable: bool,
        cwe_ids: Optional[list[str]] = None,
        cwe_descs: Optional[list[str]] = None,
    ) -> str:

        if not is_vulnerable:  # safe
            return "The function is SAFE (not vulnerable)."

        if cwe_ids: # vul with cwe_ids
            is_vulnerable_text = "The function is VULNERABLE with the following weaknesses:"
            cwe_info = self._build_cwe_info(cwe_ids=cwe_ids, cwe_descs=cwe_descs)
            return "\n".join([is_vulnerable_text, cwe_info])
        else: # vul no cwe_ids
            return "The function is VULNERABLE but no weaknesses have been specified."

    @overload
    def format_user_prompt(
        self,
        *,
        func_code: str,
        is_vulnerable: bool,
        cwe_ids: list[str] | None = None, # "CWE-XXX"
        cwe_desc: list[str] | None = None,
    ) -> str: ...

    @overload
    def format_user_prompt(self, **kwargs) -> str: ...

    def format_user_prompt(
        self,
        func_code: Optional[str] = None,
        is_vulnerable: Optional[bool] = None,
        cwe_ids: Optional[list[str]] = None,
        cwe_descs: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        """
        Format the user prompt for vulnerability analysis.

        Parameters
        ----------
        func_code : str
            C function code to analyze
        is_vulnerable : bool
            Whether the function is known to be vulnerable
        cwe_ids : list[str] | None
            List of CWE identifiers (e.g., ["CWE-119", "CWE-120"])
        cwe_descs : list[str] | None
            Descriptions for each CWE (must match cwe_ids length)
        **kwargs
            Alternative way to pass arguments (for parent compatibility)

        Returns
        -------
        str
            Formatted user prompt
        """

        # kwargs
        if func_code is None and is_vulnerable is None:
            func_code = kwargs.get("func_code")
            is_vulnerable = kwargs.get("is_vulnerable")
            cwe_ids = kwargs.get("cwe_ids")
            cwe_descs = kwargs.get("cwe_descs")

        # Validate required arguments
        if func_code is None or is_vulnerable is None:
            raise TypeError(
                "Missing required arguments: 'func_code' and 'is_vulnerable' must be provided"
            )

        known_outcome = self._build_known_outcome(
            is_vulnerable=is_vulnerable, cwe_ids=cwe_ids, cwe_descs=cwe_descs
        )
        return self.user.format(
            func_code=func_code.strip(), known_outcome=known_outcome
        )

    def as_messages(self, formatted_user_content: str) -> Messages:
        """Build messages (parent interface)."""
        return super().as_messages(formatted_user_content)

    def build_messages(
        self,
        func_code: str,
        is_vulnerable: bool,
        cwe_ids: Optional[list[CweId]] = None,
        cwe_descs: Optional[list[CWEDescription]] = None,
    ) -> Messages:
        """
        Build messages for vulnerability detection (convenience method).

        Parameters
        ----------
        func_code : str
            C function code to analyze
        is_vulnerable : bool
            Whether the function is vulnerable
        cwe_ids : list[str] | None
            List of CWE identifiers
        cwe_descs : list[str] | None
            Descriptions for each CWE

        Returns
        -------
        Messages
            List of message dicts in chat format
        """
        formatted_prompt = self.format_user_prompt(
            func_code=func_code,
            is_vulnerable=is_vulnerable,
            cwe_ids=cwe_ids,
            cwe_descs=cwe_descs,
        )
        return self.as_messages(formatted_prompt)

from jinja2 import Template

from .base import BaseVulnPrompt
from .cwe_mapping_guidance import CWE_MAPPING_GUIDANCE_V1
from ..datatypes import PromptPhase, AssumptionMode


class VulnPromptV1(BaseVulnPrompt):
    def __init__(
        self,
        phase: PromptPhase,
        assumptions_mode: AssumptionMode,
        add_cwe_guidelines: bool,
        debug_mode: bool,
    ):
        self.phase = phase
        self.assumptions_mode = assumptions_mode
        self.add_cwe_guidelines = add_cwe_guidelines
        self.debug_mode = debug_mode
        self.CWE_GUIDANCE = self.set_guidelines()
    
    @property
    def OPTIMISTIC_ASSUMPTIONS(self) -> str:
        return (
            "## Optimistic Assumptions\n\n"

            "Assume ALL unknown/external functions:\n"
            "- Return valid, correctly bounded values\n"
            "- Provide properly sized and initialized memory\n"
            "- Return null-terminated strings where expected\n"
            "- Have accurate length values\n\n"

            "FLAG as vulnerable ONLY when:\n"
            "- Vulnerability is entirely within visible code\n"
            "- Bug would occur even if all unknown functions behave correctly\n"
            "- Concrete evidence of misuse (wrong format specifier, clear off-by-one)\n\n"

            "DO NOT flag when:\n"
            "- Vulnerability requires unknown function to misbehave\n"
            "- Protective measures could plausibly exist elsewhere"
        )

    @property
    def OPTIMISTIC_REMINDER(self) -> str:
        return (
            "Remember:\n"
            "Ask yourself: \"If I reported this bug, would the maintainer say: 'that's impossible, our API guarantees X'?\"\n"
            "If YES → NOT A VULNERABILITY.\n"
            "Focus solely on bugs provable from visible code alone."
        )
    
    @property
    def PESSIMISTIC_ASSUMPTIONS(self) -> str:
        return (
            "## Pessimistic Assumptions\n\n"

            "### External Functions\n"
            "Assume ALL unknown/external functions may:\n"
            " - Violate their implied contract\n"
            " - Return invalid, out-of-bounds, or malicious values\n"
            " - Provide incorrectly sized, uninitialized or NULL memory\n"
            " - Return non-null-terminated strings\n"
            " - Have incorrect length values\n\n"

            "### Locally Provable Bugs\n"
            "FLAG when the bug is visible from the code alone:\n"
            "- Array read exceeds bounds: `int arr[5]; return arr[10];` -> CWE-125\n"
            "- Array write exceeds bounds: `int arr[5]; arr[10] = x;` -> CWE-787\n"
            "- Loop exceeds array size: `for(i=0; i<20; i++) arr[i]` on smaller array -> CWE-125/787\n"
            "- Integer overflow in arithmetic operations -> CWE-190\n"
            "- Use after free: `free(p); *p = x;` -> CWE-416\n"
            "- Double free: `free(p); free(p);` -> CWE-415\n"
            "- NULL dereference: `int *p = NULL; return *p;` -> CWE-476\n"
            "- Memory leak: allocated memory becomes unreachable -> CWE-401\n\n"

            "### FLAG as Vulnerable When\n"
            "- External data used without explicit validation\n"
            "- Missing bounds/null checks on API returns\n"
            "- Operations depending on external behavior for safety\n"
            "- Array/pointer access with index provably out of bounds\n"
            "- Bug is provable from visible code alone\n\n"

            "### SAFE Only When\n"
            "- Explicit validation of all inputs (null checks, bounds checks)\n"
            "- Buffer sizes statically known AND all accesses provably bounded\n"
            "- All error conditions and edge cases handled\n"
            "- **Utility wrappers that correctly propagate errors (NULL checks) to caller**\n"
        )

    @property
    def PESSIMISTIC_REMINDER(self) -> str:
        return (
            "## CRITICAL REMINDERS\n\n"

            "### Check Local Bugs First\n"
            "Before considering external input, verify:\n"
            "- No array access exceeds declared bounds\n"
            "- No loop iterates beyond array size\n"
            "- No NULL/freed pointer is dereferenced\n"
            "- No memory is used after free\n"
            "- No double free\n\n"

            "If ANY local bug exists -> FLAG immediately.'No external input' is NOT a defense.\n\n"

            "### Single CWE Rule\n"
            "Output ONLY the most specific CWE per vulnerability:\n\n"

            "Memory buffer operations:\n"
            "- Buffer write -> CWE-787 (not 119)\n"
            "- Buffer read -> CWE-125 (not 119)\n"
            "- Unbounded copy (strcpy, gets, sprintf, etc.) -> CWE-120 (not 119)\n\n"

            "Memory lifecycle:\n"
            "- Use after free -> CWE-416\n"
            "- Double free -> CWE-415\n"
            "- Allocation without free -> CWE-401\n\n"

            "Other:\n"
            "- NULL dereference -> CWE-476\n"
            "- Integer overflow -> CWE-190\n\n"

            "NEVER output parent and child together (e.g., never [119, 787])\n\n"

            "### Pessimistic Analysis\n"
            "FLAG as vulnerable if EITHER:\n"
            "1. Local bug is provable from code alone\n"
            "2. External data used without validation\n"
            "3. Return value from unknown API used unchecked\n\n"

            "When in doubt -> FLAG with the most specific CWE."
        )

    @property
    def FINAL_ENFORCEMENT(self) -> str:
        return(
            "## BEFORE YOU RESPOND\n\n"
            "1. Did you check for LOCAL BUGS (OOB access, NULL deref, UAF, double-free, memory leaks, etc.)?\n"
            "2. Did you ask 'Is this operation safe?' — not 'Is this exploitable?'\n"
            "3. If ANY operation violates safety -> FLAG it.\n"
            "4. Did you use the MOST SPECIFIC CWE?\n\n"

            "Do NOT second-guess. Do NOT dismiss bugs as 'not exploitable'.\n"
        )

    @property
    def SYSTEM_PROMPT(self) -> Template:
        return Template(
            (
                # Persona & task
                "You are a security expert specialized in C static code analysis. "
                "Your task is to analyze C code for security assessment and produce clear, pedagogical reasoning.\n\n"

                "## CRITICAL CONSTRAINT\n"
                "Predict the MINIMUM set of CWEs — ideally ONE per distinct root cause. " 
                "Do NOT list multiple related CWEs (e.g., do not output both 119 and 787).\n\n"

                "{% if cwe_guidance %}"
                "{{ cwe_guidance }}"
                "\n\n"
                "{% endif %}"

                "## Analysis Steps\n"
                "Follow these steps systematically:\n\n"

                "1. **Check for locally provable bugs first:**\n"
                "  Focus on OPERATIONS, not on who controls the input.\n\n"
                "  - Array/buffer: Does any index exceed declared bounds?\n"
                "  - Loops: Does iteration exceed array limits?\n"
                "  - Pointers: Is any pointer NULL or freed before dereference?\n"
                "  - Memory lifecycle: Does allocated memory become unreachable without being freed? (critical in long/infinite loops)\n"
                "  - Arithmetic: Can size/index computation overflow? (be careful about signed vs unsigned)\n\n"

                "2. **Trace data flow:**\n"
                "  - Identify function parameters, API return values, and global state\n"
                "  - Track how this data reaches buffers, pointers, or size calculations\n\n"

                "3. **Check for dangerous patterns:**\n"
                "  - Unbounded copies (strcpy, gets, sprintf, memcpy with untrusted size, etc.)\n"
                "  - Unchecked arithmetic on sizes or indices\n"
                "  - Missing bounds/NULL checks before dereferencing, indexing, or passing pointers/arrays\n\n"

                "4. **Classify using the most specific ALLOWED CWE**\n\n"

                "## Output Format\n"
                "Respond in ONLY this JSON schema (compact, single line):\n"
                "```json\n"
                "{% raw %}"
                '{"reasoning": "<string>", "vulnerabilities": [{"cwe_id": <int>, "description": "<string>"}], "verdict": {"is_vulnerable": <bool>, "cwe_list": [<int>]}}'
                "{% endraw %}\n"
                "```\n\n"

                "## Field Definitions\n"
                "- **reasoning**: Step-by-step security analysis. Include:\n"
                "  (1) Local bugs found (or confirmed absent)\n"
                "  (2) Data flow concerns\n"
                "  (3) Dangerous patterns\n"
                "  (4) Final assessment with CWE justification\n"
                "- **vulnerabilities**: Array of objects with cwe_id (int) and description (string).\n"
                "- **verdict**: Object with:\n"
                "  - **is_vulnerable**: boolean (true/false)\n"
                "  - **cwe_list**: integer list matching cwe_ids in vulnerabilities array\n\n"

                "## Output Requirements\n"
                "- JSON only — no text before or after\n"
                '- CWE IDs as integers (e.g. 787, not "CWE-787")\n'
                "- If safe: vulnerabilities=[], is_vulnerable=false\n"
                "- cwe_list must match cwe_ids in vulnerabilities array\n"
                "- MINIMUM CWEs; prefer ALLOWED over DISCOURAGED"
            )
        )

    @property
    def USER_PROMPT(self) -> Template:
        return Template(
            (
                "The code shown IS the complete context.\n" 
                "Focus more on WHAT the code does rather than WHO triggers it.\n"

                "{% if mode == 'pessimistic' and assumptions %}"
                "Apply the Pessimistic Assumptions section STRICTLY.\n\n"
                "{% elif  mode == 'optimistic' and assumptions %}"
                "Apply the Optimistic Assumptions section STRICTLY.\n\n"
                "{% else %}"
                "\n"
                "{% endif %}"

                "## Code to Analyze\n\n"
                "```c\n"
                "{{func_code}}\n"
                "```"

                "{% if assumptions %}"
                "\n\n"
                "{{ assumptions }}"
                "{% endif %}"

                "{% if reminder %}"
                "\n\n"
                "{{ reminder }}"
                "{% endif %}"

                "\n\n"
                "{{ final_reinforcement }}"
            )
        )

    def format_system_prompt(self) -> str:
        """Format system prompt with optional assumptions."""
        return self.SYSTEM_PROMPT.render(cwe_guidance=self.CWE_GUIDANCE)

    def format_user_prompt(self, func_code: str) -> str:
        """Format user prompt with optional attack patterns."""

        assumptions, reminder = self.get_assumption_reminder_combo()
        return self.USER_PROMPT.render(
            func_code=func_code,
            mode=self.assumptions_mode,
            assumptions=assumptions,
            reminder=reminder,
            final_reinforcement=self.FINAL_ENFORCEMENT,
        )

    def set_guidelines(self) -> str | None:
        return CWE_MAPPING_GUIDANCE_V1 if self.add_cwe_guidelines else None

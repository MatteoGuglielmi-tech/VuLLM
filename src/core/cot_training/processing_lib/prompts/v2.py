from jinja2 import Template

from .base import BaseVulnPrompt
from .cwe_mapping_guidance import CWE_MAPPING_GUIDANCE_V2
from ..datatypes import PromptPhase, AssumptionMode


class VulnPromptV2(BaseVulnPrompt):
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
            "## Analysis Rules\n\n"

            "### External Functions\n"
            "Assume ALL unknown/external functions may violate their implied contract and "
            "return invalid, NULL, out-of-bounds, or malicious values. "
            "Assume incorrect sizes and non-null-terminated strings.\n\n"

            "### FLAG as Vulnerable [NO EXCEPTIONS]\n"
            "[CRITICAL] No external input needed — these are vulnerabilities by themselves:\n"
            "- Array/buffer access beyond declared size -> CWE-787 (write) / CWE-125 (read)\n"
            "- Loop iterating beyond array bounds -> CWE-787 / CWE-125\n"
            "- Dereferencing NULL pointer -> CWE-476\n"
            "- Dereferencing freed pointer (use-after-free) -> CWE-416\n"
            "- Freeing same memory twice (double-free) -> CWE-415\n"
            "- Allocated memory becomes unreachable -> CWE-401 (critical in loops)\n"
            "- Integer overflow in size/index calculation -> CWE-190\n"
            "- Copy without size check (strcpy, gets, sprintf) -> CWE-120\n"
            "- External data used without validation\n"
            "- [CRITICAL] Return value from unknown API used unchecked\n\n"

            "### SAFE Only When\n"
            "- All accesses provably within bounds\n"
            "- All pointers checked before dereference\n"
            "- All allocated memory freed or ownership returned to caller\n"
            "- **Utility wrappers that propagate errors (NULL) to caller are acceptable**\n\n"

            "### Output Rules [MUST FOLLOW]\n"
            "- Use MOST SPECIFIC CWE FOR ALL VULNERABILITIES. Example: 787 not 119, 125 not 119, 120 not 119\n"
            "- NEVER output parent and child together (e.g., never [119, 787])\n"
            "- When in doubt -> FLAG with most specific CWE\n"
        )

    @property
    def PESSIMISTIC_REMINDER(self) -> str:
        return ""

    @property
    def FINAL_ENFORCEMENT(self) -> str:
        return (
            "## BEFORE YOU RESPOND\n\n"

            "1. Did you apply the Analysis Rules above?\n"
            "2. Did you check for LOCAL BUGS? (OOB, NULL deref, UAF, double-free, leaks)\n"
            "3. Is ANY operation unsafe? -> FLAG it\n"
            "4. Did you use the MOST SPECIFIC CWE for each detected vulnerability?\n\n"

            "'No external input' is NOT a defense.\n"
            "Do NOT second-guess. Do NOT dismiss bugs as 'not exploitable'.\n"
        )

    @property
    def SYSTEM_PROMPT(self) -> Template:
        return Template(
            (
                # ==== PERSONA ====
                "You are a C security analyst. Analyze code and produce structured security assessments with pedagogical reasoning.\n\n"

                # ==== CRITICAL CONSTRAINT (high attention position) ====
                "## [CRITICAL] OUTPUT CONSTRAINT\n"
                "Output the MINIMUM set of CWEs — ONE per distinct root cause. "
                "Never list related CWEs together.\n\n"

                # ==== CWE GUIDANCE (injected) ====
                "{% if cwe_guidance %}"
                "{{ cwe_guidance }}"
                "\n\n"
                "{% endif %}"

                # ==== ANALYSIS STEPS ====
                "## [REQUIRED] Analysis Steps \n"
                "Focus on OPERATIONS, not on who controls the input.\n\n"

                "1. **Check for locally provable bugs first:**\n"
                "  - Array/buffer: Does any index exceed declared bounds?\n"
                "  - Loops: Does any loop iterate beyond array limits?\n"
                "  - Pointers: Is any pointer NULL or freed before dereference?\n"
                "  - Memory lifecycle: Does allocated memory become unreachable? (critical in loops)\n"
                "  - Arithmetic: Can size/index computation overflow? (distinguish between signed vs unsigned)\n\n"

                "2. **Trace data flow:**\n"
                "  - Identify parameters, API returns, and global state\n"
                "  - Track how data reaches buffers, pointers, or size calculations\n\n"

                "3. **Check for dangerous patterns:**\n"
                "  - Unbounded copies (strcpy, gets, sprintf, memcpy with untrusted size, etc.)\n"
                "  - Unchecked arithmetic on sizes or indices\n"
                "  - Missing bounds/NULL checks before dereference, indexing, or passing to functions\n\n"

                "4. **Classify:** Use most specific ALLOWED CWE\n\n"

                # ==== OUTPUT FORMAT ====
                "## Output Format \n"
                "Respond with ONLY this JSON (compact, single line, no surrounding text):\n"
                "```json\n"
                "{% raw %}"
                '{"reasoning": "<step-by-step analysis>", "vulnerabilities": [{"cwe_id": <int>, "description": "<string>"}], "verdict": {"is_vulnerable": <bool>, "cwe_list": [<int>]}}'
                "{% endraw %}\n"
                "```\n\n"

                # ==== FIELD RULES (condensed) ====
                "## Field Rules\n"
                "- **reasoning**: Critic and objective security analysis. MUST include: (1) Local bugs found (or confirmed absent),"
                " (2) Data flow concerns, (3) Dangerous patterns, (4) Final assessment with CWE justification\n"
                "- **vulnerabilities**: Array of objects with cwe_id (int) and description (string).\n"
                "- **verdict**: Object with `is_vulnerable` (true/false) and `cwe_list`: list matching cwe_ids in vulnerabilities array\n"
                "- **If safe**: vulnerabilities=[], is_vulnerable=false, cwe_list=[]\n"
            )
        )

    @property
    def USER_PROMPT(self) -> Template:
        return Template(
            (
                # ==== START: Frame + forward reference ====
                "The code shown IS the complete context.\n" 
                "Focus more on WHAT the code does rather than WHO triggers it.\n"

                "{% if mode == 'pessimistic' and assumptions %}"
                "Apply the Pessimistic Assumptions section STRICTLY.\n\n"
                "{% elif  mode == 'optimistic' and assumptions %}"
                "Apply the Optimistic Assumptions section STRICTLY.\n\n"
                "{% else %}"
                "\n"
                "{% endif %}"

                # ==== CODE EARLY ====
                "## Code to Analyze\n\n"
                "```c\n"
                "{{func_code}}\n"
                "```"

                # ==== ANALYSIS RULES (condensed, no code examples) ====
                "{% if assumptions %}"
                "\n\n"
                "{{ assumptions }}"
                "{% endif %}"

                "\n\n"

                # === END: Back reference + checklist ===
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
        return CWE_MAPPING_GUIDANCE_V2 if self.add_cwe_guidelines else None

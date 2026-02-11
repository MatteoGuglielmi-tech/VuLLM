from dataclasses import dataclass, field
from transformers import BatchEncoding
from jinja2 import Template
from pathlib import Path

from .datatypes import PromptPhase, AssumptionMode
from .cwe_mapping_guidance import CWE_MAPPING_GUIDANCE

Message = dict[str, str | list[int] | list[str] | list[list[int]] | BatchEncoding]
Messages = list[dict[str,str]]


@dataclass
class VulnerabilityPromptConfig: 

    debug_mode: bool = False

    CWE_GUIDANCE: str = field(
        init=False,
        default=CWE_MAPPING_GUIDANCE,
        repr=False,
    )

    # ====================================================================
    # SPECIFIC ASSUMPTIONS (Technical Rules)
    # ====================================================================
    OPTIMISTIC_ASSUMPTIONS: str = field(
        init=False,
        default=(
            "## Optimistic Assumptions\n"
            "Assume ALL unknown/external functions:\n"
            "- Return valid, correctly bounded values\n"
            "- Provide properly sized and initialized memory\n"
            "- Return null-terminated strings where expected\n"
            "- Have accurate length values\n\n"

            "FLAG as vulnerable ONLY when:\n"
            "- Vulnerability is entirely within visible code\n"
            "- Bug would occur even if all unknown functions behave correctly\n"
            "- Concrete evidence of misuse (wrong format specifier, clear off-by-one)\n"

            "DO NOT flag when:\n\n"
            "- Vulnerability requires unknown function to misbehave\n"
            "- Protective measures could plausibly exist elsewhere"
        ),
        repr=False
    )

    OPTIMISTIC_REMINDER: str = field(
        init=False,
        default=(
            "Remember:\n"
            "Ask yourself: \"If I reported this bug, would the maintainer say: 'that's impossible, our API guarantees X'?\"\n"
            "If YES → NOT A VULNERABILITY.\n"
            "Focus solely on bugs provable from visible code alone."
        ),
        repr=False,
    )

    # Pessimistic assumptions
    PESSIMISTIC_ASSUMPTIONS: str = field(
        init=False,
        default=(
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
        ),
        repr=False
    )

    PESSIMISTIC_REMINDER: str = field(
        init=False,
        default=(
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
            "- Unbounded copy (strcpy, gets) -> CWE-120 (not 119)\n\n"

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
        ),
        repr=False,
    )

    FINAL_ENFORCEMENT: str = field(
        init=False,
        default=(
            "## BEFORE YOU RESPOND\n\n"
            "1. Did you check for LOCAL BUGS (OOB access, NULL deref, UAF, double-free, memory leaks, etc.)?\n"
            "2. Did you ask 'Is this operation safe?' — not 'Is this exploitable?'\n"
            "3. If ANY operation violates safety -> FLAG it.\n"
            "4. Did you use the MOST SPECIFIC CWE?\n\n"

            "Do NOT second-guess. Do NOT dismiss bugs as 'not exploitable'.\n"
        ),
        repr=False,
    )

    # ====================================================================
    # PROMPT TEMPLATES
    # ====================================================================
    SYSTEM_PROMPT: Template = field(
        init=False,
        default=Template(
            (
                # Persona & task
                "You are a security expert specialized in C static code analysis. "
                "Your task is to analyze C code for security assessment and produce clear, pedagogical reasoning.\n\n"

                "## CRITICAL CONSTRAINT\n"
                "Predict the MINIMUM set of CWEs — ideally ONE per distinct root cause. " 
                # "Multiple CWEs are valid only for INDEPENDENT vulnerabilities. "
                "Do NOT list multiple related CWEs (e.g., do not output both 119 and 787).\n\n"

                "{% if cwe_guidance %}"
                "{{ cwe_guidance }}"
                "\n\n"
                "{% endif %}"

                # "{% if assumptions %}"
                # "{{ assumptions }}"
                # "\n\n"
                # "{% endif %}"

                "## Analysis Steps\n"
                "Follow these steps systematically:\n\n"

                "1. **Check for locally provable bugs first:**\n"
                "  Focus on OPERATIONS, not on who controls the input.\n\n"
                "  - Array/buffer: Does any index exceed declared bounds?\n"
                "  - Loops: Does iteration exceed array limits?\n"
                "  - Pointers: Is any pointer NULL or freed before dereference?\n"
                "  - Memory lifecycle: Does allocated memory become unreachable without being freed? (critical in long/infinite loops)\n"
                # "- Memory lifecycle: Is memory allocated but never freed or ownership-transferred? (watch for loops)\n"
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
        ),
        repr=False,
    )

    USER_PROMPT: Template = field(
        init=False,
        default=Template(
            (
                "The code shown IS the complete context.\n" 
                "Focus more on WHAT the code does rather than WHO triggers it.\n\n"

                "## Code to Analyze\n\n"
                "```c\n"
                "{{func_code}}\n"
                "```"

                "{% if assumptions %}"
                "\n\n"
                "{{ assumptions }}"
                "{% endif %}"

                # "{% if attack_patterns %}"
                # "\n\n"
                # "{{ attack_patterns }}"
                # "{% endif %}"

                "{% if reminder %}"
                "\n\n"
                "{{ reminder }}"
                "{% endif %}"

                "\n\n"
                "{{ final_reinforcement }}"
            )
        ),
        repr=False,
    )

    def __iter__(self):
        """Allow tuple unpacking: system, user = VulnerabilityPromptConfig()"""
        return iter([self.SYSTEM_PROMPT, self.USER_PROMPT])

    def get_assumptions(self, mode: AssumptionMode) -> str | None:
        """Get assumptions text for the given mode."""
        match mode:
            case AssumptionMode.OPTIMISTIC:
                return self.OPTIMISTIC_ASSUMPTIONS
            case AssumptionMode.PESSIMISTIC:
                return self.PESSIMISTIC_ASSUMPTIONS
            case AssumptionMode.NONE:
                return None

    def get_reminder(self, mode: AssumptionMode) -> str | None:
        """Get reminder text for the given mode."""
        match mode:
            case AssumptionMode.OPTIMISTIC:
                return self.OPTIMISTIC_REMINDER
            case AssumptionMode.PESSIMISTIC:
                return self.PESSIMISTIC_REMINDER
            case AssumptionMode.NONE:
                return None

    def get_assumption_reminder_combo(
        self, phase: PromptPhase, mode: AssumptionMode
    ) -> tuple[str | None, str | None]:
        match phase:
            case (
                PromptPhase.CONSTRAINED_TRAINING
                | PromptPhase.FULL_CONSTRAINED_INFERENCE
            ):
                assumption = self.get_assumptions(mode=mode)
                reminder = self.get_reminder(mode=mode)
                return (assumption, reminder)
            case (
                PromptPhase.FREE_TRAINING
                | PromptPhase.ATTACK_CONSTRAINED_INFERENCE
                | PromptPhase.FREE_INFERENCE
            ):
                return (None, None)

    # def format_system_prompt(self, assumptions: str | None, add_hierarchy: bool) -> str:
    def format_system_prompt(self, add_hierarchy: bool) -> str:
        """Format system prompt with optional assumptions."""
        return self.SYSTEM_PROMPT.render(
            cwe_guidance=self.CWE_GUIDANCE if add_hierarchy else None
        )

    def format_user_prompt(
        self,
        func_code: str,
        # phase: PromptPhase,
        assumptions: str | None,
        reminder: str | None
    ) -> str:
        """Format user prompt with optional attack patterns."""

        # attack_patterns = (
        #     self.SEEN_ATTACK_PATTERNS
        #     if phase
        #     in [
        #         PromptPhase.FULL_CONSTRAINED_INFERENCE,
        #         PromptPhase.ATTACK_CONSTRAINED_INFERENCE,
        #     ]
        #     else None
        # )
        return self.USER_PROMPT.render(
            func_code=func_code,
            reminder=reminder,
            assumptions=assumptions,
            # attack_patterns=None,
            final_reinforcement=self.FINAL_ENFORCEMENT,
        )

    def as_messages(
        self,
        func_code: str,
        phase: PromptPhase,
        mode: AssumptionMode,
        add_hierarchy: bool,
        ground_truth: str | None = None,
    ) -> Messages:
        """
        Create chat messages.

        Parameters
        ----------
        func_code : str
            Function code to analyze
        ground_truth : str | None
            Ground truth JSON response (for training)
        mode : AssumptionMode
            Assumption mode (optimistic, pessimistic, none)
        phase : PromptPhase
            Training or inference phase
        add_hierarchy: bool
            Add cwe hierarchy guidelines to system prompt

        Returns
        -------
        list[dict[str, str]]
            Chat messages in format [{"role": "system", "content": ...}, ...]
        """

        user_assumptions, user_reminder = self.get_assumption_reminder_combo(phase=phase, mode=mode)
        system_prompt = self.format_system_prompt(add_hierarchy=add_hierarchy)

        user_prompt = self.format_user_prompt(
            func_code=func_code,
            assumptions=user_assumptions,
            reminder=user_reminder
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if (
            phase
            in [PromptPhase.CONSTRAINED_TRAINING, PromptPhase.FREE_TRAINING]
            and ground_truth is not None
        ):
            messages.append({"role": "assistant", "content": ground_truth})

        # debug step
        if self.debug_mode:
            # I know, the file is override at any entry
            self._serialize_in_use_prompt(
                phase=phase,
                mode=mode,
                add_hierarchy=add_hierarchy,
                sys_prompt=system_prompt,
                user_prompt=user_prompt,
                ground_truth=ground_truth,
            )

        return messages

    def _serialize_in_use_prompt(
        self,
        phase: PromptPhase,
        mode: AssumptionMode,
        add_hierarchy: bool,
        sys_prompt: str,
        user_prompt: str,
        ground_truth: str|None,
        dst: str = "prompt",
    ):
        parent_dir = Path(dst)
        parent_dir.mkdir(parents=True, exist_ok=True)
        filename: str = f"{phase}_{mode}{"_hierarchy" if add_hierarchy else ""}.txt"
        with open(file=parent_dir / filename, mode="w") as text_file:
            text_file.write(50 * "=" + " System prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(sys_prompt)
            text_file.write("\n\n")
            text_file.write(50 * "=" + " User prompt " + 50 * "=")
            text_file.write("\n")
            text_file.write(user_prompt)
            text_file.write("\n\n")
            if ground_truth is not None:
                text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
                text_file.write("\n")
                text_file.write(ground_truth)

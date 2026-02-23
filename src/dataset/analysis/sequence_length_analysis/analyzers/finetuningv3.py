import json

from transformers import PreTrainedTokenizer
from jinja2 import Template

from .base import BaseSequenceLengthAnalyzer
from ..datatypes import (
    Messages,
    ReasoningSample,
    TokensStats,
    VulnInfo,
    ExpectedModelResponse,
    VerdictStruct,
    AssumptionMode,
    PromptPhase,
)
from .cwe_guidance import CWE_MAPPING_GUIDANCE_V2


class FineTunePromptAnalyzerV3(BaseSequenceLengthAnalyzer):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template: str,
        prompt_phase: PromptPhase,
        assumption_mode: AssumptionMode,
        add_hierarchy: bool
    ):
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.assumption_mode = assumption_mode
        self.prompt_phase = prompt_phase
        self.add_hierarchy = add_hierarchy

        self.CWE_GUIDANCE: str = CWE_MAPPING_GUIDANCE_V2

        # ====================================================================
        # SPECIFIC ASSUMPTIONS (Technical Rules)
        # ====================================================================
        self.OPTIMISTIC_ASSUMPTIONS: str = (
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
        )

        self.OPTIMISTIC_REMINDER: str = (
            "Remember:\n"
            "Ask yourself: \"If I reported this bug, would the maintainer say: 'that's impossible, our API guarantees X'?\"\n"
            "If YES → NOT A VULNERABILITY.\n"
            "Focus solely on bugs provable from visible code alone."
        )

        # Pessimistic assumptions
        self.PESSIMISTIC_ASSUMPTIONS: str = (
            "## Analysis Rules\n\n"

            "### External Functions\n"
            "Assume ALL unknown/external functions may violate their implied contract and:\n"
            "- Return invalid, NULL, out-of-bounds, or malicious values\n"
            "- Provide incorrect sizes and non-null-terminated strings\n\n"
            "Analyze OBSERVABLE behavior only.\n"
            "[CRITICAL] Do NOT speculate about what unknown functions do internally "
            "(e.g., do not assume they free passed memory/pointers - report leak as observable).\n"

            "### FLAG as Vulnerable [NO EXCEPTIONS]\n"
            "[CRITICAL] No external input needed — these are vulnerabilities by themselves:\n"
            "- Array/buffer access beyond declared size -> CWE-787 (write) / CWE-125 (read)\n"
            "- Loop iterating beyond array bounds -> CWE-787 / CWE-125\n"
            "- Dereferencing NULL pointer -> CWE-476\n"
            "- Dereferencing freed pointer (use-after-free) -> CWE-416\n"
            "- Freeing same memory twice (double-free) -> CWE-415\n"
            "- Allocated memory never freed or returned (unreachable memory) -> CWE-401 (especially in loops)\n"
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
            "- Use MOST SPECIFIC CWE FOR ALL VULNERABILITIES. Example: 787 not 119, 125 not 119, 120 not 119, 401 not 400, etc.\n"
            "- NEVER output parent and child together (e.g., never [119, 787])\n"
            "- When in doubt -> FLAG with most specific CWE\n"
        )

        self.PESSIMISTIC_REMINDER: str = ""

        self.FINAL_ENFORCEMENT: str = (
            "## BEFORE YOU RESPOND\n\n"

            "1. Did you apply the Analysis Rules above?\n"
            "2. Did you check for LOCAL BUGS? (OOB, NULL deref, UAF, double-free, leaks)\n"
            "3. Is ANY operation unsafe? -> FLAG it\n"
            "4. Did you use the MOST SPECIFIC CWE for each detected vulnerability?\n\n"

            "'No external input' is NOT a defense.\n"
            "Do NOT second-guess. Do NOT dismiss bugs as 'not exploitable'.\n"
        )

        # ====================================================================
        # PROMPT TEMPLATES
        # ====================================================================
        self.SYSTEM_PROMPT: Template = Template(
            (
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

                # ==== FIELD RULES ====
                "## Field Rules\n"
                "- **reasoning**: Critic and objective security analysis. MUST include: (1) Local bugs found (or confirmed absent),"
                " (2) Data flow concerns, (3) Dangerous patterns, (4) Final assessment with CWE justification\n"
                "- **vulnerabilities**: Array of objects with cwe_id (int) and description (string).\n"
                "- **verdict**: Object with `is_vulnerable` (true/false) and `cwe_list`: list matching cwe_ids in vulnerabilities array\n"
                "- **If safe**: vulnerabilities=[], is_vulnerable=false, cwe_list=[]\n"
            )
        )

        self.USER_PROMPT: Template = Template(
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

                # ==== ANALYSIS RULES ====
                "{% if assumptions %}"
                "\n\n"
                "{{ assumptions }}"
                "{% endif %}"

                "\n\n"

                # === END: Back reference + checklist ===
                "{{ final_reinforcement }}"
            )
        )


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

        return self.USER_PROMPT.render(
            func_code=func_code,
            reminder=reminder,
            assumptions=assumptions,
            final_reinforcement=self.FINAL_ENFORCEMENT,
        )

    def enrich_reasoning_with_json(self, example: ReasoningSample) -> str:
        """
        Add JSON-like structure to create direct visual mapping.
        Teaches model to connect analysis to structured output.
        """
        base_reasoning = example.reasoning.strip()

        if example.target == 0:
            # Safe code
            assessment = {
                "vulnerabilities": [],
                "verdict": {"is_vulnerable": False, "cwe_list": []},
            }
        else:
            cwe_descs = example.cwe_desc

            # Build vulnerability entries
            vulnerabilities = []
            cwe_ids = []

            for i, cwe in enumerate(example.cwe):
                cwe_id = int(cwe.replace("CWE-", "").strip())
                cwe_ids.append(cwe_id)
                desc = cwe_descs[i] if i < len(cwe_descs) else f"CWE-{cwe_id}"
                vulnerabilities.append({"cwe_id": cwe_id, "description": desc})

            assessment = {
                "vulnerabilities": vulnerabilities,
                "verdict": {"is_vulnerable": True, "cwe_list": cwe_ids},
            }

        json_str = json.dumps(assessment, indent=None, ensure_ascii=False)
        return (base_reasoning + f"\n\n**Structured Assessment:**\n```json\n{json_str}\n```")

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

        return messages

    def format_sample(self, sample: ReasoningSample) -> Messages:
        enriched_reasoning = self.enrich_reasoning_with_json(example=sample)

        vulnerabilities: list[VulnInfo] = []

        if sample.target == 1 and sample.cwe:
            cwe_descs: list[str] = sample.cwe_desc
            for i, cwe in enumerate(sample.cwe):
                try:
                    cwe_id = int(cwe.replace("CWE-", "").strip())

                    description = (
                        cwe_descs[i]
                        if i < len(cwe_descs)
                        else f"CWE-{cwe_id} vulnerability"
                    )

                    # validate fields and then append
                    vuln_info = VulnInfo(cwe_id=cwe_id, description=description)
                    vulnerabilities.append(vuln_info)

                except ValueError:
                    continue

        # cwe_list from validated vulnerabilities (guaranteed match!)
        cwe_list: list[int] = [v.cwe_id for v in vulnerabilities]

        # build structured response
        response_data: ExpectedModelResponse = ExpectedModelResponse(
            reasoning=enriched_reasoning,
            vulnerabilities=vulnerabilities,
            verdict=VerdictStruct(
                is_vulnerable=bool(sample.target), cwe_list=cwe_list
            ),
        )

        # convert to formatted JSON
        messages = self.as_messages(
            func_code=sample.func,
            phase=self.prompt_phase,
            mode=self.assumption_mode,
            add_hierarchy=self.add_hierarchy,
            ground_truth=response_data.model_dump_json(indent=None, ensure_ascii=False),
        )

        return messages

    def count_individual_components(self, sample: ReasoningSample) -> TokensStats:
        messages = self.format_sample(sample)

        # System
        system_messages: dict[str, str] = messages[0]
        system_formatted = self.apply_template([system_messages])
        system_tokens = self._encode_and_count(system_formatted)

        # System + User (to get user contribution)
        system_user_messages = messages[:2] # [0, 1]
        system_user_formatted = self.apply_template(system_user_messages)
        system_user_tokens = self._encode_and_count(system_user_formatted)
        user_tokens = system_user_tokens - system_tokens

        # Full conversation (system + user + assistant)
        if len(messages) == 2:
            messages.append({"role": "assistant", "content": ""})

        full_convo = messages
        full_formatted = self.apply_template(full_convo)
        total_tokens = self._encode_and_count(full_formatted)

        # Assistant tokens = total - (system + user)
        assistant_tokens = total_tokens - system_user_tokens

        return TokensStats(
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            assistant_tokens=assistant_tokens,
            total_tokens=total_tokens,
        )

from dataclasses import dataclass, field
from transformers import BatchEncoding
from jinja2 import Template

from .datatypes import PromptPhase, AssumptionMode

Message = dict[str, str | list[int] | list[str] | list[list[int]] | BatchEncoding]
Messages = list[dict[str,str]]


@dataclass
class VulnerabilityPromptConfig: 
    SYSTEM_PROMPT: Template = field(
        init=False,
        default=Template(
            (
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

                "{% if assumptions %}"
                "{{ assumptions }}"
                "{% endif %}"

                # Response Format
                "## Output Format\n"
                "Provide your analysis in this EXACT structure:\n"

                "```json\n"
                '{"reasoning": "<string: your step-by-step security analysis>", "vulnerabilities": [{"cwe_id": <int>, "description": "<string>"}], "verdict": {"is_vulnerable": <boolean>, "cwe_list": [<int>]}}\n'
                "```\n\n"

                "Example output format:\n"
                "```json\n"
                '{"reasoning": "The function uses strcpy() to copy user input into a fixed-size buffer without bounds checking. This allows an attacker to overflow the buffer, potentially overwriting adjacent memory. This pattern matches CWE-119 (buffer overflow) and CWE-120 (classic buffer overflow).", "vulnerabilities": [{"cwe_id": 119, "description": "The function uses strcpy to copy an input string into a fixed-size buffer without checking if the input exceeds the available buffer size, leading to buffer overflow."}], "verdict": {"is_vulnerable": true, "cwe_list": [119]}}\n'
                "```\n\n"

                # Field Definitions
                "## Field Definitions\n"
                "- **reasoning**: 2 to 4 concise sentences explaining the security analysis. "
                "It covers what the code does, what makes it dangerous or safe, and which CWEs apply if vulnerable.\n"
                "- **vulnerabilities**: array of objects reporting all vulnerability information.\n"
                "- **verdict**: object with following properties:\n"
                "  - **is_vulnerable**: Boolean indicating whether the code is vulnerable (true/false)\n"
                "  - **cwe_list**: CWEs with clear, traceable evidence in the code (as integer list)\n\n"

                "## Critical Requirements\n"
                "- Output ONLY valid JSON in compact format (single line, no extra whitespace)\n"
                "- Do not include any text before or after the JSON\n"
                "- Keep reasoning concise: 2-4 sentences, maximum 100 words, as a single text block.\n"
                "- Include all 5 analysis steps in your reasoning (data flow, unknown calls, patterns, controls, classification)\n"
                "- Write in natural, narrative prose (no bullet points in reasoning)\n"
                "- The 'vulnerabilities' array should list each identified CWE with its description\n"
                "- The 'cwe_list' in verdict should mirror the CWE IDs from vulnerabilities array\n"
                "- CRITICAL: The vulnerabilities array and cwe_list MUST contain the same CWE IDs. "
                "If vulnerable, vulnerabilities array MUST NOT be empty.\n"
                "- If no vulnerabilities: set 'vulnerabilities' to [] and 'is_vulnerable' to false\n"
                "- CWE IDs must be integers without prefix (e.g., 119, not 'CWE-119')\n"
                "- Ensure proper JSON escaping"
                "- DO NOT use '>' or '<' to encapsulate field values"
            ).strip()
        ),
        repr=False,
    )

    USER_PROMPT: Template = field(
        init=False,
        default=Template(
            (
                "## Code to Analyze\n"
                "```c\n"
                "{{func_code}}\n"
                "```\n\n"

                "{% if attack_patterns %}"
                "{{ attack_patterns }}"
                "\n\n"
                "{% endif %}"

                "{% if reminder %}"
                "{{ reminder }}"
                "{% endif %}"
            )
        ),
        repr=False,
    )

    OPTIMISTIC_ASSUMPTIONS: str = field(
        init=False,
        default=(
            # Make vulnerabilities based only on clear evidence
            "CRITICAL ASSUMPTIONS (MUST FOLLOW):\n"
            "1. All unknown/external function calls are SAFE and behave correctly according to their implied contract\n"
            "2. Return values from unknown functions are VALID and PROPERLY BOUNDED\n"
            "3. Memory returned by unknown allocators is PROPERLY SIZED and INITIALIZED\n"
            "4. Strings returned by unknown functions are PROPERLY NULL-TERMINATED\n"
            "5. Length values returned alongside buffers ACCURATELY REFLECT the buffer size\n\n"

            "DO NOT flag vulnerabilities based on:\n"
            "- Speculation about how unknown functions MIGHT fail\n"
            "- Missing null-termination checks for strings from unknown APIs\n"
            "- Missing bounds checks when lengths are provided by unknown APIs\n"
            "- Theoretical edge cases that require unknown functions to violate their implied contract\n\n"

            "ONLY flag vulnerabilities where:\n"
            "- The vulnerable pattern is ENTIRELY within the visible code\n"
            "- The bug would occur EVEN IF all unknown functions behave correctly\n"
            "- There is CONCRETE evidence of misuse (e.g., wrong format specifier, obvious off-by-one in visible loop etc)\n"
            "- If context is explicitly provided and it indicates an API may be unsafe or unreliable, "
            "you MAY flag vulnerabilities that depend on that API's behavior.\n\n"

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
            "This is SAFE because we trust parse_input() returns valid bounds.\n\n"
        ),
        repr=False
    )

    OPTIMISTIC_REMINDER: str = field(
        init=False,
        default=(
            "Remember:\n"
            "During reasoning, bare in mind to ask yourself:\n"
            'If I told the maintainer about this bug, would they say: "that\'s impossible, our API guarantees X"? '
            "If yes → NOT A VULNERABILITY.\n"
            "Focus solely on clear and solid evidence."
        ),
        repr=False,
    )

    # Pessimistic assumptions
    PESSIMISTIC_ASSUMPTIONS: str = field(
        init=False,
        default=(
            "CRITICAL ASSUMPTIONS (MUST FOLLOW):\n"
            "1. All unknown/external function calls are POTENTIALLY UNSAFE and may violate their implied contract\n"
            "2. Return values from unknown functions may be INVALID, OUT-OF-BOUNDS, or MALICIOUS\n"
            "3. Memory returned by unknown allocators may be IMPROPERLY SIZED, UNINITIALIZED, or NULL\n"
            "4. Strings returned by unknown functions may NOT be NULL-TERMINATED\n"
            "5. Length values returned alongside buffers may be INCORRECT, INCONSISTENT, or ATTACKER-CONTROLLED\n\n"

            "FLAG vulnerabilities when:\n"
            "- Unknown functions are used WITHOUT explicit validation of their return values\n"
            "- Buffers, pointers, or lengths from external sources are used without bounds checking\n"
            "- Missing null-termination checks for strings from unknown APIs\n"
            "- Missing bounds checks when lengths are provided by unknown APIs\n"
            "- Any operation that could fail if an unknown function returns unexpected values\n\n"
            
            "ONLY consider code SAFE when:\n"
            "- There is EXPLICIT validation of all external inputs (null checks, bounds checks, size validation)\n"
            "- Buffer sizes are STATICALLY KNOWN and operations are PROVABLY bounded\n"
            "- Error handling is present for all potentially-failing operations\n"
            "- There is CONCRETE evidence that the code handles all edge cases\n\n"

            "EXAMPLES OF VULNERABILITIES (TO BE FLAG):\n"

            "Example 1 - Un-trusted API return values:\n"
            "```c\n"
            "char *buf = get_buffer(&len);  // Unknown API\n"
            "memcpy(dest, buf, len);        // Unsafe: risky memcpy without bound checking\n"
            "```\n"

            "Example 2 - Dangerous null-termination:\n"
            "```c\n"
            "char *str = get_string();      // Unknown API\n"
            "int n = strlen(str);           // UNSAFE: lack of null-termination checking\n"
            "```\n"

            "Example 3 - Insecure bounded values:\n"
            "```c\n"
            "parse_input(input, &offset, &length);  // Unknown API\n"
            "memcpy(dest, input + offset, length);  // UNSAFE: offset/length unknown\n"
            "```\n"
        ),
        repr=False
    )
    
    PESSIMISTIC_REMINDER: str = field(
        init=False,
        default=(
            "Remember:\n"
            "During reasoning, bare in mind to ask yourself:\n"
            '"If this unknown function behaved maliciously or returned an error, would this code be vulnerable?" '
            "If yes → FLAG AS VULNERABILITY.\n"
            "Assume unknown functions may fail or return unexpected values unless explicitly validated.\n"
            "When in doubt, flag potential vulnerabilities - it\'s better to be cautious."
        ),
        repr=False,
    )

    SEEN_ATTACK_PATTERNS: str = field(
        init=False,
        default=(
            "Look ONLY for the following vulnerabilities:\n"
            "   - CWE-787: The product writes data past the end, or before the beginning, of the intended buffer.\n"
            "   - CWE-119: The product performs operations on a memory buffer, but it reads from or writes to a memory location outside the buffer's intended boundary. This may result in read or write operations on unexpected memory locations that could be linked to other variables, data structures, or internal program data.\n"
            "   - CWE-125: The product reads data past the end, or before the beginning, of the intended buffer.\n"
            "   - CWE-20: The product receives input or data, but it does not validate or incorrectly validates that the input has the properties that are required to process the data safely and correctly.\n"
            "   - CWE-416: The product reuses or references memory after it has been freed. At some point afterward, the memory may be allocated again and saved in another pointer, while the original pointer references a location somewhere within the new allocation. Any operations using the original pointer are no longer valid because the memory belongs to the code that operates on the new pointer.\n"
            "   - CWE-190: The product performs a calculation that can produce an integer overflow or wraparound when the logic assumes that the resulting value will always be larger than the original value. This occurs when an integer value is incremented to a value that is too large to store in the associated representation. When this occurs, the value may become a very small or negative number.\n"
            "   - CWE-476: The product dereferences a pointer that it expects to be valid but is NULL.\n"
            "   - CWE-200: The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information.\n"
            "   - CWE-703: The product does not properly anticipate or handle exceptional conditions that rarely occur during normal operation of the product.\n"
            "   - CWE-362: The product contains a concurrent code sequence that requires temporary, exclusive access to a shared resource, but a timing window exists in which the shared resource can be modified by another code sequence operating concurrently.\n"
            "   - CWE-400: The product does not properly control the allocation and maintenance of a limited resource.\n"
            "   - CWE-120: The product copies an input buffer to an output buffer without verifying that the size of the input buffer is less than the size of the output buffer.\n"
            "   - CWE-401: The product does not sufficiently track and release allocated memory after it has been used, making the memory unavailable for reallocation and reuse.\n"
            "   - CWE-415: The product calls free() twice on the same memory address.\n"
            "   - CWE-284: The product does not restrict or incorrectly restricts access to a resource from an unauthorized actor.\n"
            "   - CWE-269: The product does not properly assign, modify, track, or check privileges for an actor, creating an unintended sphere of control for that actor."
        ),
        repr=False
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

    def format_system_prompt(self, assumptions: str|None) -> str:
        """Format system prompt with optional assumptions."""
        return self.SYSTEM_PROMPT.render(assumptions=assumptions)

    def format_user_prompt(
        self,
        func_code: str,
        phase: PromptPhase,
        reminder: str | None
    ) -> str:
        """Format user prompt with optional attack patterns."""

        attack_patterns = (
            self.SEEN_ATTACK_PATTERNS
            if phase
            in [
                PromptPhase.FULL_CONSTRAINED_INFERENCE,
                PromptPhase.ATTACK_CONSTRAINED_INFERENCE,
            ]
            else None
        )
        return self.USER_PROMPT.render(
            func_code=func_code, reminder=reminder, attack_patterns=attack_patterns
        )

    def as_messages(
        self,
        func_code: str,
        phase: PromptPhase,
        mode: AssumptionMode,
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

        Returns
        -------
        list[dict[str, str]]
            Chat messages in format [{"role": "system", "content": ...}, ...]
        """

        system_assumptions, user_reminder = self.get_assumption_reminder_combo(phase=phase, mode=mode)
        system_prompt = self.format_system_prompt(assumptions=system_assumptions)

        user_prompt = self.format_user_prompt(
            func_code=func_code,
            phase=phase,
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


if __name__ == "__main__":
    from ..utilities import rich_rule
    from pathlib import Path

    config = VulnerabilityPromptConfig()

    rich_rule(title="TRAINING MODE OPTIMISTIC")
    # Training mode (with assumptions, no attack patterns)
    training_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.CONSTRAINED_TRAINING,
        mode=AssumptionMode.OPTIMISTIC
    )

    parent_dir = Path(__file__).parent / "prompts"
    parent_dir.mkdir(parents=True, exist_ok=True)

    with open(
        parent_dir / "full_training_mode_optimistic_prompt.txt", "w"
    ) as text_file:
        text_file.write(50 * "=" + " System prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[2]["content"])

    print(50*"#" + " System prompt training " + 50 * "#" )
    print(training_messages[0]["content"])  # System prompt with assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + training_messages[1]["content"])  # User prompt without attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    print("\n" + training_messages[2]["content"])  # User prompt without attack patterns
    rich_rule()

    rich_rule(title="TRAINING MODE PESSIMISTIC")
    # Training mode (with assumptions, no attack patterns)
    training_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.CONSTRAINED_TRAINING,
        mode=AssumptionMode.PESSIMISTIC
    )
    with open(parent_dir / "full_training_mode_pessimistic_prompt.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[2]["content"])

    print(50*"#" + " System prompt training " + 50 * "#" )
    print(training_messages[0]["content"])  # System prompt with assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + training_messages[1]["content"])  # User prompt without attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    print("\n" + training_messages[2]["content"])  # User prompt without attack patterns
    rich_rule()

    rich_rule(title="TRAINING NO ASSUMPTIONS MODE")
    training_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.FREE_TRAINING,
        mode=AssumptionMode.NONE
    )
    with open(parent_dir / "training_mode_no_assumptions_prompt.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt training " + 50 * "=")
        text_file.write("\n")
        text_file.write(training_messages[2]["content"])

    print(50*"#" + " System prompt training " + 50 * "#" )
    print(training_messages[0]["content"])  # System prompt with assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + training_messages[1]["content"])  # User prompt without attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    print("\n" + training_messages[2]["content"])

    rich_rule()

    rich_rule(title="INFERENCE FULL MODE OPTIMISTIC")
    # Inference mode (no assumptions, with attack patterns)
    inference_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.FULL_CONSTRAINED_INFERENCE,
        mode=AssumptionMode.OPTIMISTIC
    )
    with open(parent_dir / "full_inference_mode_optimistic_prompt.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
        text_file.write("\n")
        try:
            text_file.write(inference_messages[2]["content"])
        except Exception as e:
            text_file.write("None")

    print(50 * "#" + " System prompt training " + 50 * "#")
    print(inference_messages[0]["content"])  # System prompt without assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + inference_messages[1]["content"])  # User prompt with attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    try:
        print("\n" + inference_messages[2]["content"])
    except Exception as e:
        print(e)

    rich_rule()

    rich_rule(title="INFERENCE FULL MODE PESSIMISTIC")
    # Inference mode (no assumptions, with attack patterns)
    inference_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.FULL_CONSTRAINED_INFERENCE,
        mode=AssumptionMode.PESSIMISTIC
    )
    with open(parent_dir / "full_inference_mode_pessimistic_prompt.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
        text_file.write("\n")
        try:
            text_file.write(inference_messages[2]["content"])
        except Exception as e:
            text_file.write("None")

    print(50 * "#" + " System prompt training " + 50 * "#")
    print(inference_messages[0]["content"])  # System prompt without assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + inference_messages[1]["content"])  # User prompt with attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    try:
        print("\n" + inference_messages[2]["content"])
    except Exception as e:
        print(e)

    rich_rule()
    rich_rule(title="INFERENCE ATTACKS ONLY MODE")
    inference_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.ATTACK_CONSTRAINED_INFERENCE,
        mode=AssumptionMode.NONE
    )
    with open(parent_dir / "inference_mode_attacks_only_prompt.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
        text_file.write("\n")
        try:
            text_file.write(inference_messages[2]["content"])
        except Exception as e:
            text_file.write("None")

    print(50*"#" + " System prompt training " + 50 * "#" )
    print(inference_messages[0]["content"])  # System prompt without assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + inference_messages[1]["content"])  # User prompt with attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    try:
        print("\n" + inference_messages[2]["content"])
    except Exception as e:
        print(e)

    rich_rule()
    rich_rule(title="INFERENCE BAREBONE MODE")
    inference_messages = config.as_messages(
        func_code="void unsafe() { char buf[10]; gets(buf); }",
        ground_truth='{"reasoning": "...", "vulnerabilities": [...], "verdict": {...}}',
        phase=PromptPhase.FREE_INFERENCE,
        mode=AssumptionMode.NONE,
    )
    with open(parent_dir / "inference_mode_barebone.txt", "w") as text_file:
        text_file.write(50 * "=" + " System prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[0]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " User prompt " + 50 * "=")
        text_file.write("\n")
        text_file.write(inference_messages[1]["content"])
        text_file.write("\n\n")
        text_file.write(50 * "=" + " Assistant prompt " + 50 * "=")
        text_file.write("\n")
        try:
            text_file.write(inference_messages[2]["content"])
        except Exception as e:
            text_file.write("None")

    print("\n\n" + 50 * "=" + " INFERENCE BAREBONE MODE " + 50 * "=")
    print(50*"#" + " System prompt training " + 50 * "#" )
    print(inference_messages[0]["content"])  # System prompt without assumptions
    print(50*"#" + " User prompt training " + 50 * "#" )
    print("\n" + inference_messages[1]["content"])  # User prompt with attack patterns
    print(50*"#" + " Assistant prompt training " + 50 * "#" )
    try:
        print("\n" + inference_messages[2]["content"])
    except Exception as e:
        print(e)

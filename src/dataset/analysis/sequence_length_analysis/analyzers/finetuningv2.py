"""
Sequence Length Analyzer for fine-tuning.
Analyzes token distribution in CoT dataset to determine optimal max_seq_length.

Features:
- Memory-efficient streaming (handles large JSONL files)
- Multiple tokenizer support
- Comprehensive statistics and visualizations
- Truncation impact analysis
"""

import json

from transformers import PreTrainedTokenizer

from .base import BaseSequenceLengthAnalyzer
from ..datatypes import ReasoningSample, TokensStats, VulnInfo, ResponseStruct


class FineTunePromptAnalyzerV2(BaseSequenceLengthAnalyzer):
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
    ).strip()

    USER_PROMPT: str = (
        "## Code to Analyze\n"
        "```c\n"
        "{func_code}\n"
        "```\n\n"
        "Remember:\n"
        "During reasoning, bare in mind to ask yourself:\n"
        'If I told the maintainer about this bug, would they say: "that\'s impossible, our API guarantees X"? '
        "If yes → NOT A VULNERABILITY.\n"
        "Focus solely on clear and solid evidence."
    ).strip()

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: str):
        super().__init__(tokenizer, chat_template)

    def format_sample(self, sample: ReasoningSample) -> tuple[str, str, str]:
        vulnerabilities: list[VulnInfo] = []
        cwe_list: list[int] = []

        if sample.target == 1 and sample.has_cwes:
            cwes = sample.cwe
            cwe_descs = sample.cwe_desc

            for i, cwe in enumerate(cwes):
                try:
                    cwe_id = int(cwe.replace("CWE-", "").strip())
                    cwe_list.append(cwe_id)

                    description = (
                        cwe_descs[i]
                        if i < len(cwe_descs)
                        else f"CWE-{cwe_id} vulnerability"
                    )

                    vulnerabilities.append(
                        {"cwe_id": cwe_id, "description": description}
                    )

                except ValueError:
                    print(f"Error parsing for CWE '{cwe}`")
                    continue

        # build structured response matching prompt schema
        response_data: ResponseStruct = {
            "reasoning": sample.reasoning.strip(),
            "vulnerabilities": vulnerabilities,
            "verdict": {"is_vulnerable": bool(sample.target), "cwe_list": cwe_list},
        }

        # convert to formatted JSON
        ground_truth: str = json.dumps(response_data, indent=2, ensure_ascii=False)

        return (
            self.SYSTEM_PROMPT,
            self.USER_PROMPT.format(func_code=sample.func),
            ground_truth,
        )

    def count_tokens_for_sample(self, sample: ReasoningSample) -> TokensStats:
        system_content, user_content, assistant_content = self.format_sample(sample)

        # Count individual components
        # System
        system_messages = [{"role": "system", "content": system_content}]
        system_formatted = self.tokenizer.apply_chat_template(
            system_messages, tokenize=False, add_generation_prompt=False
        )
        system_tokens = len(self.tokenizer.encode(system_formatted, add_special_tokens=False))  # type: ignore

        # System + User (to get user contribution)
        system_user_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        system_user_formatted = self.tokenizer.apply_chat_template(
            system_user_messages, tokenize=False, add_generation_prompt=False
        )
        system_user_tokens = len(self.tokenizer.encode(system_user_formatted, add_special_tokens=True))  # type: ignore
        user_tokens = system_user_tokens - system_tokens

        # Full conversation (system + user + assistant)
        full_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_formatted = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        total_tokens = len(self.tokenizer.encode(full_formatted, add_special_tokens=True))  # type: ignore

        # Assistant tokens = total - (system + user)
        assistant_tokens = total_tokens - system_user_tokens

        # Split assistant into reasoning and answer
        if sample.target == 1 and sample.cwe:
            cwe_string = ", ".join(sample.cwe)
            final_answer_str = f"\n\nFinal Answer: YES ({cwe_string})"
        else:
            final_answer_str = "\n\nFinal Answer: NO"

        # approximation: can't easily separate comps post-tokenization
        answer_tokens_approx = self._encode_and_count(final_answer_str)
        reasoning_tokens_approx = assistant_tokens - answer_tokens_approx

        total_approx = reasoning_tokens_approx + answer_tokens_approx
        if total_approx > 0:
            answer_tokens = int(
                assistant_tokens * (answer_tokens_approx / total_approx)
            )
            reasoning_tokens = assistant_tokens - answer_tokens
        else:
            reasoning_tokens = assistant_tokens
            answer_tokens = 0

        return TokensStats(
            system_tokens=system_tokens,
            user_tokens=user_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            assistant_tokens=assistant_tokens,
            total_tokens=total_tokens,
        )

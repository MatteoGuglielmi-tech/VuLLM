# Prompt Templates

This directory contains the complete prompt templates used for training and inference in the vulnerability detection experiments.

## Overview

The prompt system uses a modular architecture with fixed and conditional components, optimized for attention patterns (critical information at start and end positions):

| Component | Location | Position | Condition |
|-----------|----------|----------|-----------|
| Persona + Critical Output Constraint | System prompt | Start (high attention) | Always |
| CWE Mapping Guidance | System prompt | Middle | Optional |
| Analysis Steps | System prompt | Middle | Always |
| Output Format + Field Rules | System prompt | End | Always |
| Frame + Forward Reference | User prompt | Start (high attention) | Always |
| Code to Analyze | User prompt | Middle | Always |
| Assumptions | User prompt | Middle | Optional |
| Final Enforcement | User prompt | End (high attention) | Always |

## Experimental Configurations

We employ a 3×2 factorial design with six configurations:

| # | Configuration | Assumptions | CWE Guidance | Description |
|---|---------------|:-----------:|:------------:|-------------|
| 1 | Free | ✗ | ✗ | Minimal configuration; raw model capability |
| 2 | Free + CWE | ✗ | ✓ | CWE guidance only; tests taxonomy impact |
| 3 | Optimistic | ✓ | ✗ | Trust unknown APIs; flag only clear evidence |
| 4 | Optimistic + CWE | ✓ | ✓ | Optimistic + taxonomy guidance |
| 5 | Pessimistic | ✓ | ✗ | Distrust unknown APIs; flag potential issues |
| 6 | Pessimistic + CWE | ✓ | ✓ | Most comprehensive configuration |

All configurations include the Final Enforcement checklist. Components are consistent between training and inference (no inference-only additions).

## Prompt Files

### Training Prompts
Used during fine-tuning. The assistant response (ground truth) is appended during training.

| File | Configuration |
|------|---------------|
| `training_free.txt` | Config 1: No assumptions, no CWE guidance |
| `training_free_cwe.txt` | Config 2: No assumptions, with CWE guidance |
| `training_optimistic.txt` | Config 3: Optimistic assumptions, no CWE guidance |
| `training_optimistic_cwe.txt` | Config 4: Optimistic assumptions, with CWE guidance |
| `training_pessimistic.txt` | Config 5: Pessimistic assumptions, no CWE guidance |
| `training_pessimistic_cwe.txt` | Config 6: Pessimistic assumptions, with CWE guidance |

### Inference Prompts
Used during evaluation. Identical to training prompts (without assistant response).

| File | Configuration |
|------|---------------|
| `inference_free.txt` | Config 1 |
| `inference_free_cwe.txt` | Config 2 |
| `inference_optimistic.txt` | Config 3 |
| `inference_optimistic_cwe.txt` | Config 4 |
| `inference_pessimistic.txt` | Config 5 |
| `inference_pessimistic_cwe.txt` | Config 6 |

## Assumption Modes

### Optimistic Assumptions
The model is instructed to **trust** unknown/external APIs:
- Return values are valid and correctly bounded
- Memory is properly sized and initialized
- Strings are properly null-terminated
- Length values accurately reflect buffer sizes

Flag vulnerabilities **only when**:
- Vulnerability is entirely within visible code
- Bug would occur even if all unknown functions behave correctly
- Concrete evidence of misuse exists

**Use case:** Minimizing false positives; suitable when external APIs are well-tested and trusted.

### Pessimistic Assumptions
The model is instructed to **distrust** unknown/external APIs:
- Return values may be invalid, NULL, out-of-bounds, or malicious
- Memory may be improperly sized, uninitialized, or NULL
- Strings may not be null-terminated
- Analyze observable behavior only (no speculation about internals)

Flag vulnerabilities **without exception** for:
- Array/buffer access beyond declared size → CWE-787/125
- Dereferencing NULL or freed pointer → CWE-476/416
- Allocated memory never freed → CWE-401
- Integer overflow in size calculation → CWE-190
- Copy without size check → CWE-120
- Return value from unknown API used unchecked

**Use case:** Maximizing vulnerability coverage; suitable for security-critical applications where false negatives are costly.

## CWE Mapping Guidance

The CWE guidance component provides taxonomy preferences to improve classification accuracy:

**Mapping Rules:**
1. Use the MOST SPECIFIC CWE that fits
2. Report ROOT CAUSE, not consequence
3. Never output parent and child together (e.g., never [119, 787])

**Preferred CWEs:**
- Memory buffer: CWE-787 (write), CWE-125 (read), CWE-120 (copy) over CWE-119
- Memory lifecycle: CWE-416 (UAF), CWE-415 (double-free), CWE-401 (leak) over CWE-400
- Pointer/Numeric: CWE-476 (NULL deref), CWE-190 (overflow)

**Causal Chain Rule:** When A causes B, report A (e.g., integer overflow causes buffer overflow → report CWE-190)

## Final Enforcement

Always-present checklist at the end of user prompt (high attention position):

```
1. Did you apply the Analysis Rules above?
2. Did you check for LOCAL BUGS? (OOB, NULL deref, UAF, double-free, leaks)
3. Is ANY operation unsafe? -> FLAG it
4. Did you use the MOST SPECIFIC CWE for each detected vulnerability?

'No external input' is NOT a defense.
Do NOT second-guess. Do NOT dismiss bugs as 'not exploitable'.
```

## Prompt Generation

Prompts are dynamically generated using Jinja2 templates. See `src/core/cot_training/processing_lib/prompts/` for the implementation; there you can find different prompt versions. Refer to `v2`:

```python
# src/core/cot_training/processing_lib/prompts/factory.py
fact = VulnPromptFactory()

config = fact.create(
    version=args.prompt_version,
    prompt_phase=args.prompt_mode,
    assumptions_mode=args.assumption_mode,
    add_cwe_guidelines=args.add_hierarchy
)
training_messages = config.as_messages(
    func_code="...",
    ground_truth="...",
)
```

## License

See the repository root for license information.

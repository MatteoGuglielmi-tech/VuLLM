# Prompt Templates

This directory contains the complete prompt templates used for training and inference in the vulnerability detection experiments. These prompts accompany the thesis:

> **LLM Fine-tuning for C Code Vulnerability Detection**

## Overview

The prompt system uses a modular architecture with fixed and conditional components:

| Component | Location | Condition |
|-----------|----------|-----------|
| Persona + Task | System prompt | Always included |
| Analysis Framework | System prompt | Always included |
| Assumptions | System prompt | Only if `mode ≠ NONE` |
| Output Format + Requirements | System prompt | Always included |
| Code to Analyze | User prompt | Always included |
| Attack Patterns | User prompt | Inference only |
| Reminder | User prompt | Only if `mode ≠ NONE` |

## Training Prompts

Used during fine-tuning. The assistant response (ground truth) is appended during training.

| File | Configuration | Description |
|------|---------------|-------------|
| `training_no_assumptions.txt` | Baseline | No assumption guidance; model relies on pre-trained knowledge |
| `training_optimistic.txt` | Optimistic | Trust unknown APIs; flag only clear evidence |
| `training_pessimistic.txt` | Pessimistic | Distrust unknown APIs; flag potential issues |

## Inference Prompts

Used during evaluation. No assistant response is provided.

| File | Configuration | Assumptions | Reminder | Attack Patterns |
|------|---------------|:-----------:|:--------:|:---------------:|
| `inference_barebone.txt` | Barebone | ✗ | ✗ | ✗ |
| `inference_attacks_only.txt` | Attacks only | ✗ | ✗ | ✓ |
| `inference_optimistic.txt` | Full optimistic | ✓ | ✓ | ✓ |
| `inference_pessimistic.txt` | Full pessimistic | ✓ | ✓ | ✓ |

## Assumption Modes

### Optimistic Assumptions
The model is instructed to **trust** unknown/external APIs:
- Return values are valid and properly bounded
- Memory is properly sized and initialized
- Strings are properly null-terminated
- Length values accurately reflect buffer sizes

**Use case:** Minimizing false positives; suitable when external APIs are well-tested and trusted.

### Pessimistic Assumptions
The model is instructed to **distrust** unknown/external APIs:
- Return values may be invalid, out-of-bounds, or malicious
- Memory may be improperly sized, uninitialized, or null
- Strings may not be null-terminated
- Length values may be incorrect or attacker-controlled

**Use case:** Maximizing vulnerability coverage; suitable for security-critical applications where false negatives are costly.

## Attack Patterns

The attack patterns component (inference only) provides descriptions of the 16 CWE types present in the dataset. This guides the model toward the specific vulnerability classes under evaluation without biasing training.

**Included CWEs:** 119, 120, 125, 190, 200, 269, 284, 362, 400, 401, 415, 416, 476, 703, 787, 20

## Prompt Generation

Prompts are dynamically generated using Jinja2 templates. See `src/prompt_config.py` for the implementation:

```python
from prompt_config import VulnerabilityPromptConfig, PromptPhase, AssumptionMode

config = VulnerabilityPromptConfig()
messages = config.as_messages(
    func_code="void foo() { ... }",
    phase=PromptPhase.CONSTRAINED_TRAINING,
    mode=AssumptionMode.PESSIMISTIC,
    ground_truth='{"reasoning": "...", ...}'
)
```

## License

See the repository root for license information.

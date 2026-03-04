"""
CWE Mapping Guidance for Vulnerability Detection Prompt

Built from MITRE CWE documentation (View-1003: Weaknesses for Simplified Mapping)
Reference: https://cwe.mitre.org/

This module provides the CWE_MAPPING_GUIDANCE string to be included in the
system prompt for vulnerability detection models.
"""

# =============================================================================
# CWE MAPPING GUIDANCE - Based on MITRE CWE View-1003
# =============================================================================
CWE_MAPPING_GUIDANCE_VERBOSE: str = """## CWE Mapping Guidelines (Source: MITRE CWE)

### Abstraction Levels
CWE uses four abstraction levels for weaknesses:
- **Pillar**: Highest-level, most abstract. Represents broad themes. Should not be used for mapping unless no alternative exists.
- **Class**: Abstract, typically technology-independent. Describes issues in terms of behavior, property, or resource. Often DISCOURAGED for root cause mapping.
- **Base**: Abstract but with sufficient detail for detection and prevention. Describes issues in terms of behavior, property, technology, language, or resource. PREFERRED for root cause mapping.
- **Variant**: Most specific, linked to specific language or technology. PREFERRED for root cause mapping.

### Mapping Priority
When assigning a CWE, follow this priority:
1. Use the most specific ALLOWED CWE (Variant or Base) that fits
2. If no specific CWE fits, consider ALLOWED-WITH-REVIEW CWEs
3. Use DISCOURAGED CWEs (Class/Pillar) only as a last resort when no alternative exists

### Available CWEs by Mapping Status

**ALLOWED (Preferred for root cause mapping):**
- CWE-787 (Base): Out-of-bounds Write
- CWE-125 (Base): Out-of-bounds Read  
- CWE-190 (Base): Integer Overflow or Wraparound
- CWE-476 (Base): NULL Pointer Dereference
- CWE-416 (Variant): Use After Free
- CWE-415 (Variant): Double Free
- CWE-401 (Variant): Missing Release of Memory after Effective Lifetime

**ALLOWED-WITH-REVIEW (Use with careful consideration):**
- CWE-120 (Base): Buffer Copy without Checking Size of Input
  → Use ONLY for unbounded copy operations (strcpy, gets, sprintf without bounds)
  → For general out-of-bounds writes, prefer CWE-787
  → For out-of-bounds reads, use CWE-125
- CWE-362 (Class): Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)
  → Acceptable when no more specific CWE fits

**ALLOWED (in this context - alternatives not in label set):**
- CWE-200 (Class): Exposure of Sensitive Information to an Unauthorized Actor
  → Note: MITRE discourages as it describes impact, not root cause
  → Accepted here because specific alternatives (CWE-203, CWE-209, CWE-532) are not available
- CWE-703 (Pillar): Improper Check or Handling of Exceptional Conditions
  → Note: MITRE discourages as it is extremely high-level
  → Accepted here because specific alternatives (CWE-248, CWE-391, CWE-392) are not available

**DISCOURAGED (Use only when no ALLOWED alternative fits):**
- CWE-119 (Class): Improper Restriction of Operations within the Bounds of a Memory Buffer
  → INSTEAD USE: CWE-787 (writes), CWE-125 (reads), or CWE-120 (unbounded copy)
- CWE-20 (Class): Improper Input Validation
  → Too abstract; specific alternatives not in label set
  → Use only when the issue is purely about validation, not its consequences
- CWE-400 (Class): Uncontrolled Resource Consumption
  → For memory leaks specifically, use CWE-401 instead
  → For other resource exhaustion, CWE-400 is acceptable as fallback

### Causal Relationships
Some vulnerabilities can cause or follow others. Report the ROOT CAUSE, not the consequence:

- CWE-190 (Integer Overflow) → can lead to → CWE-119/787/125 (Buffer Errors)
  → If integer overflow causes buffer overflow, report CWE-190
- CWE-20 (Input Validation) → can lead to → CWE-119/190 (Buffer/Integer Errors)
  → If missing validation causes the issue, consider CWE-20
- CWE-362 (Race Condition) → can lead to → CWE-416 (Use After Free), CWE-476 (NULL Pointer Dereference)
  → If race condition causes UAF/NPD, report CWE-362 as root cause

### CWE Relationships Summary

Memory Buffer Errors:
  CWE-119 (Class, DISCOURAGED)
    ├── CWE-787 (Base, ALLOWED) - Out-of-bounds Write
    ├── CWE-125 (Base, ALLOWED) - Out-of-bounds Read
    └── CWE-120 (Base, ALLOWED-WITH-REVIEW) - Buffer Copy without Size Check

Memory Lifecycle:
  CWE-416 (Variant, ALLOWED) - Use After Free ←┐
  CWE-415 (Variant, ALLOWED) - Double Free     ├── Peers (same parent CWE-672)
  CWE-401 (Variant, ALLOWED) - Memory Leak     │
                                               │
  CWE-362 (Class, ALLOWED-WITH-REVIEW) ────────┘ can precede CWE-416, CWE-476

Integer/Numeric:
  CWE-190 (Base, ALLOWED) - Integer Overflow ──── can precede CWE-119 and children

Pointer:
  CWE-476 (Base, ALLOWED) - NULL Pointer Dereference
""".strip()

CWE_MAPPING_GUIDANCE: str = """## CWE Mapping Guidelines (Source: MITRE CWE)

### Abstraction Levels
- **Pillar**: Highest-level, most abstract. Should not be used for mapping unless no alternative exists.
- **Class**: Abstract, technology-independent. Often DISCOURAGED for root cause mapping.
- **Base**: Sufficient detail for detection/prevention. PREFERRED for root cause mapping.
- **Variant**: Most specific, language/technology-linked. PREFERRED for root cause mapping.

### Mapping Priority
1. Use the most specific ALLOWED CWE (Variant or Base) that fits
2. If no specific CWE fits, consider ALLOWED-WITH-REVIEW CWEs
3. Use DISCOURAGED CWEs (Class/Pillar) only as last resort when no alternative exists

### Available CWEs by Mapping Status

**ALLOWED (Preferred for root cause mapping):**
- CWE-787 (Base): Out-of-bounds Write
- CWE-125 (Base): Out-of-bounds Read
- CWE-190 (Base): Integer Overflow or Wraparound
- CWE-476 (Base): NULL Pointer Dereference
- CWE-416 (Variant): Use After Free
- CWE-415 (Variant): Double Free
- CWE-401 (Variant): Missing Release of Memory after Effective Lifetime

**ALLOWED-WITH-REVIEW (Use with careful consideration):**
- CWE-120 (Base): Buffer Copy without Checking Size of Input
  -> ONLY for unbounded copy operations (strcpy, gets, sprintf without bounds)
  -> For general out-of-bounds writes, prefer CWE-787
  -> For out-of-bounds reads, use CWE-125
- CWE-362 (Class): Race Condition
  -> Acceptable when no more specific CWE fits

**ALLOWED (in this context - alternatives not in label set):**
- CWE-200 (Class): Exposure of Sensitive Information
  -> MITRE discourages (describes impact, not root cause); accepted here as alternatives unavailable
- CWE-703 (Pillar): Improper Check or Handling of Exceptional Conditions
  -> MITRE discourages (extremely high-level); accepted here as alternatives unavailable

**DISCOURAGED (Use only when no ALLOWED alternative fits):**
- CWE-119 (Class): Improper Restriction of Buffer Operations
  -> INSTEAD USE: CWE-787 (writes), CWE-125 (reads), or CWE-120 (unbounded copy)
- CWE-20 (Class): Improper Input Validation
  -> Too abstract; use only when the issue is purely about validation, not its consequences
- CWE-400 (Class): Uncontrolled Resource Consumption
  -> For memory leaks, use CWE-401 instead

### Causal Relationships
Report the ROOT CAUSE, not the consequence:
- CWE-190 (Integer Overflow) -> CWE-119/787/125: report CWE-190
- CWE-20 (Input Validation) -> CWE-119/190: consider CWE-20 if validation is root cause
- CWE-362 (Race Condition) -> CWE-416/476: report CWE-362

### CWE Hierarchy

Memory Buffer Errors:
  CWE-119 (Class, DISCOURAGED)
    |-- CWE-787 (Base, ALLOWED) - Out-of-bounds Write
    |-- CWE-125 (Base, ALLOWED) - Out-of-bounds Read
    +-- CWE-120 (Base, ALLOWED-WITH-REVIEW) - Buffer Copy without Size Check

Memory Lifecycle:
  CWE-416 (Variant, ALLOWED) - Use After Free
  CWE-415 (Variant, ALLOWED) - Double Free
  CWE-401 (Variant, ALLOWED) - Memory Leak
  CWE-362 (Class, ALLOWED-WITH-REVIEW) -- can precede --> CWE-416, CWE-476

Integer/Numeric:
  CWE-190 (Base, ALLOWED) - Integer Overflow -- can precede --> CWE-119 and children

Pointer:
  CWE-476 (Base, ALLOWED) - NULL Pointer Dereference"""

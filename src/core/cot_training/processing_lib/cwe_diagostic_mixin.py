from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any
from enum import Enum

from ..utilities import validate_filepath_extension, dump_yaml
from .datatypes import ExpectedModelResponse, GenerationError

CWE_HIERARCHY: dict[int, set[int]] = {
    119: {787, 125, 120},  # Buffer errors
    400: {401},  # Resource consumption
    672: {416, 415},  # Resource lifecycle (UAF, double-free)
}


def is_hierarchically_acceptable(
    pred_cwes: set[int],
    gt_cwes: set[int],
    hierarchy: dict[int, set[int]] = CWE_HIERARCHY,
) -> bool:
    """
    Check if prediction is acceptable under hierarchical rules:
    1. Exact match
    2. Pred has child when GT has parent (more specific)
    3. Pred has parent when GT has child (same category)
    4. Pred has parent + child together (redundant but correct)
    """
    if pred_cwes == gt_cwes:
        return True
    if not pred_cwes and not gt_cwes:
        return True
    if not pred_cwes or not gt_cwes:
        return False

    # Build reverse mapping: child -> parent
    child_to_parent: dict[int, int] = {}
    for parent, children in hierarchy.items():
        for child in children:
            child_to_parent[child] = parent

    for gt_cwe in gt_cwes:
        is_covered = False

        if gt_cwe in pred_cwes: # direct match ("EXACT")
            is_covered = True
        elif gt_cwe in hierarchy and (pred_cwes & hierarchy[gt_cwe]): # if gt parent key, check match in children ("ACCEPT")
            is_covered = True
        elif gt_cwe in child_to_parent and child_to_parent[gt_cwe] in pred_cwes: # if gt is child, check parent in predicted ("OVER GEN")
            is_covered = True
        if not is_covered: # prediction completely miss CWE
            return False

    return True


class AmbiguityLevel(Enum):
    UNAMBIGUOUS = "unambiguous"
    LOW_AMBIGUITY = "low_ambiguity"
    MODERATE = "moderate"
    HIGH_AMBIGUITY = "high_ambiguity"


@dataclass
class DiagnosticCase:
    """A single diagnostic test case with neutral naming."""

    id: str
    code: str
    expected_cwes: list[int]
    acceptable_cwes: list[int]
    ambiguity: AmbiguityLevel
    category: str
    # Internal notes - NOT exposed to model
    _notes: str = field(default="", repr=False)


@dataclass
class DiagnosticResult:
    """Result from running a single diagnostic case."""

    case_id: str
    func: str
    expected_cwes: list[int]
    acceptable_cwes: list[int]
    predicted_cwes: list[int]
    predicted_is_vulnerable: bool
    model_reasoning: str
    raw_response: str
    error: str | None = None
    is_hierarchically_acceptable: bool = False

    def __post_init__(self):
        self.is_hierarchically_acceptable = is_hierarchically_acceptable(
            set(self.predicted_cwes), set(self.expected_cwes)
        )

    @property
    def is_exact_match(self) -> bool:
        return bool(set(self.predicted_cwes) & set(self.expected_cwes))

    @property
    def is_acceptable(self) -> bool:
        return bool(
            set(self.predicted_cwes)
            & (set(self.expected_cwes) | set(self.acceptable_cwes))
        )

    @property
    def is_over_generalized(self) -> bool:
        """Predicted parent CWE when child was expected."""
        if self.is_exact_match:
            return False
        pred_set = set(self.predicted_cwes)
        # Check if predicted generic CWEs when specific were expected
        generic_cwes = {119, 20, 400, 703, 200, 664, 682}
        return bool(pred_set & generic_cwes) and bool(
            pred_set & set(self.acceptable_cwes)
        )

    @property
    def is_false_negative(self) -> bool:
        """Missed a vulnerability entirely."""
        return len(self.expected_cwes) > 0 and len(self.predicted_cwes) == 0

    @property
    def is_false_positive(self) -> bool:
        """Flagged safe code as vulnerable."""
        return len(self.expected_cwes) == 0 and len(self.predicted_cwes) > 0


@dataclass
class DiagnosticReport:
    """Complete diagnostic report with aggregated metrics."""

    timestamp: str
    model_info: dict
    results: list[DiagnosticResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def exact_matches(self) -> int:
        return sum(1 for r in self.results if r.is_exact_match)

    @property
    def acceptable_matches(self) -> int:
        return sum(1 for r in self.results if r.is_acceptable)

    @property
    def hierarchy_matches(self) -> int:
        return sum(1 for r in self.results if r.is_hierarchically_acceptable)

    @property
    def over_generalizations(self) -> int:
        return sum(1 for r in self.results if r.is_over_generalized)

    @property
    def false_negatives(self) -> int:
        return sum(1 for r in self.results if r.is_false_negative)

    @property
    def false_positives(self) -> int:
        return sum(1 for r in self.results if r.is_false_positive)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "model_info": self.model_info,
            "summary": {
                "total": self.total,
                "exact_matches": self.exact_matches,
                "acceptable_matches": self.acceptable_matches,
                "over_generalizations": self.over_generalizations,
                "false_negatives": self.false_negatives,
                "false_positives": self.false_positives,
                "errors": self.errors,
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "expected_cwes": r.expected_cwes,
                    "acceptable_cwes": r.acceptable_cwes,
                    "predicted_cwes": r.predicted_cwes,
                    "predicted_is_vulnerable": r.predicted_is_vulnerable,
                    "model_reasoning": r.model_reasoning,
                    "is_exact_match": r.is_exact_match,
                    "is_acceptable": r.is_acceptable,
                    "is_over_generalized": r.is_over_generalized,
                    "is_false_negative": r.is_false_negative,
                    "is_false_positive": r.is_false_positive,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ============================================================================
# DIAGNOSTIC TEST CASES
# ============================================================================

DIAGNOSTIC_CASES: list[DiagnosticCase] = [
    # ------------------------------------------------------------------------
    # Category: mem_write (Expected: CWE-787)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_a01",
        code="""
void process_array_a(void) {
    char data[10];
    data[100] = 'A';
}
""",
        expected_cwes=[787],
        acceptable_cwes=[119],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="mem_write",
        _notes="Direct indexed write beyond bounds",
    ),
    DiagnosticCase(
        id="case_a02",
        code="""
void process_loop_a(void) {
    int data[5];
    for (int i = 0; i <= 10; i++) {
        data[i] = i * 2;
    }
}
""",
        expected_cwes=[787],
        acceptable_cwes=[119, 121],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="mem_write",
        _notes="Loop writes beyond array bounds",
    ),
    DiagnosticCase(
        id="case_a03",
        code="""
void copy_data_a(void) {
    char dst[16];
    char src[64] = "Hello world, this is your captain speaking. I hope you are doing well";
    memcpy(dst, src, 64);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="mem_write",
        _notes="memcpy with explicit size mismatch",
    ),
    # ------------------------------------------------------------------------
    # Category: mem_read (Expected: CWE-125)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_b01",
        code="""
int read_array_b(void) {
    int data[5] = {1, 2, 3, 4, 5};
    return data[10];
}
""",
        expected_cwes=[125],
        acceptable_cwes=[119],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="mem_read",
        _notes="Direct indexed read beyond bounds",
    ),
    DiagnosticCase(
        id="case_b02",
        code="""
int sum_array_b(void) {
    char data[8] = "ABCDEFG";
    int result = 0;
    for (int i = 0; i < 20; i++) {
        result += data[i];
    }
    return result;
}
""",
        expected_cwes=[125],
        acceptable_cwes=[119],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="mem_read",
        _notes="Loop reads beyond array bounds",
    ),
    # ------------------------------------------------------------------------
    # Category: unbounded_copy (Expected: CWE-120)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_c01",
        code="""
void handle_input_c(char *input) {
    char buffer[64];
    strcpy(buffer, input);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="unbounded_copy",
        _notes="strcpy without length check",
    ),
    DiagnosticCase(
        id="case_c02",
        code="""
void read_line_c(void) {
    char buffer[100];
    gets(buffer);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="unbounded_copy",
        _notes="gets() is inherently unbounded",
    ),
    DiagnosticCase(
        id="case_c03",
        code="""
void format_message_c(char *name) {
    char output[50];
    sprintf(output, "Hello, %s! Welcome to the system.", name);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="unbounded_copy",
        _notes="sprintf with unbounded %s",
    ),
    # ------------------------------------------------------------------------
    # Category: integer_arith (Expected: CWE-190)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_d01",
        code="""
void* allocate_items_d(size_t count) {
    size_t total = count * sizeof(int);
    void *ptr = malloc(total);
    return ptr;
}
""",
        expected_cwes=[190],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="integer_arith",
        _notes="Multiplication overflow in allocation size",
    ),
    DiagnosticCase(
        id="case_d02_v2",
        code="""
void copy_data(char *dst, char *src, int len) {
    int total = len + 100;
    char *buf = malloc(total);
    memcpy(buf, src, len);
}
""",
        expected_cwes=[475],
        acceptable_cwes=[703],
        ambiguity=AmbiguityLevel.LOW_AMBIGUITY,
        category="integer_arith",
        _notes="malloc can fail on huge size (from negative ints as well). Missing NULL check → NULL deref. CWE-703 valid but less specific.",
    ),
    DiagnosticCase(
        id="case_d03",
        code="""
int compute_sum_d(int a, int b) {
    int result = a + b;
    return result;
}
""",
        expected_cwes=[],  # Basic arithmetic isn't a vulnerability
        acceptable_cwes=[190],
        ambiguity=AmbiguityLevel.HIGH_AMBIGUITY,
        category="integer_arith",
        _notes="Signed integer overflow (UB in C)",
    ),
    # ------------------------------------------------------------------------
    # Category: ptr_lifetime (Expected: CWE-416, 415, 401)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_e01",
        code="""
void process_data_e1(void) {
    char *ptr = malloc(100);
    free(ptr);
    ptr[0] = 'A';
}
""",
        expected_cwes=[416],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="ptr_lifetime",
        _notes="Write after free",
    ),
    DiagnosticCase(
        id="case_e02",
        code="""
int get_value_e2(void) {
    int *data = malloc(sizeof(int));
    *data = 42;
    free(data);
    return *data;
}
""",
        expected_cwes=[416],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="ptr_lifetime",
        _notes="Read after free",
    ),
    DiagnosticCase(
        id="case_e03",
        code="""
void cleanup_e3(void) {
    char *ptr = malloc(64);
    free(ptr);
    free(ptr);
}
""",
        expected_cwes=[415],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="ptr_lifetime",
        _notes="Double free",
    ),
    DiagnosticCase(
        id="case_e04",
        code="""
void process_e4(void) {
    char *data = malloc(1024);
    return;
}
""",
        expected_cwes=[],  # Leak alone isn't a vulnerability
        acceptable_cwes=[401],
        ambiguity=AmbiguityLevel.HIGH_AMBIGUITY,
        category="ptr_lifetime",
        _notes="Memory leak is a bug but rarely a security vulnerability without context",
    ),
    DiagnosticCase(
        id="case_e04_v2",
        code="""
void handle_request(char *input) {
    while (1) {
        char *data = malloc(strlen(input) + 1);
        strcpy(data, input);
        process(data);
    }
}
""",
        expected_cwes=[401],
        acceptable_cwes=[476, 703],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="ptr_lifetime",
        _notes="Memory leak in infinite loop. Model should NOT speculate about process() freeing memory. CWE-476 (missing NULL check) also acceptable.",
    ),
    DiagnosticCase(
        id="case_e04_v3",
        code="""
void handle_request(char *input) {
    while (1) {
        char *data = malloc(strlen(input) + 1);
        strcpy(data, input);
        process(data);
    }
}
""",
        expected_cwes=[401],
        acceptable_cwes=[476],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="ptr_lifetime",
        _notes="malloc in infinite loop, no free visible. Pessimistic: do NOT assume process() frees it. CWE-476 also acceptable (no NULL check).",

    ),
    # ------------------------------------------------------------------------
    # Category: null_access (Expected: CWE-476)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_f01",
        code="""
void init_large_buffer_f(void) {
    char *ptr = malloc(1000000000000ULL);
    ptr[0] = 'x';
}
""",
        expected_cwes=[476],
        acceptable_cwes=[400, 703],  # Both are valid secondary findings
        ambiguity=AmbiguityLevel.LOW_AMBIGUITY,
        category="null_access",
        _notes="Primary: NULL deref if malloc fails (476). Secondary: missing error handling (703)",
    ),
    DiagnosticCase(
        id="case_f02",
        code="""
int read_sensor_data(sensor_t *sensor) {
    int *data_ptr = NULL;
    if (sensor->is_active) {
        data_ptr = sensor->reading;
    }
    return *data_ptr;
}
    """,
        expected_cwes=[476],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="null_access",
        _notes="NULL deref in realistic context - should not be dismissed",
    ),
    # ------------------------------------------------------------------------
    # Category: control_benign (Expected: NO vulnerabilities)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_g01",
        code="""
void copy_string_g(const char *src) {
    char dst[64];
    strncpy(dst, src, sizeof(dst) - 1);
    dst[sizeof(dst) - 1] = '\\0';
}
""",
        expected_cwes=[],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="control_benign",
        _notes="Properly bounded copy",
    ),
    DiagnosticCase(
        id="case_g02",
        code="""
int access_array_g(int *arr, size_t arr_len, size_t index) {
    if (index >= arr_len) {
        return -1;
    }
    return arr[index];
}
""",
        expected_cwes=[],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="control_benign",
        _notes="Bounds check before access",
    ),
    DiagnosticCase(
        id="case_g03",
        code="""
char* create_buffer_g(size_t size) {
    char *ptr = malloc(size);
    if (ptr == NULL) { 
        return NULL;
    }
    memset(ptr, 0, size);
    return ptr;
}
""",
        expected_cwes=[],
        acceptable_cwes=[],
        ambiguity=AmbiguityLevel.UNAMBIGUOUS,
        category="control_benign",
        _notes="NULL check after malloc",
    ),
    # ------------------------------------------------------------------------
    # Category: ambiguous_context (For baseline comparison)
    # ------------------------------------------------------------------------
    DiagnosticCase(
        id="case_h01",
        code="""
void transfer_data_h(char *dst, char *src, size_t len) {
    memcpy(dst, src, len);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.MODERATE,
        category="ambiguous_context",
        _notes="Depends on caller; write direction is clear",
    ),
    DiagnosticCase(
        id="case_h02",
        code="""
void handle_external_h(void) {
    size_t len;
    char *data = get_data(&len);
    char buffer[256];
    memcpy(buffer, data, len);
}
""",
        expected_cwes=[120],
        acceptable_cwes=[119, 787],
        ambiguity=AmbiguityLevel.MODERATE,
        category="ambiguous_context",
        _notes="Unknown API; pessimistic should flag",
    ),
]


# ============================================================================
# PROTOCOL FOR TYPE SAFETY
# ============================================================================

from typing import Protocol, runtime_checkable, Callable


@runtime_checkable
class InferenceProvider(Protocol):
    """Protocol defining the required inference interface."""

    def run_inference(self, input_code: str, **kwargs) -> str:
        """
        Run inference on code and return JSON response string.

        Args:
            input_code: The C source code to analyze
            **kwargs: Additional arguments (e.g., n_retries)

        Returns:
            JSON string with structure:
            {
                "reasoning": str,
                "vulnerabilities": [...],
                "verdict": {"is_vulnerable": bool, "cwe_list": [int, ...]}
            }
        """
        ...


# ============================================================================
# MIXIN CLASS FOR TestHandlerPlain
# ============================================================================


class CWEDiagnosticMixin:
    """
    Mixin class providing CWE diagnostic capabilities.

    Requirements:
        The class using this mixin MUST implement `run_inference(input_code: str) -> str`

    Usage:
        class TestHandlerPlain(CWEDiagnosticMixin):
            def run_inference(self, input_code: str, **kwargs) -> str:
                # your implementation
                ...

        handler = TestHandlerPlain(...)
        report = handler.run_cwe_diagnostic()
    """

    run_inference: Callable[..., ExpectedModelResponse]

    def run_cwe_diagnostic(
        self,
        cases: list[DiagnosticCase] | None = None,
        verbose: bool = True,
        categories: list[str] | None = None,
    ) -> DiagnosticReport:
        """
        Run CWE diagnostic tests on unambiguous cases.

        Args:
            cases: Custom test cases (uses built-in if None)
            verbose: Print progress during execution
            categories: Filter to specific categories (runs all if None)

        Returns:
            DiagnosticReport with all results and metrics

        Raises:
            NotImplementedError: If run_inference is not implemented
        """
        self._verify_inference_method()

        if cases is None:
            cases = DIAGNOSTIC_CASES

        if categories is not None:  # allows to conduce analysis only on target CWE IDs
            cases = [c for c in cases if c.category in categories]

        results: list[DiagnosticResult] = []

        for i, case in enumerate(cases):
            if verbose:
                print(f"[{i+1}/{len(cases)}] Running {case.id}...", end=" ", flush=True)

            result = self._run_single_diagnostic(case)
            results.append(result)

            if verbose:
                status = self._get_status_symbol(result)
                print(f"{status} pred={result.predicted_cwes}")

        report = DiagnosticReport(
            timestamp=datetime.now().isoformat(),
            model_info=self._get_model_info(),
            results=results,
        )

        if verbose:
            self._print_diagnostic_summary(report, cases)

        return report

    def _verify_inference_method(self) -> None:
        """Verify that run_inference method is available."""
        if not hasattr(self, "run_inference") or not callable(
            getattr(self, "run_inference")
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement 'run_inference(input_code: str) -> str' "
                "method to use CWEDiagnosticMixin.\n"
                "Expected signature: def run_inference(self, input_code: str, **kwargs) -> str"
            )

    def _run_single_diagnostic(self, case: DiagnosticCase) -> DiagnosticResult:
        """Run inference on a single diagnostic case."""

        try:
            raw_response: ExpectedModelResponse = self.run_inference(
                input_code=case.code
            )

            return DiagnosticResult(
                case_id=case.id,
                func=case.code,
                expected_cwes=case.expected_cwes,
                acceptable_cwes=case.acceptable_cwes,
                predicted_cwes=raw_response.cwe_list,
                predicted_is_vulnerable=raw_response.is_vulnerable,
                model_reasoning=raw_response.reasoning,
                raw_response=raw_response.model_dump_json(indent=2),
            )

        except (GenerationError, RuntimeError) as e:
            return DiagnosticResult(
                case_id=case.id,
                func=case.code,
                expected_cwes=case.expected_cwes,
                acceptable_cwes=case.acceptable_cwes,
                predicted_cwes=[],
                predicted_is_vulnerable=False,
                model_reasoning="",
                raw_response="",
                error=f"Generation failed: {e}",
            )

    def _get_model_info(self) -> dict:
        """Extract model configuration info. Override in subclass for custom info."""
        info: dict[str, Any] = {}

        for attr in [
            "model_name",
            "model_path",
            "lora_path",
            "prompt_mode",
            "assumption_mode",
            "max_new_tokens",
            "max_seq_length",
            "max_length",
            "chat_template",
        ]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                info[attr] = str(val) if val is not None else None

        model = getattr(self, "model", None)
        if model is not None:
            config = getattr(model, "config", None)
            if config is not None:
                info["model_type"] = getattr(config, "_name_or_path", None)
                info["model_class"] = config.__class__.__name__

        return info

    @staticmethod
    def _get_status_symbol(result: DiagnosticResult) -> str:
        """Get a status symbol for a result."""
        if result.error:
            return "❌ ERROR"

        expected_benign = len(result.expected_cwes) == 0
        predicted_benign = len(result.predicted_cwes) == 0

        if expected_benign and predicted_benign:
            return "✅ CORRECT"
        elif expected_benign and not predicted_benign:
            return "⚠️  FP"  # False positive
        elif not expected_benign and predicted_benign:
            return "❌ FN"  # False negative
        elif result.is_exact_match:
            return "✅ EXACT"
        elif result.is_over_generalized:
            return "🔶 OVER-GEN"
        elif result.is_acceptable:
            return "✅ ACCEPT"
        else:
            return "❌ WRONG"

    @staticmethod
    def _print_diagnostic_summary(
        report: DiagnosticReport, cases: list[DiagnosticCase]
    ) -> None:
        """Print a summary of diagnostic results."""
        total = report.total

        print("=" * 70)
        print("CWE DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"Total cases:              {total}")
        print(
            f"Exact matches:            {report.exact_matches:3d} ({100*report.exact_matches/total:5.1f}%)"
        )
        print(
            f"Acceptable:           {report.acceptable_matches:3d} ({100*report.acceptable_matches/total:5.1f}%)"
        )
        print(
            f"Hierarchical acceptable:  {report.hierarchy_matches:3d} ({100*report.hierarchy_matches/total:5.1f}%)"
        )
        print(
            f"Over-generalized:     {report.over_generalizations:3d} ({100*report.over_generalizations/total:5.1f}%)"
        )
        print(
            f"False negatives:      {report.false_negatives:3d} ({100*report.false_negatives/total:5.1f}%)"
        )
        print(
            f"False positives:      {report.false_positives:3d} ({100*report.false_positives/total:5.1f}%)"
        )
        print(f"Errors:               {report.errors:3d}")

        # Per-category breakdown
        print("\nPer-category breakdown:")
        print(
            f"  {'Category':<20} {'Total':>5} {'Exact':>6} {'Accept':>7} {'OverGen':>8} {'FN':>4} {'FP':>4}"
        )
        print("  " + "-" * 55)

        # Map case_id to category
        id_to_category = {c.id: c.category for c in cases}

        from collections import defaultdict

        cat_stats = defaultdict(
            lambda: {
                "total": 0,
                "exact": 0,
                "accept": 0,
                "overgen": 0,
                "fn": 0,
                "fp": 0,
            }
        )

        for r in report.results:
            cat = id_to_category.get(r.case_id, "unknown")
            cat_stats[cat]["total"] += 1
            if r.is_exact_match:
                cat_stats[cat]["exact"] += 1
            if r.is_acceptable:
                cat_stats[cat]["accept"] += 1
            if r.is_over_generalized:
                cat_stats[cat]["overgen"] += 1
            if r.is_false_negative:
                cat_stats[cat]["fn"] += 1
            if r.is_false_positive:
                cat_stats[cat]["fp"] += 1

        for cat in sorted(cat_stats.keys()):
            s = cat_stats[cat]
            print(
                f"  {cat:<20} {s['total']:>5} {s['exact']:>6} {s['accept']:>7} {s['overgen']:>8} {s['fn']:>4} {s['fp']:>4}"
            )

        # Key insights
        print("\n" + "-" * 70)
        og_rate = report.over_generalizations / total if total > 0 else 0
        if og_rate > 0.2:
            print(
                "⚠️  HIGH OVER-GENERALIZATION: Model prefers generic CWEs (e.g., 119 over 787)"
            )
        elif og_rate > 0.1:
            print("🔶 MODERATE OVER-GENERALIZATION detected")
        else:
            print("✅ Over-generalization rate is acceptable")

        if report.false_positives > 0:
            print(f"⚠️  FALSE POSITIVES on benign code: {report.false_positives} cases")

        if report.false_negatives > total * 0.2:
            print(
                f"⚠️  HIGH FALSE NEGATIVE RATE: Model missed {report.false_negatives} issues"
            )

        print("=" * 70)

    @validate_filepath_extension(arg_name="filepath", allowed_suffixes=[".yaml"])
    def save_diagnostic_report(
        self, report: DiagnosticReport, filepath: str | Path
    ) -> None:
        """Save diagnostic report to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(file=filepath, mode="w") as f:
            # json.dump(report.to_dict(), f, indent=2)
            dump_yaml(report.to_dict(), f)

        print(f"Report saved to: {filepath}")

    def run_diagnostic_by_category(
        self,
        category: str,
        verbose: bool = True,
    ) -> DiagnosticReport:
        """Run diagnostic for a specific category only."""
        return self.run_cwe_diagnostic(
            categories=[category],
            verbose=verbose,
        )

    @staticmethod
    def get_diagnostic_cases() -> list[DiagnosticCase]:
        """Return the built-in diagnostic cases."""
        return DIAGNOSTIC_CASES.copy()

    @staticmethod
    def get_diagnostic_categories() -> list[str]:
        """Return list of available diagnostic categories."""
        return list(set(c.category for c in DIAGNOSTIC_CASES))

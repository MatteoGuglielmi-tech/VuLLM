import pytest

from dataset.restructure.shared.proc_utils import extract_function_signature

test_samples_func_signature = [
    (
        "complex_func_signature",
        """
        /* A complex function definition */
        static inline unsigned long
        process_data(int p_id, const char* name) {
          // function body...
        }
        """,
        "static inline unsigned long process_data(int p_id, const char* name)",
        "c"
    )
]

@pytest.mark.parametrize(
    "test_id, input_code, expected_func_signature, lang",
    test_samples_func_signature,
    ids=[t[0] for t in test_samples_func_signature],
)
def test_run_multi_pass_fix_integration(test_id: str, input_code: str, expected_func_signature:str, lang: str):
    candidate_func_signature = extract_function_signature(code=input_code, language_name=lang)
    assert candidate_func_signature is not None
    assert candidate_func_signature.strip() == expected_func_signature.strip()




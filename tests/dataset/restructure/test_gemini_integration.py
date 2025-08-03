import pytest
import json

from dataset.restructure.gemini_client import Gemini

pytestmark: pytest.MarkDecorator = pytest.mark.slow_api

@pytest.fixture(scope="module")
def gemini_client() -> Gemini:
    """Provides an initialized Gemini client for the tests."""

    return Gemini(model_name="gemini-2.5-flash")

def test_generate_description_returns_valid_json_string(gemini_client: Gemini):
    """This is an integration test that makes a real API call to Gemini.
    It verifies that the response is a valid JSON string and respects the
    length constraints.
    """

    probe_code: str = """
    int calculate_sum(int a, int b) {
        // This function returns the sum of two integers.
        return a + b;
    }
    """.strip()

    # <---- API call ---->
    response_text: str = gemini_client.generate_description(func_str=probe_code)

    # <---- validation ---->
    assert isinstance(response_text, str)
    assert len(response_text) > 0

    # when response_mime_type="application/json" a string
    # containing a JSON object is returned.
    try:
        parsed_data = json.loads(s=response_text)
        assert isinstance(parsed_data, str), "The parsed JSON content should be a string."
    except json.JSONDecodeError:
        pytest.fail(f"The response from Gemini was not a valid JSON string: {response_text}")

    # check if the length of the content is reasonable.
    # note: max_output_tokens is not a hard limit on characters.
    print(f"\nReceived description: '{parsed_data}'")
    assert len(parsed_data) < 500, "Description seems excessively long."

    print("\n✅ Test passed: Gemini API integration works as expected.")

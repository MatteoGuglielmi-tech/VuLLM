import pytest
import json

from dataset.restructure.llm_clients.gemini_describer import GeminiClient

pytestmark: pytest.MarkDecorator = pytest.mark.slow_api

@pytest.fixture(scope="module")
def gemini_client() -> GeminiClient:
    """Provides an initialized Gemini client for the tests."""

    return GeminiClient(model_name="gemini-2.5-flash")

def test_generate_description_returns_valid_json_string(gemini_client: GeminiClient):
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
    response_text: list[str] = gemini_client.generate_batch_descriptions(c_code_batch=[probe_code])

    # <---- validation ---->
    assert isinstance(response_text[0], str)
    assert len(response_text) > 0

    # when response_mime_type="application/json" a string
    # containing a JSON object is returned.
    try:
        parsed_data = json.loads(s=response_text[0])
        assert isinstance(parsed_data, str), "The parsed JSON content should be a string."
    except json.JSONDecodeError:
        pytest.fail(f"The response from Gemini was not a valid JSON string: {response_text}")

    # check if the length of the content is reasonable.
    print(f"\nReceived description: '{parsed_data}'")
    assert len(parsed_data) < 500, "Description seems excessively long."

    print("\n✅ Test passed: Gemini API integration works as expected.")

import pytest
import os
import json
import shutil

from dataset.restructure.vulcan_pipeline import Vulcan
from dataset.restructure.llama_describer import LlamaCodeDescriber

@pytest.fixture(scope="session")
def project_root():
    """Returns project root directory."""

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

@pytest.fixture(scope="session")
def test_data_dir(tmpdir_factory):
    """Creates a temporary directory for test data."""

    return tmpdir_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def setup_test_environment(test_data_dir, project_root):
    """Sets up the necessary dummy files for the integration test.
    This runs only once per test session."""

    clang_format_fp: str = os.path.join(project_root, ".clang-format")
    if not os.path.exists(clang_format_fp):
        pytest.fail(f".clang-format file not found at expected path: {clang_format_fp}")

    # create dummy config.json in the temporary test directory
    config_path = test_data_dir.join("config.json")
    config_data = {
        "clang_format_path": clang_format_fp,
        "default_input_path": str(test_data_dir.join("input.jsonl")),
        "default_output_path": str(test_data_dir.join("output.jsonl")),
        "default_metadata_path": str(test_data_dir.join("metadata.jsonl"))
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    # create dummy metadata.jsonl
    metadata_path = test_data_dir.join("metadata.jsonl")
    metadata_entry = {
        "project": "test-project",
        "commit_id": "0123456789abcdef",
        "CWE": "CWE-125",
        "CVE": "CVE-2025-12345",
        "bug_info": "Test out-of-bounds read."
    }
    with open(metadata_path, 'w') as f:
        f.write(json.dumps(metadata_entry) + '\n')

    return config_data

@pytest.fixture
def vulcan_pipeline(monkeypatch, setup_test_environment, test_data_dir):
    """Provides an initialized Vulcan pipeline instance for each test.
    It changes the CWD to the test directory so Vulcan can find config.json.
    """

    monkeypatch.setattr("dataset.restructure.code_augmentor.LlamaCodeDescriber", MockLlamaCodeDescriber)

    original_cwd = os.getcwd()
    os.chdir(test_data_dir)

    try: yield Vulcan()
    finally: os.chdir(original_cwd)

    misc_dir = os.path.join(os.getcwd(), "misc")
    if os.path.exists(misc_dir):
        shutil.rmtree(misc_dir)



class MockLlamaCodeDescriber:
    """A lightweight mock that simulates the real LlamaCodeDescriber."""
    def __init__(self, model_name: str = "mock_model"):
        # This __init__ does nothing; it doesn't load any models.
        print("Initialized MOCK LlamaCodeDescriber.")
        pass

    def generate_description(self, c_code: str) -> str:
        # It always returns a predictable, hardcoded string.
        return "This is a mock function description."

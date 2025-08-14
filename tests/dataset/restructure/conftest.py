import pytest
import os
import json
import shutil

from dataset.restructure.vulcan_pipeline import Vulcan


@pytest.fixture(scope="session")
def project_root():
    """Returns the absolute path to the project's root directory.

    This fixture is session-scoped, so the path is calculated only once.

    Returns
    -------
    str:
        The absolute path to the project root.
    """

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


@pytest.fixture(scope="session")
def test_data_dir(tmpdir_factory: pytest.TempdirFactory):
    """Creates a temporary directory for test data for the entire session.

    Parameters
    ----------
    tmpdir_factory : pytest.TempdirFactory
        The pytest factory for creating temporary directories.

    Returns
    -------
    py.path.local
        A path object representing the temporary directory.
    """

    return tmpdir_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def setup_test_environment(test_data_dir, project_root:str) -> dict[str,str]:
    """Sets up the necessary dummy files for integration tests.

    This fixture runs only once per test session. It creates a dummy
    `config.json` and `metadata.jsonl` file within the session-scoped
    temporary directory, pointing to the real `.clang-format` file.

    Parameters
    ----------
    test_data_dir : py.path.local
        The temporary directory for storing test data.
    project_root : str
        The root directory of the project, used to locate `.clang-format`.

    Returns
    -------
    dict[str,str]
        A dictionary containing the configuration data that was written
        to `config.json`.

    Raises
    ------
    pytest.fail
        If the `.clang-format` file is not found at the expected location.
    """

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


class MockLlamaCodeDescriber:
    """
    A mock that accurately mimics the LlamaCodeDescriber interface.
    """
    def __init__(self, model_name: str = "mock_model"):
        print("Initialized MOCK LlamaCodeDescriber.")
        pass

    def generate_batch_descriptions(self, c_code_batch: list[str]) -> list[str]:
        """
        Returns a list of mock descriptions, one for each input snippet.
        """
        return ["This is a mock function description."] * len(c_code_batch)

    def generate_batch_cwe_descriptions(self, cwe_ids_batch: list[str]) -> list[str]:
        """
        Returns a list of mock CWE descriptions, one for each input CWE.
        """
        return ["This is a mock CWE description."] * len(cwe_ids_batch)


@pytest.fixture
def vulcan_pipeline(monkeypatch: pytest.MonkeyPatch, setup_test_environment: dict[str,str], test_data_dir):
    """
    Provides an initialized Vulcan pipeline instance for each test.
    """

    monkeypatch.setattr("dataset.restructure.vulcan_pipeline.LlamaCodeDescriber", MockLlamaCodeDescriber)
    original_cwd = os.getcwd()
    os.chdir(test_data_dir)
    vulcan_instance = None
    try:
        vulcan_instance = Vulcan()
        yield vulcan_instance
    finally:
        os.chdir(original_cwd)
        misc_dir = test_data_dir.join("misc")
        if os.path.exists(misc_dir):
            shutil.rmtree(misc_dir)


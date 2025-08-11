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

@pytest.fixture
def vulcan_pipeline(monkeypatch: pytest.MonkeyPatch, setup_test_environment: dict[str,str], test_data_dir):
    """Provides an initialized Vulcan pipeline instance for each test.

    This fixture prepares the environment for a single test by:
    1. Mocking the `LlamaCodeDescriber` to prevent actual model calls.
    2. Changing the current working directory to the test data directory
       so that `Vulcan` can automatically find its `config.json`.
    3. Cleaning up by restoring the original CWD and removing any generated
       `misc` directories after the test completes.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The pytest fixture for modifying classes, methods, etc.
    setup_test_environment : dict[str,str]
        A fixture that ensures the test environment and dummy files are set up.
    test_data_dir : py.path.local
        The temporary directory where test files are located.

    Yields
    ------
    Vulcan
        An initialized instance of the Vulcan pipeline.
    """

    monkeypatch.setattr("dataset.restructure.code_augmentor.description_generator", MockLlamaCodeDescriber)

    original_cwd = os.getcwd()
    os.chdir(test_data_dir)

    try: yield Vulcan()
    finally: os.chdir(original_cwd)

    misc_dir = os.path.join(os.getcwd(), "misc")
    if os.path.exists(misc_dir):
        shutil.rmtree(misc_dir)



class MockLlamaCodeDescriber:
    """A lightweight mock that simulates the real LlamaCodeDescriber.

    This class is used during testing to replace the actual LlamaCodeDescriber,
    avoiding the need for a real language model and ensuring tests are fast
    and deterministic. It provides the same interface but returns a
    hardcoded, predictable output.
    """

    def __init__(self, model_name: str = "mock_model"):
        """Initializes the mock code describer.

        Parameters
        ----------
        model_name : str, optional
            A placeholder for the model name, by default "mock_model".
            This parameter is ignored in the mock implementation.
        """

        print("Initialized MOCK LlamaCodeDescriber.")
        pass

    def generate_description(self, c_code: str) -> str:
        """Generates a mock description for a given C code snippet.

        This method simulates the behavior of the real `generate_description`
        by returning a static, predefined string, regardless of the input.

        Parameters
        ----------
        c_code : str
            The C code for which to generate a description. This parameter
            is ignored in the mock implementation.

        Returns
        -------
        str
            A hardcoded mock function description.
        """

        return "This is a mock function description."

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from dataset.analysis.utils import UNUSED
from dataset.analysis.check_repo_language import analyze_repositories


@pytest.fixture
def fake_links_file(tmp_path: Path) -> str:
    """Creates a fake JSON Lines file with existing project metadata."""
    file_path: Path = tmp_path / "links.jsonl"
    with open(file_path, "w") as f:
        # A project that is C++
        f.write('{"project": "project-A", "repo_url": "https://github.com/ownerA/project-A"}\n')
        # A project that is not C++
        f.write('{"project": "project-B", "repo_url": "https://github.com/ownerB/project-B"}\n')
        # A duplicate entry that should be ignored
        f.write('{"project": "project-A", "repo_url": "https://github.com/ownerA/project-A"}\n')
    return str(file_path)

@pytest.fixture
def fake_reference_file(tmp_path: Path) -> str:
    """Creates a fake reference dataset to control which projects are processed."""
    file_path: Path = tmp_path / "reference.jsonl"
    with open(file_path, "w") as f:
        f.write('{"project": "project-A"}\n') # Project with metadata
        f.write('{"project": "project-B"}\n') # Project with metadata
        f.write('{"project": "project-C"}\n') # Project that will require a search
    return str(file_path)


def test_analyze_repositories_counts_cpp_projects(mocker, fake_links_file: str, fake_reference_file: str):
    """
    Tests that the analyzer correctly processes projects from metadata,
    searches for missing ones, calls the GitHub API, and counts C++ projects.
    """

    # Mock the GITHUB_TOKEN to prevent the real function from raising an error
    mocker.patch("dataset.analysis.check_repo_language.GITHUB_TOKEN", "ghp_fake_token")

    mock_api_languages = {
        "ownerA/project-A": {"C++": 50000, "Python": 1000},
        "ownerB/project-B": {"C": 80000, "Makefile": 200},
        "ownerC/project-C": {"C++": 12345}, # This project is found via search
    }

    mock_api_repo_data = {
        "ownerA/project-A": {"full_name": "ownerA/project-A", "fork": False},
        "ownerB/project-B": {"full_name": "ownerB/project-B", "fork": False},
        "ownerC/project-C": {"full_name": "ownerC/project-C", "fork": False},
    }

    def mock_get(url: str, headers: str):
        """
        A smarter mock that handles the different GitHub API endpoints called
        by the function under test (search, repo data, and languages).
        """
        UNUSED(headers)
        response_mock = MagicMock()
        response_mock.raise_for_status.return_value = None

        # 1. Handle REPO SEARCH call (from find_best_github_match)
        if "/search/repositories" in url:
            if "project-C" in url:
                # Simulate finding project-C via search
                response_mock.json.return_value = {
                    "items": [
                        {
                            "full_name": "ownerC/project-C",
                            "html_url": "https://github.com/ownerC/project-C",
                            "stargazers_count": 100
                        }
                    ]
                }
            else:
                # Simulate not finding other projects
                response_mock.json.return_value = {"items": []}

        # 2. Handle REPO LANGUAGES call
        elif "/languages" in url:
            repo_name = url.split("repos/")[1].split("/languages")[0]
            response_mock.json.return_value = mock_api_languages.get(repo_name, {})

        # 3. Handle GET REPO DATA call (from get_github_repo_data)
        elif "/repos/" in url:
            repo_name = url.split("repos/")[1]
            response_mock.json.return_value = mock_api_repo_data.get(repo_name, {})
        
        else:
            # Fallback for any unexpected URL
            response_mock.json.return_value = {}

        return response_mock

    # Patch requests.get to use our smart mock
    mocker.patch("dataset.analysis.check_repo_language.requests.get", side_effect=mock_get)
    # Patch time.sleep to make the test run instantly
    mocker.patch("time.sleep")
    # Patch non-GitHub API calls to simplify the test
    mocker.patch("dataset.analysis.check_repo_language.find_best_gitlab_match", return_value=None)


    # --- Act ---
    # Call the function with our fake, small, and fast test files
    cpp_projects: list[str] = analyze_repositories(
        fp=fake_links_file, reference_dataset=fake_reference_file
    )

    # --- Assert ---
    # The result should contain project-A (from metadata) and project-C (from search)
    assert len(cpp_projects) == 2
    assert "project-A" in cpp_projects
    assert "project-C" in cpp_projects
    assert "project-B" not in cpp_projects # Verify non-C++ project is excluded

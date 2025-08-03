import os
import json
import argparse
from dotenv import load_dotenv

from check_repo_language import analyze_repositories

load_dotenv()
GITHUB_TOKEN = os.environ.get("GITHUB_PAT")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}


def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(
        description="Analyze repositories to produce a list of C++ projects."
    )
    parser.add_argument(
        "--dataset_file",
        required=True,
        help="Path to the main JSON file with all functions.",
    )
    parser.add_argument(
        "--metadata_file",
        required=True,
        help="Path to the JSONL file with repository metadata.",
    )
    parser.add_argument(
        "--output_file",
        default="./assets/cpp_projects.json",
        help="Path to save the final list of C++ projects.",
    )
    args = parser.parse_args()

    if not GITHUB_TOKEN or not GITHUB_TOKEN.startswith("ghp_"):
        print("ERROR: GitHub Personal Access Token is not set.")
        return

    # The analyze_repositories function correctly finds the C++ projects
    cpp_projects: list[str] = analyze_repositories(
        fp=args.metadata_file, reference_dataset=args.dataset_file
    )

    print(f"\nSaving list of {len(cpp_projects)} C++ projects to {args.output_file}...")
    with open(file=args.output_file, mode="w", encoding="utf-8") as f:
        json.dump(obj=sorted(cpp_projects), fp=f, indent=2)

    print("✅ Done.")


if __name__ == "__main__":
    main()

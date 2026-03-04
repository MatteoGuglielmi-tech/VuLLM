import sys

from pathlib import Path

from .analyzer import analyze_project_distribution
from .utilities import load_dataset_from_disk


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python -m src.dataset.analysis.project_analysis.main <path_to_dataset_dir> <path_to_output_dir>")
        sys.exit(1)

    input_dataset_dir: Path = Path(sys.argv[1])
    output_dir: Path = Path(sys.argv[2])
    dataset_dict = load_dataset_from_disk(input_dir=input_dataset_dir)

    results = analyze_project_distribution(
        dataset_dict=dataset_dict, output_dir=output_dir
    )

    # Access typed results
    print(f"Total projects: {results.total_projects}")
    print(f"Train projects: {results.stats['train']['projects']}")
    print(f"Data leakage: {results.leakage['total_overlap']}")

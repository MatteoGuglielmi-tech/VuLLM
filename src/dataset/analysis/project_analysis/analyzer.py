from pathlib import Path
from collections import defaultdict
from datasets import DatasetDict

from .types import SplitStats, LeakageStats, AnalysisResult, ProjectData


def collect_project_data(dataset_dict: DatasetDict) -> ProjectData:
    """
    Collect project names and sample counts from dataset.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dictionary with 'train', 'validation', 'test' splits

    Returns
    -------
    ProjectData
        Named tuple with project names and sample counts
    """
    splits = ["train", "validation", "test"]

    prj_names: dict[str, set[str]] = defaultdict(set)
    prj_sample_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for split in splits:
        for ex in dataset_dict[split]:
            if ex["target"] == 1:  # type: ignore[reportCallIssue, reportArgumentType]
                prj_name = ex.get("project", [])  # type: ignore[reportAttributeAccessIssue]
                if prj_name:
                    prj_names[split].add(prj_name)
                    prj_sample_counts[split][prj_name] += 1

    return ProjectData(project_names=prj_names, sample_counts=prj_sample_counts)


def calculate_split_statistics(
    project_data: ProjectData, total_projects: int
) -> dict[str, SplitStats]:
    """
    Calculate statistics for each split.

    Parameters
    ----------
    project_data : ProjectData
        Project data collected from dataset
    total_projects : int
        Total number of unique projects

    Returns
    -------
    dict[str, SplitStats]
        Statistics for each split
    """
    splits = ["train", "validation", "test"]
    stats: dict[str, SplitStats] = {}

    for split in splits:
        n_projects = len(project_data.project_names[split])
        n_samples = sum(project_data.sample_counts[split].values())
        avg_samples = n_samples / n_projects if n_projects > 0 else 0.0

        stats[split] = SplitStats(
            projects=n_projects,
            percentage=(
                (n_projects / total_projects * 100) if total_projects > 0 else 0.0
            ),
            avg_samples_per_project=avg_samples,
            total_samples=n_samples,
        )

    return stats


def calculate_leakage_statistics(project_names: dict[str, set[str]]) -> LeakageStats:
    """
    Calculate data leakage statistics.

    Parameters
    ----------
    project_names : dict[str, set[str]]
        Project names per split

    Returns
    -------
    LeakageStats
        Leakage statistics across splits
    """
    intersec_train_val = project_names["train"].intersection(
        project_names["validation"]
    )
    intersec_train_test = project_names["train"].intersection(project_names["test"])
    intersec_val_test = project_names["validation"].intersection(project_names["test"])

    return LeakageStats(
        train_validation=len(intersec_train_val),
        train_test=len(intersec_train_test),
        validation_test=len(intersec_val_test),
        total_overlap=len(intersec_train_val | intersec_train_test | intersec_val_test),
    )


def generate_latex_table(
    stats: dict[str, SplitStats],
    total_projects: int,
    leakage_stats: LeakageStats,
    output_dir: Path,
) -> None:
    """
    Generate LaTeX table code.

    Parameters
    ----------
    stats : dict[str, SplitStats]
        Statistics per split
    total_projects : int
        Total unique projects
    leakage_stats : LeakageStats
        Data leakage statistics
    output_dir : Path
        Output directory
    """
    splits = ["train", "validation", "test"]

    # Calculate overall average
    total_samples = sum(stats[s]["total_samples"] for s in splits)
    overall_avg = total_samples / total_projects if total_projects > 0 else 0.0

    latex_code = r"""\begin{table}[htbp]
\centering
\caption{Project distribution across splits}
\label{tab:project-distribution}
\begin{tabular}{lrrr}
\toprule
\textbf{Split} & \textbf{Projects} & \textbf{\% of Total} & \textbf{Avg Samples/Project} \\
\midrule
"""

    # Add data rows
    for split in splits:
        s = stats[split]
        latex_code += f"{split.capitalize()} & {s['projects']} & {s['percentage']:.1f}\\% & {s['avg_samples_per_project']:.1f} \\\\\n"

    latex_code += r"""\midrule
"""
    latex_code += f"\\textbf{{Total (unique)}} & \\textbf{{{total_projects}}} & \\textbf{{100.0\\%}} & \\textbf{{{overall_avg:.1f}}} \\\\\n"

    latex_code += r"""\bottomrule
\end{tabular}
\par\vspace{0.5em}
"""

    # Add caption based on leakage
    if leakage_stats["total_overlap"] == 0:
        min_avg = min(stats[s]["avg_samples_per_project"] for s in splits)
        max_avg = max(stats[s]["avg_samples_per_project"] for s in splits)
        latex_code += f"{{\\small Project counts reflect unique codebase distribution. No project appears in multiple splits (zero data leakage). Average samples per project relatively uniform ({min_avg:.1f}--{max_avg:.1f}), indicating projects of similar size.}}\n"
    else:
        latex_code += f"{{\\small WARNING: Data leakage detected. {leakage_stats['total_overlap']} projects appear in multiple splits.}}\n"

    latex_code += r"""\end{table}
"""

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "project_distribution_table.tex", "w") as f:
        f.write(latex_code)

    print(f"✅ Saved LaTeX table to {output_dir / 'project_distribution_table.tex'}")
    print("\n" + "=" * 60)
    print("LaTeX Table Code:")
    print("=" * 60)
    print(latex_code)
    print("=" * 60)


def print_summary(
    stats: dict[str, SplitStats],
    total_projects: int,
    leakage_stats: LeakageStats,
    project_names: dict[str, set[str]],
) -> None:
    """
    Print summary statistics to console.

    Parameters
    ----------
    stats : dict[str, SplitStats]
        Statistics per split
    total_projects : int
        Total unique projects
    leakage_stats : LeakageStats
        Data leakage statistics
    project_names : dict[str, set[str]]
        Project names per split
    """
    print("\n" + "=" * 70)
    print("PROJECT DISTRIBUTION ANALYSIS")
    print("=" * 70)

    print("\n📊 Distribution Summary:")
    for split, s in stats.items():
        print(f"\n  {split.upper()}:")
        print(f"    Projects: {s['projects']} ({s['percentage']:.1f}%)")
        print(f"    Total samples: {s['total_samples']}")
        print(f"    Avg samples/project: {s['avg_samples_per_project']:.1f}")

    print(f"\n  TOTAL UNIQUE PROJECTS: {total_projects}")

    print("\n🔍 Data Leakage Check:")

    intersec_train_val = project_names["train"].intersection(
        project_names["validation"]
    )
    print(f"  Train ∩ Validation: {leakage_stats['train_validation']} projects")
    if intersec_train_val:
        print(
            f"    → {list(intersec_train_val)[:5]}{'...' if len(intersec_train_val) > 5 else ''}"
        )

    intersec_train_test = project_names["train"].intersection(project_names["test"])
    print(f"  Train ∩ Test: {leakage_stats['train_test']} projects")
    if intersec_train_test:
        print(
            f"    → {list(intersec_train_test)[:5]}{'...' if len(intersec_train_test) > 5 else ''}"
        )

    intersec_val_test = project_names["validation"].intersection(project_names["test"])
    print(f"  Validation ∩ Test: {leakage_stats['validation_test']} projects")
    if intersec_val_test:
        print(
            f"    → {list(intersec_val_test)[:5]}{'...' if len(intersec_val_test) > 5 else ''}"
        )

    if leakage_stats["total_overlap"] == 0:
        print("\n  ✅ ZERO DATA LEAKAGE - All splits have disjoint project sets!")
    else:
        print(
            f"\n  ⚠️  WARNING: {leakage_stats['total_overlap']} projects overlap across splits!"
        )

    print("\n" + "=" * 70)


def analyze_project_distribution(
    dataset_dict: DatasetDict, output_dir: Path
) -> AnalysisResult:
    """
    Analyze project distribution across splits with visualizations.

    Parameters
    ----------
    dataset_dict : DatasetDict
        Dataset dictionary with 'train', 'validation', 'test' splits
    output_dir : Path
        Directory to save output files

    Returns
    -------
    AnalysisResult
        Named tuple with complete analysis results
    """
    from .plots import (
        plot_distribution_overview,
        plot_venn_diagram,
        plot_sample_distribution,
        plot_size_comparison,
    )

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    print("🔍 Collecting project data...")
    project_data = collect_project_data(dataset_dict)

    # Calculate statistics
    total_projects = len(set.union(*project_data.project_names.values()))
    stats = calculate_split_statistics(project_data, total_projects)
    leakage_stats = calculate_leakage_statistics(project_data.project_names)

    # Create visualizations
    print("\n📊 Generating visualizations...")
    plot_distribution_overview(stats, total_projects, plots_dir)
    plot_venn_diagram(project_data.project_names, plots_dir)
    plot_sample_distribution(project_data.sample_counts, plots_dir)
    plot_size_comparison(project_data.sample_counts, plots_dir)

    # Generate LaTeX table
    print("\n📝 Generating LaTeX table...")
    generate_latex_table(stats, total_projects, leakage_stats, output_dir)

    # Print summary
    print_summary(stats, total_projects, leakage_stats, project_data.project_names)

    return AnalysisResult(
        stats=stats,
        total_projects=total_projects,
        leakage=leakage_stats,
        project_names=project_data.project_names,
        sample_counts=project_data.sample_counts,
    )

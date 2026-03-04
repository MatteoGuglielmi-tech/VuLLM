from typing import TypedDict, NamedTuple


class SplitStats(TypedDict):
    """Statistics for a single split."""
    projects: int
    percentage: float
    avg_samples_per_project: float
    total_samples: int


class LeakageStats(TypedDict):
    """Data leakage statistics across splits."""
    train_validation: int
    train_test: int
    validation_test: int
    total_overlap: int


class AnalysisResult(NamedTuple):
    """Complete analysis results."""
    stats: dict[str, SplitStats]
    total_projects: int
    leakage: LeakageStats
    project_names: dict[str, set[str]]
    sample_counts: dict[str, dict[str, int]]


class ProjectData(NamedTuple):
    """Collected project data from dataset."""
    project_names: dict[str, set[str]]
    sample_counts: dict[str, dict[str, int]]

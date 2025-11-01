import logging
from pathlib import Path

from .distribution import plot_filtering_distributions
from .radar import plot_judge_comparison_radar
from .sankey import plot_filtering_sankey

logger = logging.getLogger(__name__)


def visualize_results(
    stats: dict,
    output_dir: Path,
    quality_threshold: float,
    agreements_threshold: float,
):
    """Complete visualization suite for filtering results."""

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_filtering_distributions(
        stats,
        quality_threshold=quality_threshold,
        agreement_threshold=agreements_threshold,
        output_dir=output_dir,
    )
    plot_filtering_sankey(stats, output_dir=output_dir)
    plot_judge_comparison_radar(judge_evaluations=stats["per_judge_metrics"], output_dir=output_dir)

    logger.info(f"✓ Visualizations saved to {output_dir}/")

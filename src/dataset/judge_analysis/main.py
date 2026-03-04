import logging
import sys

from pathlib import Path

from .logging_config import setup_logger
from .utilities import stateless_progress
from .confidence_analysis import plot_judge_confidence_summary, load_evaluations
from .disagreement_analysis import plot_disagreement_analysis
from .judge_paired_disagreement import plot_judge_pair_disagreement
from .cum_dist_quality_scores import run_quality_analysis


logger = logging.getLogger(__name__)

# Global name mapping
JUDGE_NAME_MAP = {
    "Qwen2.5-Coder-32B-Instruct-bnb-4bit": "Qwen-Coder",
    "Qwen2.5-72B": "Qwen-72B",
    "Phi-4": "Phi-4",
    "DeepSeek-R1-Distill-Llama": "DeepSeek",
}


def run_complete_analysis(jsonl_path: Path, output_dir: Path) -> None:
    """Run all analyses."""
    df = load_evaluations(jsonl_path)

    nb_judges = df["judge_name"].nunique()
    logger.debug(f"Judges found: {nb_judges}")
    logger.debug(
        f"Loaded {len(df)} evaluations across {df['entry_id'].nunique()} entries (nb_judges * {len(df)/nb_judges})"
    )

    with stateless_progress(
        description="1️⃣  Generating confidence analysis..."
    ) as status:
        plot_judge_confidence_summary(
            df=df, output_dir=output_dir, names_map=JUDGE_NAME_MAP
        )

        status.update("2️⃣ Generating disagreement analysis ...")
        disagreement_df = plot_disagreement_analysis(df, output_dir)

        status.update("3️⃣ Generating pairwise disagreement analysis...")
        plot_judge_pair_disagreement(df, output_dir, names_map=JUDGE_NAME_MAP)

    run_quality_analysis(df, output_dir, names_map=JUDGE_NAME_MAP)

    # disagreement_df.to_csv(output_dir / "disagreement_metrics.csv", index=False)  # type:ignore[reportAttributeAccessIssue]
    # logger.info(f"💾 Saved disagreement metrics to {output_dir / 'disagreement_metrics.csv'}")

    logger.info("\n✅ Analysis complete!")


if __name__ == "__main__":
    setup_logger()

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.dataset.judge_analysis.main <per_judge_evaluations.jsonl> [output_dir]"
        )
        print("\nExample:")
        print(
            "  python -m src.dataset.judge_analysis.main ./DiverseVul/merged/best_reasonings.jsonl src/dataset/judge_analysis/assets"
        )
        sys.exit(1)

    run_complete_analysis(jsonl_path=Path(sys.argv[1]), output_dir=Path(sys.argv[2]))

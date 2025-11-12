from pathlib import Path
from typing import Any
import jsonlines
import logging

from tqdm import tqdm

from .judge import LLMJudge
from .judge_types import JudgeConfig
from ..datatypes import EvaluationResult
from ..utilities import (
    iter_jsonl_samples,
    rich_rule,
    rich_panel,
    build_table,
)

logger = logging.getLogger(__name__)


class SingleJudgeEvaluator:
    """Evaluate all samples with a single judge (optimized)."""

    def __init__(self, judge_config: JudgeConfig):
        self.judge_config = judge_config
        self.judge = None

        # pretty log
        table = build_table(
            data=judge_config.to_dict,
            columns=["Attribute", "Position"],
        )
        rich_panel(
            table,
            panel_title=f"Configuration for judge {judge_config.model_name}",
            border_style="green",
            panel_padding=(1, 3),
        )
        rich_rule()

    def evaluate_dataset(
        self, input_jsonl: Path, output_jsonl: Path, save_interval: int = 100
    ):
        """Evaluate entire dataset with one judge."""

        logger.info(f"🚀 Starting evaluation with {self.judge_config.model_name} 🚀")
        self.judge = LLMJudge(self.judge_config)
        self.judge.load()

        try:
            with open(file=input_jsonl, mode="r") as f:
                total_samples: int = sum(1 for _ in f)

            logger.info(f"🔎 Evaluating {total_samples} samples... 🔎")

            processed: int = 0
            batch: list[dict[str, Any]] = []
            with jsonlines.open(file=output_jsonl, mode="w") as writer:
                for sample in tqdm(
                    iter_jsonl_samples(input_jsonl),
                    total=total_samples,
                    desc=f"Judge: {self.judge_config.ref_name}",
                ):
                    eval_result: EvaluationResult = self.judge.evaluate(sample)

                    output: dict[str, Any] = {
                        "sample_id": sample.sample_id,
                        "judge_name": self.judge_config.ref_name,
                        "evaluation": eval_result.to_dict(),
                        "judge_config": self.judge_config.to_dict,
                    }

                    batch.append(output)
                    processed += 1

                    if len(batch) >= save_interval:
                        writer.write_all(batch)
                        batch = []

                if batch:
                    writer.write_all(batch)

            logger.info(f"✓ Completed: {processed} samples evaluated")
            logger.info(f"✓ Results saved to {output_jsonl}")
        finally:
            self.judge.unload()
            self.judge = None

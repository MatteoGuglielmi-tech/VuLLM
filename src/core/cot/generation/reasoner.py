import os
import logging
import json
import jsonlines
from typing import Any, Generator
from tqdm import tqdm
from dataclasses import dataclass

from src.core.cot.generation.llm_clients.base import ReasoningGenerator


logger = logging.getLogger(__name__)


@dataclass
class Reasoner:
    input_fp: str
    output_fp: str
    cot_generator: ReasoningGenerator
    batch_size: int
    max_completion_tokens: int

    def _read_and_parse_lines(self) -> Generator[dict[str, Any], None, None]:
        """Streamline reading of JSONL file."""

        with open(file=self.input_fp, mode="r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line {i}. Skipping.")

    def run(self):
        """Executes the full generation pipeline by streaming data in chunks."""

        output_dir = os.path.dirname(self.output_fp)
        if output_dir:
            os.makedirs(name=output_dir, exist_ok=True)

        with open(file=self.input_fp, mode="r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        line_generator = self._read_and_parse_lines()

        with (
            jsonlines.open(file=self.output_fp, mode="w") as writer,
            tqdm(total=total_lines, desc="Generating Reasonings", unit="line") as pbar,
        ):
            chunk = []
            for line_obj in line_generator:
                chunk.append(line_obj)
                if len(chunk) == self.batch_size:
                    # batch
                    # reasonings = self.cot_generator.generate_reasoning(entries=chunk, batch_size=self.batch_size)
                    reasonings = self.cot_generator.generate_reasoning(
                        mini_batch=chunk,
                        max_completion_tokens=self.max_completion_tokens,
                    )
                    processed_chunk = [
                        {**line, "reasoning": reasoning}
                        for line, reasoning in zip(chunk, reasonings)
                    ]
                    writer.write_all(processed_chunk)
                    pbar.update(len(chunk))
                    chunk = []

            # for remaining samples
            if chunk:
                reasonings = self.cot_generator.generate_reasoning(
                    mini_batch=chunk, max_completion_tokens=self.max_completion_tokens
                )
                processed_chunk = [
                    {**line, "reasoning": reason}
                    for line, reason in zip(chunk, reasonings)
                ]
                writer.write_all(processed_chunk)
                pbar.update(len(chunk))

        logger.info(f"✅ Finished dataset saved to: {self.output_fp} ✅")

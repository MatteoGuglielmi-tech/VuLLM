"""
BatchCoTGenerator: Cost-effective bulk Chain-of-Thought generation using
OpenAI/Azure Batch API.

The Batch API offers ~50% cost savings in exchange for higher latency
(results within 24 hours). Ideal for large-scale dataset generation
where real-time responses aren't needed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

from openai import OpenAI

from ..parsers import OutputParser, ParseResult
from ..utilities import rich_print

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Possible states of a batch job."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Tracks a submitted batch job."""

    batch_id: str
    input_file_id: str
    status: BatchStatus
    total_requests: int | None = None
    completed_requests: int = 0
    failed_requests: int = 0
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if batch has reached a terminal state."""
        return self.status in {
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
        }

    @property
    def is_successful(self) -> bool:
        """Check if batch completed successfully."""
        return self.status == BatchStatus.COMPLETED


@dataclass
class BatchStatistics:
    """Statistics for batch processing."""

    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    parse_successes: int = 0
    parse_failures: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BATCH PROCESSING SUMMARY",
            "=" * 60,
            f"Submitted to API:   {self.total_submitted:,}",
            f"API completed:      {self.total_completed:,}",
            f"API failed:         {self.total_failed:,}",
            f"Parse successes:    {self.parse_successes:,}",
            f"Parse failures:     {self.parse_failures:,}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BatchCoTGenerator:
    """
    Generates Chain-of-Thought reasoning using OpenAI Batch API.

    Workflow:
    1. prepare_batch() - Create JSONL file with all requests
    2. submit_batch() - Upload and start batch job
    3. wait_for_completion() - Poll until done (or check manually)
    4. retrieve_results() - Download and parse results

    Attributes
    ----------
    client : OpenAI
        OpenAI client instance.
    model : str
        Model to use (e.g., "o4-mini").
    prompt_builder : Callable
        Function that builds messages from a sample.
    parser : OutputParser
        Parser for extracting reasoning and verdict.
    stats : BatchStatistics
        Statistics from batch processing.

    Examples
    --------
    >>> generator = BatchCoTGenerator(
    ...     client=OpenAI(),
    ...     model="o4-mini",
    ...     prompt_builder=my_prompt_builder,
    ... )
    >>>
    >>> # Step 1: Prepare batch file
    >>> generator.prepare_batch(samples, Path("batch_input.jsonl"))
    >>>
    >>> # Step 2: Submit
    >>> job = generator.submit_batch(Path("batch_input.jsonl"))
    >>> print(f"Submitted: {job.batch_id}")
    >>>
    >>> # Step 3: Wait (can be done later / in separate script)
    >>> job = generator.wait_for_completion(job.batch_id, poll_interval=300)
    >>>
    >>> # Step 4: Retrieve results
    >>> results = generator.retrieve_results(job, Path("batch_output.jsonl"))
    """

    def __init__(
        self,
        client: OpenAI,
        model: str,
        prompt_builder: Callable[[dict[str, Any]], list[dict[str, str]]],
        max_completion_tokens: int = 4096,
        parser: OutputParser | None = None,
    ) -> None:
        """
        Initialize the batch generator.

        Parameters
        ----------
        client : OpenAI
            OpenAI client (works with Azure OpenAI too).
        model : str
            Model identifier.
        prompt_builder : Callable
            Function that takes a sample dict and returns messages list.
        max_completion_tokens : int
            Maximum tokens for completions.
        parser : OutputParser | None
            Parser instance (creates default if None).
        """
        self.client = client
        self.model = model
        self.prompt_builder = prompt_builder
        self.max_completion_tokens = max_completion_tokens
        self.parser = parser or OutputParser()
        self.stats = BatchStatistics()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = BatchStatistics()
        self.parser.reset_stats()

    def _create_batch_request(
        self, sample: dict[str, Any], custom_id: str
    ) -> dict[str, Any]:
        """Create a single batch request entry."""
        messages = self.prompt_builder(sample)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "max_completion_tokens": self.max_completion_tokens,
                "messages": messages,
                "temperature": 0.1,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
        }

    def prepare_batch(
        self,
        samples: list[dict[str, Any]],
        output_path: Path,
        id_field: str = "id",
    ) -> int:
        """
        Prepare batch input JSONL file.

        Parameters
        ----------
        samples : list[dict]
            Samples to process.
        output_path : Path
            Where to write the batch input file.
        id_field : str
            Field in sample to use as custom_id (falls back to index).

        Returns
        -------
        int
            Number of requests written.
        """
        logger.info(f"Preparing batch file with {len(samples)} samples...")

        with open(file=output_path, mode="w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                custom_id = str(sample.get(id_field, i))
                request = self._create_batch_request(sample, custom_id)
                f.write(json.dumps(request) + "\n")

        logger.info(f"Wrote batch file to {output_path}")
        return len(samples)

    def submit_batch(
        self,
        input_path: Path,
        completion_window: Literal["24h"] = "24h",
        metadata: dict[str, str] | None = None,
    ) -> BatchJob:
        """
        Upload batch file and submit job.

        Parameters
        ----------
        input_path : Path
            Path to batch input JSONL file.
        completion_window : str
            Time window for completion (default "24h").
        metadata : dict | None
            Optional metadata to attach to batch.

        Returns
        -------
        BatchJob
            Job tracking object.
        """

        logger.info(f"Uploading batch file {input_path}...")

        with open(file=input_path, mode="rb") as f:
            uploaded_file = self.client.files.create(file=f, purpose="batch")

        logger.info(f"Uploaded file: {uploaded_file.id}")

        # Count requests
        with open(file=input_path, mode="r") as f:
            total_requests = sum(1 for line in f if line.strip())

        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )

        self.reset_stats()
        self.stats.total_submitted = total_requests

        logger.info(f"Submitted batch: {batch.id}")

        return BatchJob(
            batch_id=batch.id,
            input_file_id=uploaded_file.id,
            status=BatchStatus(batch.status),
            total_requests=total_requests,
        )

    def check_status(self, batch_id: str) -> BatchJob:
        """
        Check current status of a batch job.

        Parameters
        ----------
        batch_id : str
            The batch ID to check.

        Returns
        -------
        BatchJob
            Updated job status.
        """
        batch_response = self.client.batches.retrieve(batch_id)

        # examine job status
        rich_print(batch_response.model_dump_json(indent=2, ensure_ascii=False))

        return BatchJob(
            batch_id=batch_response.id,
            input_file_id=batch_response.input_file_id,
            status=BatchStatus(batch_response.status),
            total_requests=(
                batch_response.request_counts.total
                if batch_response.request_counts is not None
                else None
            ),
            completed_requests=(
                batch_response.request_counts.completed
                if batch_response.request_counts is not None
                else 0
            ),
            failed_requests=(
                batch_response.request_counts.failed
                if batch_response.request_counts is not None
                else 0
            ),
            output_file_id=batch_response.output_file_id,
            error_file_id=batch_response.error_file_id,
            created_at=batch_response.created_at
        )

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 300,
        timeout: int | None = None,
    ) -> BatchJob:
        """
        Poll until batch completes or timeout.

        Parameters
        ----------
        batch_id : str
            The batch ID to wait for.
        poll_interval : int
            Seconds between status checks (default 5 min).
        timeout : int | None
            Maximum seconds to wait (None = no timeout).

        Returns
        -------
        BatchJob
            Final job status.

        Raises
        ------
        TimeoutError
            If timeout exceeded before completion.
        """

        job: BatchJob | None = None

        while True:

            if job is not None:
                time.sleep(poll_interval)

            job = self.check_status(batch_id)

            logger.info(
                f"Batch {batch_id}: {job.status.value} "
                f"({job.completed_requests}/{job.total_requests} completed, "
                f"{job.failed_requests} failed)"
            )

            if job.is_terminal:
                return job

            if timeout and job.created_at:
                elapsed = time.time() - job.created_at
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Batch {batch_id} did not complete within {timeout}s "
                        f"(created {elapsed:.0f}s ago)"
                    )


    def save_errors(self, job: BatchJob, output_path: Path) -> int:
        """
        Download and save batch errors to a JSON file.

        Parameters
        ----------
        job : BatchJob
            Batch job (may be failed or have partial failures).
        output_path : Path
            Where to save the errors.

        Returns
        -------
        int
            Number of errors saved.
        """
        if not job.error_file_id:
            logger.info("No error file available")
            return 0

        logger.info(f"Downloading errors from {job.error_file_id}...")
        content = self.client.files.content(job.error_file_id)

        errors = []
        for line in content.text.strip().split("\n"):
            if line.strip():
                try:
                    errors.append(json.loads(line))
                except json.JSONDecodeError:
                    errors.append({"raw": line})

        output_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        logger.info(f"Saved {len(errors)} errors to {output_path}")
        return len(errors)

    def retrieve_results(
        self,
        job: BatchJob,
        output_path: Path | None = None,
    ) -> list[tuple[str, dict | None]]:
        """
        Download and parse batch results.

        Parameters
        ----------
        job : BatchJob
            Completed batch job.
        output_path : Path | None
            Optional path to directory where saving raw results to.

        Returns
        -------
        list[tuple[str, ParsedSample | None]]
            List of (custom_id, parsed_sample) tuples.
            parsed_sample is None if parsing failed.
        """
        if not job.is_successful:
            if job.error_file_id:
                error_path = (
                    output_path.parent / "batch_errors.json"
                    if output_path
                    else Path("batch_errors.json")
                )
                self.save_errors(job=job, output_path=error_path)
            raise ValueError(f"Batch not successful: {job.status.value}")

        if not job.output_file_id:
            raise ValueError("No output file available")

        logger.info(f"Downloading results from {job.output_file_id}...")

        content = self.client.files.content(job.output_file_id)
        raw_results = content.text

        if output_path:
            output_path.write_text(data=raw_results, encoding="utf-8")
            logger.info(f"Saved raw results to {output_path}")

        # Parse results
        results: list[tuple[str, dict | None]] = []

        for raw_response in raw_results.strip().split("\n"):
            if not raw_response.strip():
                continue

            # extracting respose from response json file
            try:
                json_response = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse result line: {raw_response[:100]}...")
                continue

            custom_id = json_response.get("custom_id", "unknown")
            response = json_response.get("response", {})

            if json_response.get("error"):
                logger.warning(f"API error for {custom_id}: {json_response['error']}")
                self.stats.total_failed += 1
                results.append((custom_id, None))
                continue

            self.stats.total_completed += 1

            try:
                body = response.get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    raise ValueError("No choices in response")

                content = choices[0].get("message", {}).get("content", "")
            except (KeyError, IndexError) as e:
                logger.warning(f"Failed to extract content for {custom_id}: {e}")
                self.stats.parse_failures += 1
                results.append((custom_id, None))
                continue

            # parse and validation
            parse_result: ParseResult = self.parser.parse(content)

            if parse_result.success:
                self.stats.parse_successes += 1
                results.append((custom_id, parse_result.sample.model_dump()))
            else:
                logger.debug(f"Parse failed for {custom_id}: {parse_result.error}")
                self.stats.parse_failures += 1
                results.append((custom_id, None))

        logger.info(
            f"Retrieved {len(results)} results: "
            f"{self.stats.parse_successes} parsed, "
            f"{self.stats.parse_failures} failed"
        )

        return results

    def cancel_batch(self, batch_id: str) -> BatchJob:
        """
        Cancel a running batch job.

        Parameters
        ----------
        batch_id : str
            The batch ID to cancel.

        Returns
        -------
        BatchJob
            Updated job status.
        """
        self.client.batches.cancel(batch_id)
        return self.check_status(batch_id)


def run_batch_pipeline(
    client: OpenAI,
    model: str,
    samples: list[dict[str, Any]],
    prompt_builder: Callable[[dict[str, Any]], list[dict[str, str]]],
    working_dir: Path,
    poll_interval: int = 300,
    max_completion_tokens: int = 4096,
) -> tuple[list[ParseResult], BatchStatistics]:
    """
    Convenience function to run complete batch pipeline.

    Parameters
    ----------
    client : OpenAI
        OpenAI client.
    model : str
        Model to use.
    samples : list[dict]
        Samples to process.
    prompt_builder : Callable
        Function to build prompts.
    working_dir : Path
        Directory for intermediate files.
    poll_interval : int
        Seconds between status checks.
    max_completion_tokens : int
        Max tokens per completion.

    Returns
    -------
    tuple[list[ParsedSample], BatchStatistics]
        Successful samples and statistics.
    """
    working_dir.mkdir(parents=True, exist_ok=True)

    generator = BatchCoTGenerator(
        client=client,
        model=model,
        prompt_builder=prompt_builder,
        max_completion_tokens=max_completion_tokens,
    )

    # Prepare and submit
    input_path = working_dir / "batch_input.jsonl"
    generator.prepare_batch(samples, input_path)
    job = generator.submit_batch(input_path)

    # Wait for completion
    logger.info(f"Waiting for batch {job.batch_id} to complete...")
    job = generator.wait_for_completion(job.batch_id, poll_interval=poll_interval)

    if not job.is_successful:
        logger.error(f"Batch failed with status: {job.status.value}")
        return [], generator.stats

    # Retrieve results
    output_path = working_dir / "batch_output.jsonl"
    results = generator.retrieve_results(job=job, output_path=output_path)

    # Filter successful parses
    successful = [sample for _, sample in results if sample is not None]

    return successful, generator.stats

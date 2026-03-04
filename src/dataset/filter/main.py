"""
Filter JSONL vulnerability dataset.

Rules:
1. Keep ALL safe samples (target=0) and set their `cwe` field to []
2. Keep vulnerable samples (target=1) only if:
   - cwe list is non-empty
   - cwe_desc does not contain any "Description not found."
   - ALL CWEs in the sample have frequency > threshold (default: 50)

Remove fields: agreement, quality_score, per_judge_evaluations
"""

import json
import sys
import logging
import csv

from typing import TypedDict
from collections import defaultdict
from pathlib import Path

from torch import mul

from .utilities import (
    RichColors,
    rich_progress,
    rich_status,
    rich_rule,
    stateless_progress,
    rich_panels_grid,
    build_table,
    build_panel,
)
from .logging_config import setup_logger

logger = logging.getLogger(__name__)


class RawSample(TypedDict):
    project: str
    target: int
    func: str
    cwe: list[str]
    cwe_desc: list[str]
    reasoning: str
    agreement: int
    quality_score: int
    per_judge_evaluations: dict


class FilteredSample(TypedDict):
    project: str
    target: int
    func: str
    cwe: list[str]
    cwe_desc: list[str]
    reasoning: str


invalid_cwes = defaultdict(int)


def count_jsonl_lines(jsonl_path: Path | str) -> int:
    with open(file=jsonl_path, mode="r") as f:
        return sum(1 for _ in f)


def is_valid_example(example: RawSample) -> bool:
    """Check if example is valid for training."""
    # keep  vulnerable samples with:
    # - filled cwe list
    # - valid vulenrability descrpions
    cwes = example["cwe"]
    if not cwes:
        invalid_cwes["empty_cwe"] = + 1 
        return False

    # Check descriptions are valid
    cwe_descs = example["cwe_desc"]
    if not cwe_descs:
        invalid_cwes["empty_desc"] = + 1 
        return False

    for cwe, d in zip(cwes, cwe_descs):
        if (not d.strip()) or (d.strip() == "Description not found."):
            invalid_cwes[cwe] += 1

    # Ensure no placeholder descriptions
    return all(desc.strip() and desc != "Description not found." for desc in cwe_descs)


def remove_unwanted_fields(
    entry: RawSample | FilteredSample,
    fields_to_remove: list[str] = [
        "agreement",
        "quality_score",
        "per_judge_evaluations",
    ],
):
    for field in fields_to_remove:
        entry.pop(field, None)


def get_total_count(cwes_data: dict) -> int:
    """Get total count across all CWEs."""
    return sum(data["count"] for data in cwes_data.values())


def save_cwes_to_csv(cwes_data: dict, output_path: Path | str, filename: str) -> None:
    """Save CWE statistics to CSV."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / filename

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["CWE_ID", "Count", "Description"])

        # Sort by count descending
        for cwe_id, data in sorted(
            cwes_data.items(), key=lambda x: x[1]["count"], reverse=True
        ):
            writer.writerow([cwe_id, data["count"], data["description"]])

    logger.info(f"✅ Saved {len(cwes_data)} CWEs to {csv_path}")


def filter_dataset_loose(
    input_path: str, output_path: str, frequency_threshold: int = 50
) -> None:
    """Filter the dataset and generate logs."""

    logger.info("Pass 1: Counting CWE frequencies...")

    cwe_frequency = defaultdict(int)
    valid_vulnerable_entries: list[FilteredSample] = []
    safe_entries = []

    original_total = 0
    original_safe = 0
    original_vulnerable = 0

    invalid_cwe_filtered = defaultdict(int)

    with open(file=input_path, mode="r", encoding="utf-8") as f:
        for line_num, line in rich_progress(
            enumerate(f, 1),
            total=count_jsonl_lines(input_path),
            description="Filtering dataset",
        ):
            line = line.strip()
            if not line:
                continue

            try:
                entry: RawSample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Warning: Skipping line {line_num} due to JSON error: {e}"
                )
                continue

            original_total += 1
            target, cwe_list = entry["target"], entry["cwe"]

            if target == 0:
                original_safe += 1
                safe_entries.append(entry)

            elif target == 1:
                original_vulnerable += 1
                if is_valid_example(example=entry):
                    for cwe in cwe_list:
                        cwe_frequency[cwe] += 1
                    valid_vulnerable_entries.append(entry)
                else:
                    for cwe in cwe_list:
                        invalid_cwe_filtered[cwe] += 1
            else:
                # unlikely
                logger.warning(
                    f"Warning: Line {line_num} has unexpected target value: {target}"
                )

    logger.info(f"  Found {len(cwe_frequency)} unique CWEs in valid vulnerable samples")

    cwes_above_threshold = {
        cwe for cwe, count in cwe_frequency.items() if count > frequency_threshold
    }
    cwes_below_threshold = {
        cwe for cwe, count in cwe_frequency.items() if count <= frequency_threshold
    }

    logger.info(
        f"  CWEs above threshold (> {frequency_threshold}): {len(cwes_above_threshold)}"
    )
    logger.info(
        f"  CWEs at or below threshold (≤ {frequency_threshold}): {len(cwes_below_threshold)}"
    )

    logger.info("Pass 2: Filtering based on frequency threshold...")
    with rich_status(description="Pass 2: Filtering based on frequency threshold..."):
        filtered_entries = []
        filtered_safe = 0
        filtered_vulnerable = 0

        cwes_kept = defaultdict(lambda: {"count": 0, "description": ""})
        cwes_filtered_low_freq = defaultdict(lambda: {"count": 0, "description": ""})

        for entry in safe_entries:
            entry["cwe"] = []
            entry["cwe_desc"] = []
            remove_unwanted_fields(entry)
            filtered_entries.append(entry)
            filtered_safe += 1

        # Process valid vulnerable entries
        for filtered_entry in valid_vulnerable_entries:
            cwe_list = filtered_entry["cwe"]
            cwe_desc_list = filtered_entry["cwe_desc"]

            all_cwes_above_threshold = all(
                cwe in cwes_above_threshold for cwe in cwe_list
            )
            if all_cwes_above_threshold:
                for idx, cwe in enumerate(cwe_list):
                    cwes_kept[cwe]["count"] += 1  # type: ignore[reportOperatorIssue]
                    if not cwes_kept[cwe]["description"] and idx < len(cwe_desc_list):
                        cwes_kept[cwe]["description"] = cwe_desc_list[idx]

                remove_unwanted_fields(entry=filtered_entry)
                filtered_entries.append(filtered_entry)
                filtered_vulnerable += 1
            else:
                for i, cwe in enumerate(cwe_list):
                    cwes_filtered_low_freq[cwe]["count"] += 1  # type: ignore[reportOperatorIssue]
                    if not cwes_filtered_low_freq[cwe]["description"] and i < len(
                        cwe_desc_list
                    ):
                        cwes_filtered_low_freq[cwe]["description"] = cwe_desc_list[i]

        filtered_total = filtered_safe + filtered_vulnerable

    # Write filtered output
    with (
        open(file=output_path, mode="w", encoding="utf-8") as f,
        stateless_progress(
            description=f"📂 Saving filtered dataset to {output_path}"
        ) as status,
    ):
        for filtered_entry in filtered_entries:
            f.write(json.dumps(filtered_entry, ensure_ascii=False) + "\n")
        status.stop()

    # Write stats
    with stateless_progress(description=f"🎒 Saving csv stats files") as status:
        status.update("Saving kept statistics")
        save_cwes_to_csv(cwes_kept, Path(__file__).parent / "stats", "cwes_kept.csv")
        status.update("Saving rejected statistics")
        save_cwes_to_csv(
            cwes_filtered_low_freq, Path(__file__).parent / "stats", "cwes_filtered.csv"
        )
        status.stop()

    rich_rule(title="[green4]FILTERING COMPLETE[/green4]", style=RichColors.GREEN4)

    original_stats = {
        "Total samples": f"{original_total:,}",
        "Safe (target=0)": f"{original_safe:,}",
        "Vulnerable (target=1)": f"{original_vulnerable:,}",
    }
    filtered_stats = {
        "Total samples": f"{filtered_total:,}",
        "Safe (target=0)": f"{filtered_safe:,}",
        "Vulnerable (target=1)": f"{filtered_vulnerable:,}",
    }

    original_table = build_table(
        data=original_stats, title="Original Dataset", columns=["Stat", "Value"]
    )
    filtered_table = build_table(
        data=filtered_stats, title="Filtered Dataset", columns=["Stat", "Value"]
    )
    p1 = build_panel(
        renderable=[original_table, filtered_table],
        panel_title=f"Frequency threshold: > {frequency_threshold}",
        border_style=RichColors.CORNFLOWER_BLUE,
    )

    removed_vulnerable = original_vulnerable - filtered_vulnerable
    text = f"Vulnerable samples removed: {removed_vulnerable:,}\n"

    # Breakdown of why vulnerable samples were removed
    initially_invalid = original_vulnerable - len(valid_vulnerable_entries)
    low_freq_filtered = len(valid_vulnerable_entries) - filtered_vulnerable

    text += f"    - Invalid CWE (empty or bad desc): {initially_invalid:,}\n"
    text += (
        f"    - Low frequency CWE (≤{frequency_threshold}):    {low_freq_filtered:,}"
    )

    p2 = build_panel(text, panel_title="Result", border_style=RichColors.GREEN_YELLOW)
    rich_panels_grid([p1, p2], grid_shape=(1, 2))
    rich_rule(style=RichColors.GREEN4)

    print("\n" + "-" * 70)
    print(f"CWEs FILTERED OUT - Invalid (empty or 'Description not found.')")
    print("-" * 70)
    if invalid_cwe_filtered:
        for cwe in sorted(invalid_cwe_filtered.keys()):
            print(f"  {cwe}: {invalid_cwe_filtered[cwe]:,}")
        print(f"\n  Total occurrences: {sum(invalid_cwe_filtered.values()):,}")
        print(f"  Unique CWEs: {len(invalid_cwe_filtered)}")
    else:
        print("  None")

    print("\n" + "-" * 70)
    print(f"CWEs FILTERED OUT - Low Frequency (≤{frequency_threshold})")
    print("-" * 70)
    if cwes_filtered_low_freq:
        # Show with their original frequency
        for cwe in sorted(cwes_filtered_low_freq.keys()):
            orig_freq = cwe_frequency[cwe]
            print(
                f"  {cwe}: {cwes_filtered_low_freq[cwe]["count"]:,} (original freq: {orig_freq})"
            )

        print(
            f"\n  Total occurrences filtered: {get_total_count(cwes_filtered_low_freq):,}"
        )
        print(f"  Unique CWEs: {len(cwes_filtered_low_freq)}")
    else:
        print("  None")

    print("\n" + "-" * 70)
    print(f"CWEs KEPT (frequency > {frequency_threshold})")
    print("-" * 70)
    if cwes_kept:
        for cwe in sorted(cwes_kept.keys()):
            orig_freq = cwe_frequency[cwe]
            print(f"  {cwe}: {cwes_kept[cwe]["count"]:,} (original freq: {orig_freq})")

        print(f"\n  Total kept CWE occurrences: {get_total_count(cwes_kept):,}")
        print(f"  Unique CWEs kept: {len(cwes_kept)}")

    else:
        print("  None")

    print("\n" + "=" * 70)
    print(f"Output written to: {output_path}")
    print("=" * 70)


def compute_cwe_frequencies(
    entries: list[RawSample | FilteredSample],
) -> dict[str, int]:
    """Compute CWE frequencies from a list of entries."""
    cwe_frequency: dict[str, int] = defaultdict(int)
    for entry in entries:
        for cwe in entry["cwe"]:
            cwe_frequency[cwe] += 1
    return dict(cwe_frequency)


def filter_by_threshold(
    entries: list[RawSample | FilteredSample],
    cwe_frequency: dict[str, int],
    frequency_threshold: int,
) -> tuple[list[RawSample | FilteredSample], list[RawSample | FilteredSample]]:
    """
    Filter entries keeping only those where ALL CWEs are above threshold.

    Returns:
        Tuple of (kept_entries, filtered_entries)
    """
    cwes_above_threshold = {
        cwe for cwe, count in cwe_frequency.items() if count > frequency_threshold
    }

    kept: list[RawSample | FilteredSample] = []
    filtered: list[RawSample | FilteredSample] = []

    for entry in entries:
        if all(cwe in cwes_above_threshold for cwe in entry["cwe"]):
            kept.append(entry)
        else:
            filtered.append(entry)

    return kept, filtered


def iterative_frequency_filter(
    entries: list[RawSample | FilteredSample],
    frequency_threshold: int,
    max_iterations: int = 20,
) -> tuple[
    list[RawSample | FilteredSample],
    list[RawSample | FilteredSample],
    dict[str, int],
    dict[str, int],
    list[dict],
]:
    """
    Iteratively filter entries until all remaining CWEs are above threshold.

    This handles the case where removing samples causes other CWEs to drop
    below threshold, requiring additional filtering rounds.

    Parameters
    ----------
    entries : list
        List of vulnerable samples to filter.
    frequency_threshold : int
        Minimum frequency required for each CWE.
    max_iterations : int
        Safety limit to prevent infinite loops.

    Returns
    -------
    tuple
        - kept_entries: Samples that passed all filtering
        - filtered_entries: Samples that were filtered out
        - initial_frequencies: CWE frequencies before filtering
        - final_frequencies: CWE frequencies after filtering
        - iteration_log: Log of each iteration's statistics
    """
    current_entries = entries.copy()
    all_filtered: list[RawSample | FilteredSample] = []
    iteration_log: list[dict] = []

    # Record initial frequencies
    initial_frequencies = compute_cwe_frequencies(current_entries)

    logger.info(f"  Starting iterative filtering with {len(current_entries)} samples")
    logger.info(f"  Initial unique CWEs: {len(initial_frequencies)}")

    for iteration in range(1, max_iterations + 1):
        # Compute current frequencies
        cwe_frequency = compute_cwe_frequencies(current_entries)

        cwes_above = sum(1 for c in cwe_frequency.values() if c > frequency_threshold)
        cwes_below = len(cwe_frequency) - cwes_above

        # Filter
        kept, filtered = filter_by_threshold(
            entries=current_entries,
            cwe_frequency=cwe_frequency,
            frequency_threshold=frequency_threshold,
        )

        # Log iteration stats
        iter_stats = {
            "iteration": iteration,
            "samples_before": len(current_entries),
            "samples_after": len(kept),
            "samples_removed": len(filtered),
            "cwes_above_threshold": cwes_above,
            "cwes_below_threshold": cwes_below,
        }
        iteration_log.append(iter_stats)

        logger.info(
            f"  Iteration {iteration}: "
            f"{len(current_entries):,} → {len(kept):,} samples "
            f"(-{len(filtered):,}), "
            f"CWEs above threshold: {cwes_above}/{len(cwe_frequency)}"
        )

        # Collect filtered samples
        all_filtered.extend(filtered)

        # Check for convergence
        if len(kept) == len(current_entries):
            logger.info(f"  ✅ Converged after {iteration} iteration(s)")
            break

        if len(kept) == 0:
            logger.warning(
                f"  ⚠️ All samples filtered out after {iteration} iterations!"
            )
            break

        current_entries = kept
    else:
        logger.warning(f"  ⚠️ Did not converge after {max_iterations} iterations")

    # Compute final frequencies
    final_frequencies = compute_cwe_frequencies(current_entries)

    # Verify all CWEs now pass threshold
    violations = [
        (cwe, count)
        for cwe, count in final_frequencies.items()
        if count <= frequency_threshold
    ]
    if violations:
        logger.error(f"  ❌ BUG: {len(violations)} CWEs still below threshold:")
        for cwe, count in violations:
            logger.error(f"      {cwe}: {count}")
    else:
        logger.info(
            f"  ✅ All {len(final_frequencies)} CWEs have frequency > {frequency_threshold}"
        )

    return (
        current_entries,
        all_filtered,
        initial_frequencies,
        final_frequencies,
        iteration_log,
    )


def filter_dataset(
    input_path: str, output_path: str, frequency_threshold: int = 50
) -> None:
    """Filter the dataset with iterative frequency-based filtering."""

    logger.info("Pass 1: Loading and validating samples...")

    valid_vulnerable_entries: list[RawSample] = []
    safe_entries: list[RawSample] = []

    original_total = 0
    original_safe = 0
    original_vulnerable = 0

    original_cwe: dict[str, int] = defaultdict(int)
    invalid_cwe_filtered: dict[str, int] = defaultdict(int)
    unique_projects = defaultdict(int)
    invalid_entries: list[RawSample] = []

    with open(file=input_path, mode="r", encoding="utf-8") as f:
        for line_num, line in rich_progress(
            enumerate(f, 1),
            total=count_jsonl_lines(input_path),
            description="Loading samples",
        ):
            line = line.strip()
            if not line:
                continue

            try:
                entry: RawSample = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num} due to JSON error: {e}")
                continue

            original_total += 1
            target = entry["target"]

            if target == 0:
                original_safe += 1
                safe_entries.append(entry)
            elif target == 1:
                for cwe in entry["cwe"]:
                    original_cwe[cwe] += 1

                original_vulnerable += 1
                if is_valid_example(example=entry):
                    valid_vulnerable_entries.append(entry)
                else:
                    invalid_entries.append(entry)
                    for cwe in entry["cwe"]:
                        invalid_cwe_filtered[cwe] += 1
            else:
                logger.warning(f"Line {line_num} has unexpected target value: {target}")

    logger.info(f"  Total samples loaded: {original_total:,}")
    logger.info(f"  Safe samples: {original_safe:,}")
    logger.info(f"  Vulnerable samples: {original_vulnerable:,}")
    logger.info(f"  Valid vulnerable samples: {len(valid_vulnerable_entries):,}")
    logger.info(f"  Invalid vulnerable samples: {len(invalid_entries):,}")
    logger.info(f"  Original CWE IDs: {len(original_cwe)}")
    logger.info(f"  Invalid CWE IDs: {invalid_cwes}")


    # print(f"Invalid cwe filtered: {invalid_cwe_filtered}")

    # Pass 2: Iterative frequency-based filtering
    logger.info(
        f"\nPass 2: Iterative frequency filtering (threshold > {frequency_threshold})..."
    )

    (
        kept_vulnerable,
        filtered_vulnerable,
        initial_frequencies,
        final_frequencies,
        iteration_log,
    ) = iterative_frequency_filter(
        entries=valid_vulnerable_entries,  # type: ignore
        frequency_threshold=frequency_threshold,
        max_iterations=20,
    )

    # Build final filtered entries list
    logger.info("\nPass 3: Building final dataset...")

    filtered_entries: list[FilteredSample] = []

    # Process safe entries
    with rich_status(description="Processing safe samples..."):
        for entry in safe_entries:
            entry["cwe"] = []
            entry["cwe_desc"] = []
            remove_unwanted_fields(entry)
            filtered_entries.append(entry)  # type: ignore

            unique_projects[entry["project"]] += 1

    # Process kept vulnerable entries
    cwes_kept: dict[str, dict] = defaultdict(lambda: {"count": 0, "description": ""})

    multi_cwe = 0

    with rich_status(description="Processing vulnerable samples..."):
        for entry in kept_vulnerable:  # type: ignore
            cwe_list = entry["cwe"]
            cwe_desc_list = entry["cwe_desc"]
            unique_projects[entry["project"]] += 1
            if len(entry["cwe"]) > 1:
                multi_cwe += 1

            for idx, cwe in enumerate(cwe_list):
                cwes_kept[cwe]["count"] += 1
                if not cwes_kept[cwe]["description"] and idx < len(cwe_desc_list):
                    cwes_kept[cwe]["description"] = cwe_desc_list[idx]

            remove_unwanted_fields(entry)
            filtered_entries.append(entry)  # type: ignore

    # Build stats for filtered CWEs (low frequency)
    cwes_filtered_low_freq: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "description": ""}
    )

    for entry in filtered_vulnerable:  # type: ignore
        cwe_list = entry["cwe"]
        cwe_desc_list = entry.get("cwe_desc", [])

        for idx, cwe in enumerate(cwe_list):
            cwes_filtered_low_freq[cwe]["count"] += 1
            if not cwes_filtered_low_freq[cwe]["description"] and idx < len(
                cwe_desc_list
            ):
                cwes_filtered_low_freq[cwe]["description"] = cwe_desc_list[idx]

    filtered_safe = len(safe_entries)
    filtered_vulnerable_count = len(kept_vulnerable)
    filtered_total = filtered_safe + filtered_vulnerable_count

    # Write filtered output
    with (
        open(file=output_path, mode="w", encoding="utf-8") as f,
        stateless_progress(
            description=f"📂 Saving filtered dataset to {output_path}"
        ) as status,
    ):
        for filtered_entry in filtered_entries:
            f.write(json.dumps(filtered_entry, ensure_ascii=False) + "\n")
        status.stop()

    # Write stats
    with stateless_progress(description="🎒 Saving CSV stats files") as status:
        stats_dir = Path(__file__).parent / "stats"
        status.update("Saving kept statistics")
        save_cwes_to_csv(cwes_kept, stats_dir, "cwes_kept.csv")
        status.update("Saving rejected statistics")
        save_cwes_to_csv(cwes_filtered_low_freq, stats_dir, "cwes_filtered.csv")
        status.stop()

    # Display results
    rich_rule(title="[green4]FILTERING COMPLETE[/green4]", style=RichColors.GREEN4)

    original_stats = {
        "Total samples": f"{original_total:,}",
        "Safe (target=0)": f"{original_safe:,}",
        "Vulnerable (target=1)": f"{original_vulnerable:,}",
    }
    filtered_stats = {
        "Total samples": f"{filtered_total:,}",
        "Safe (target=0)": f"{filtered_safe:,}",
        "Vulnerable (target=1)": f"{filtered_vulnerable_count:,}",
    }

    original_table = build_table(
        data=original_stats, title="Original Dataset", columns=["Stat", "Value"]
    )
    filtered_table = build_table(
        data=filtered_stats, title="Filtered Dataset", columns=["Stat", "Value"]
    )
    p1 = build_panel(
        renderable=[original_table, filtered_table],
        panel_title=f"Frequency threshold: > {frequency_threshold}",
        border_style=RichColors.CORNFLOWER_BLUE,
    )

    removed_vulnerable = original_vulnerable - filtered_vulnerable_count
    text = f"Vulnerable samples removed: {removed_vulnerable:,}\n"

    initially_invalid = len(invalid_entries)
    low_freq_filtered = len(filtered_vulnerable)

    text += f"  - Invalid CWE (empty or bad desc): {initially_invalid:,}\n"
    text += (
        f"  - Low frequency CWE (≤{frequency_threshold}):    {low_freq_filtered:,}\n"
    )
    text += f"\nIterations to converge: {len(iteration_log)}"

    p2 = build_panel(text, panel_title="Result", border_style=RichColors.GREEN_YELLOW)
    rich_panels_grid([p1, p2], grid_shape=(1, 2))

    # Iteration log
    if len(iteration_log) > 1:
        print("\n" + "-" * 70)
        print("ITERATION LOG")
        print("-" * 70)
        for log in iteration_log:
            print(
                f"  Iteration {log['iteration']}: "
                f"{log['samples_before']:,} → {log['samples_after']:,} samples "
                f"(-{log['samples_removed']:,} samples, -{log['cwes_below_threshold']:,} CWEs), "
                f"CWEs: {log['cwes_above_threshold']}/{log['cwes_above_threshold'] + log['cwes_below_threshold']}"
            )

    # CWEs filtered out - Invalid
    print("\n" + "-" * 70)
    print("CWEs FILTERED OUT - Invalid (empty or 'Description not found.')")
    print("-" * 70)
    if invalid_cwe_filtered:
        for cwe in sorted(invalid_cwe_filtered.keys()):
            print(f"  {cwe}: {invalid_cwe_filtered[cwe]:,}")
        print(f"\n  Total occurrences: {sum(invalid_cwe_filtered.values()):,}")
        print(f"  Unique CWEs: {len(invalid_cwe_filtered)}")
    else:
        print("  None")

    # CWEs filtered out - Low frequency
    print("\n" + "-" * 70)
    print(f"CWEs FILTERED OUT - Low Frequency (≤{frequency_threshold})")
    print("-" * 70)
    if cwes_filtered_low_freq:
        for cwe in sorted(cwes_filtered_low_freq.keys()):
            initial_freq = initial_frequencies.get(cwe, 0)
            print(
                f"  {cwe}: {cwes_filtered_low_freq[cwe]['count']:,} "
                f"(initial freq: {initial_freq})"
            )
        print(
            f"\n  Total occurrences filtered: {get_total_count(cwes_filtered_low_freq):,}"
        )
        print(f"  Unique CWEs filtered: {len(cwes_filtered_low_freq)}")
    else:
        print("  None")

    # CWEs kept
    print("\n" + "-" * 70)
    print(f"CWEs KEPT (final frequency > {frequency_threshold})")
    print("-" * 70)
    if cwes_kept:
        for cwe in sorted(cwes_kept.keys()):
            initial_freq = initial_frequencies.get(cwe, 0)
            final_freq = final_frequencies.get(cwe, 0)
            print(
                f"  {cwe}: {cwes_kept[cwe]['count']:,} "
                f"(initial: {initial_freq}, final: {final_freq})"
            )
        print(f"\n  Total kept CWE occurrences: {get_total_count(cwes_kept):,}")
        print(f"  Unique CWEs kept: {len(cwes_kept)}")
        print(f" Unique projects: {len(unique_projects)}")
        print(f" Multi cwe: {multi_cwe}")

        # Verify final counts
        print("\n  Verification:")
        all_above = all(
            final_frequencies.get(cwe, 0) > frequency_threshold
            for cwe in cwes_kept.keys()
        )
        if all_above:
            print(f"  ✅ All kept CWEs have final frequency > {frequency_threshold}")
        else:
            print(f"  ❌ Some CWEs are below threshold!")
    else:
        print("  None")

    rich_rule(style=RichColors.GREEN4)
    print(f"\nOutput written to: {output_path}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.dataset.filter.main <input.jsonl> [output.jsonl] [frequency_threshold]"
        )
        print("If output path is not specified, it defaults to <input>_filtered.jsonl")
        print("If frequency_threshold is not specified, it defaults to 50")
        print("\nExample:")
        print("  python filter_vulnerabilities.py data.jsonl filtered.jsonl 50")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_filtered{input_p.suffix}")

    if len(sys.argv) >= 4:
        try:
            frequency_threshold = int(sys.argv[3])
        except ValueError:
            print(f"Error: frequency_threshold must be an integer, got: {sys.argv[3]}")
            sys.exit(1)
    else:
        frequency_threshold = 50

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    setup_logger()
    filter_dataset(input_path, output_path, frequency_threshold)


if __name__ == "__main__":
    main()

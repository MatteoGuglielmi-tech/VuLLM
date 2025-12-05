import jsonlines
import logging

from collections import Counter
from pathlib import Path
from dataclasses import dataclass

from ..utilities import (
    rich_progress_manual,
    ensure_dir,
    validate_jsonl,
    require_file,
    count_total_lines,
    read_and_parse_lines,
)
from ..datatypes import Sample, CWEId, UniqueCWEs

logger = logging.getLogger(__name__)


@dataclass
class Statistics:
    threshold: int = 0
    support: int = 0
    samples_kept: int = 0
    safe_samples_kept: int = 0
    removed_rare_only: int = 0
    removed_mixed: int = 0
    removed_vul_empty_cwes: int = 0
    kept_all_frequent: int = 0

    @property
    def total_removed(self) -> int:
        return self.removed_rare_only + \
            self.removed_mixed + \
            self.removed_vul_empty_cwes


class CWEDatasetFilter:
    @require_file("jsonl_path")
    def __init__(self, jsonl_path: Path):
        self.input_fp: Path = jsonl_path
        self.cwe_counts: Counter = Counter()
        self.total_samples: int = 0
        self.nb_samples = count_total_lines(jsonl_path)

    def _extract_cwe_id(self, cwe_string: str) -> CWEId:
        """
        Extract numeric CWE ID from string like 'CWE-476'
        Returns: 476
        """
        return int(cwe_string.replace("CWE-", ""))


    def count_cwe_frequencies(self) -> Counter:
        """First pass: Count CWE occurrences"""

        line_generator = read_and_parse_lines(self.input_fp)
        errors: int = 0

        with rich_progress_manual(
            total=self.nb_samples, description="Counting CWEs"
        ) as pbar:
            for line_num, sample in enumerate(line_generator, start=1):
                self.total_samples += 1

                try:
                    for cwe_str in sample.get("cwe", []):
                        cwe_id: int = self._extract_cwe_id(cwe_str)
                        self.cwe_counts[cwe_id] += 1  # register

                except Exception:
                    errors += 1

                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix(
                        {
                            "✓ Processed": f"{line_num}/{self.nb_samples} lines",
                            "✗ Errors": f"{errors}",
                            "✗ Error rate": (
                                f"{errors/self.total_samples:.1%}" if bool(errors) else "0.0%"
                            ),
                        }
                    )

        logger.info(f"Total samples: {self.total_samples:,}")
        logger.info(f"Unique CWEs: {len(self.cwe_counts)}")

        return self.cwe_counts

    def get_frequent_cwes(self, min_threshold: int=100) -> UniqueCWEs:
        """
        Get set of CWE IDs that meet the minimum threshold

        Paramters
        ---------
        min_threshold: int, optional (default=100)
            Minimum number of samples a CWE must have to be considered frequent

        Returns
        -------
        UniqueCWEs
            Set of CWE IDs that appear in at least min_threshold samples
        """

        frequent = {
            cwe for cwe, count in self.cwe_counts.items() if count >= min_threshold
        }

        logger.info(
            f"CWEs with ≥ {min_threshold} samples: {len(frequent)}/{len(self.cwe_counts)}"
        )

        return frequent

    def preview_filtering_impact(self, min_threshold: int = 100) -> dict:
        """Preview how many samples would be kept with given threshold
        WITHOUT actually filtering (fast check)

        Returns:
            Statistics about what would be kept/removed
        """
        frequent_cwes = self.get_frequent_cwes(min_threshold)

        stats = {
            "threshold": min_threshold,
            "frequent_cwes": len(frequent_cwes),
            "samples_kept": 0,
            "safe_kept": 0,
            "vulnerable_kept": 0,
            "samples_removed": 0,
            "removed_rare_only": 0,
            "removed_mixed": 0,
        }

        line_generator = read_and_parse_lines(self.input_fp)

        with rich_progress_manual(
            total=self.nb_samples,
            description=f"Preview filtering with threshold={min_threshold}",
        ) as pbar:
            for sample in line_generator:
                cwes = sample.get("cwe", [])
                target = sample.get("target", 1)

                try:
                    # do not check cwe: sometimes present even
                    # in case of target=0
                    if target == 0:
                        stats["samples_kept"] += 1
                        stats["safe_kept"] += 1
                        # continue # here commented to avoid duplication of pbar update

                    if target == 1 and cwes:  # vulnerable
                        cwe_ids: UniqueCWEs = {
                            self._extract_cwe_id(cwe) for cwe in cwes
                        }
                        rare_cwes: UniqueCWEs = cwe_ids - frequent_cwes

                        if not rare_cwes:
                            stats["samples_kept"] += 1
                            stats["vulnerable_kept"] += 1
                        else:
                            # Has rare CWEs - would remove
                            stats["samples_removed"] += 1
                            if cwe_ids & frequent_cwes:
                                stats["removed_mixed"] += 1
                            else:
                                stats["removed_rare_only"] += 1
                finally:
                    pbar.update(advance=1)
                    pbar.set_postfix(
                        {
                            "✓ Kept": f"{stats["samples_kept"]}",
                            "✓ Kept safe": f"{stats["safe_kept"]}",
                            "⚠ Kept vul": f"{stats["vulnerable_kept"]}",
                            "✗ Removed": f"{stats["samples_removed"]}",
                            "✗ Removed rare": f"{stats["removed_rare_only"]}",
                            "✗ Removed mixed": f"{stats["removed_mixed"]}",
                        }
                    )

        total = stats["samples_kept"] + stats["samples_removed"]
        print("\n" + "=" * 60)
        print(f"FILTERING PREVIEW (threshold={min_threshold})")
        print("=" * 60)
        print(f"Total samples:              {total:>10,}")
        print(f"Frequent CWEs:              {stats['frequent_cwes']:>10}")
        print()
        print(
            f"Would KEEP:                 {stats['samples_kept']:>10,}\
            ({stats['samples_kept']/total*100:.1f}%)"
        )
        print(f"  Safe samples:             {stats['safe_kept']:>10,}")
        print(f"  Vulnerable (all freq):    {stats['vulnerable_kept']:>10,}")
        print()
        print(
            f"Would REMOVE:               {stats['samples_removed']:>10,}\
            ({stats['samples_removed']/total*100:.1f}%)"
        )
        print(f"  Rare CWEs only:           {stats['removed_rare_only']:>10,}")
        print(f"  Mixed (freq + rare):      {stats['removed_mixed']:>10,}")
        print("=" * 60)

        return stats

    @validate_jsonl(target_param_name="output_fp")
    def filter_dataset(
        self, output_fp: Path, min_threshold: int = 100, save_interval=200
    ) -> Statistics:
        """
        Second pass: Filter dataset and write to new JSONL file
        Only keeps samples where ALL CWEs appear in at least min_threshold samples

        Parameters
        ----------
            output_fp: Output file path
            min_threshold: Minimum number of samples a CWE must have to be kept

        Returns
        -------
            Statistics about the filtering
        """

        def _update_progress(advance: int | None = None) -> None:
            pbar.update(advance=len(samples_batch) if not advance else advance)
            pbar.set_postfix(
                {
                    "✓ Kept": f"{stats.samples_kept}",
                    "✓ Kept safe": f"{stats.safe_samples_kept}",
                    "✓ Kept vul": f"{stats.kept_all_frequent}",
                    "✗ Removed": f"{stats.total_removed}",
                    "✗ Removed rare": f"{stats.removed_rare_only}",
                    "✗ Removed mixed": f"{stats.removed_mixed}",
                }
            )

        frequent_cwes = self.get_frequent_cwes(min_threshold=min_threshold)

        stats = Statistics(threshold=min_threshold)

        logger.info(f"Filtering dataset (min_threshold: {min_threshold})...")
        logger.info(f"Frequent CWEs (≥{min_threshold}): {len(frequent_cwes)}")
        logger.info(f"Writing to: {output_fp}")

        line_generator = read_and_parse_lines(self.input_fp)

        samples_batch: list[Sample] = []
        with (
            jsonlines.open(file=output_fp, mode="w") as writer,
            rich_progress_manual(
                total=self.nb_samples, description="Filtering dataset"
            ) as pbar,
        ):
            for sample in line_generator:
                stats.support += 1

                cwes = sample["cwe"]
                target = sample["target"]

                # keep safe samples (target=0)
                if target == 0:
                    stats.samples_kept += 1
                    stats.safe_samples_kept += 1
                    samples_batch.append(sample)

                    if len(samples_batch) % save_interval == 0:
                        writer.write_all(samples_batch)
                        _update_progress()
                        samples_batch.clear()
                    continue

                # vulnerable samples
                if target == 1:
                    # if cwes
                    if cwes and len(cwes) == 1:
                        # check CWE frequency
                        cwe_ids = {self._extract_cwe_id(cwe) for cwe in cwes}
                        nb_rare_cwes = cwe_ids - frequent_cwes

                        if not nb_rare_cwes:
                            # All CWEs are frequent - KEEP
                            stats.samples_kept += 1
                            stats.kept_all_frequent += 1

                            samples_batch.append(sample)
                            if len(samples_batch) % save_interval == 0:
                                writer.write_all(samples_batch)
                                _update_progress()
                                samples_batch.clear()
                        else:
                            # Has rare CWEs - REMOVE
                            if cwe_ids & frequent_cwes:
                                # Mixed: has both frequent and rare
                                stats.removed_mixed += 1
                            else:
                                # Only rare CWEs
                                stats.removed_rare_only += 1

                            _update_progress(advance=1)
                    # if not cwes, skip it
                    if not cwes:
                        stats.removed_vul_empty_cwes += 1
                        _update_progress(advance=1)

            if samples_batch:
                writer.write_all(samples_batch)
                _update_progress(advance=len(samples_batch))

        return stats

    def _print_filtering_stats(self, stats: Statistics, frequent_cwes: UniqueCWEs):
        """Print detailed filtering statistics"""

        print("\n" + "="*60)
        print("FILTERING RESULTS")
        print("="*60)
        print(f"Threshold:                  {stats.threshold:>10}")
        print(f"Input samples:              {stats.support:>10,}")
        print(f"Output samples:             {stats.samples_kept:>10,} ({stats.samples_kept/stats.support*100:.1f}%)")
        print()
        print("Kept samples breakdown:")
        print(f"  Safe (target=0):          {stats.safe_samples_kept:>10,}")
        print(f"  All CWEs frequent:        {stats.kept_all_frequent:>10,}")
        print()
        print("Removed samples:")
        print(f"  Rare CWEs only:           {stats.removed_rare_only:>10,}")
        print(f"  Mixed (freq + rare):      {stats.removed_mixed:>10,}")
        print(f"  Total removed:            {stats.support - stats.samples_kept:>10,}")
        print()
        print(f"Frequent CWEs kept:         {len(frequent_cwes):>10}")
        print("="*60)

    def analyze_cwe_distribution(
        self, top_n: int = 20, bottom_n: int = 10, highlight_threshold: int = 100
    ):
        """
        Print CWE distribution analysis

        Parameters
        ----------
        top_n: int, default=20
            Number of top CWEs to show
        bottom_n: int, defualt=10
            Number of bottom CWEs to show
        highlight_threshold: int, default=100
            Highlight CWEs above this threshold
        """
        print("\n" + "=" * 60)
        print("CWE DISTRIBUTION ANALYSIS")
        print("=" * 60)

        # Count frequent vs rare
        frequent_count = sum(
            1 for count in self.cwe_counts.values() if count >= highlight_threshold
        )
        rare_count = len(self.cwe_counts) - frequent_count

        print(f"\nTotal CWEs: {len(self.cwe_counts)}")
        print(f"Frequent (≥{highlight_threshold}): {frequent_count}")
        print(f"Rare (<{highlight_threshold}): {rare_count}")

        print(f"\nTop {top_n} most common CWEs:")
        print(f"{'CWE ID':<10} | {'Count':<10} | {'Percentage':<10} | {'Status'}")
        print("-" * 55)
        for cwe_id, count in self.cwe_counts.most_common(top_n):
            pct = count / self.total_samples * 100
            status = "✓ KEEP" if count >= highlight_threshold else "✗ REMOVE"
            print(f"CWE-{cwe_id:<5} | {count:<10,} | {pct:>6.2f}%    | {status}")

        if len(self.cwe_counts) > top_n:
            print(f"\nBottom {bottom_n} rarest CWEs:")
            print(f"{'CWE ID':<10} | {'Count':<10} | {'Percentage':<10} | {'Status'}")
            print("-" * 55)
            for cwe_id, count in list(self.cwe_counts.most_common())[-bottom_n:]:
                pct = count / self.total_samples * 100
                status = "✓ KEEP" if count >= highlight_threshold else "✗ REMOVE"
                print(f"CWE-{cwe_id:<5} | {count:<10,} | {pct:>6.2f}%    | {status}")

        print("=" * 60)

    @ensure_dir(target_param_name="output_dir")
    @validate_jsonl(target_param_name="filename")
    def exec(self, output_dir: Path, filename: Path) -> None:
        """
        Example usage with fixed minimum threshold
        """

        _ = self.count_cwe_frequencies()
        self.analyze_cwe_distribution(top_n=20, bottom_n=10, highlight_threshold=100)
        self.preview_filtering_impact(min_threshold=100)

        response = input("\nProceed with filtering? (y/n): ")
        if response.lower() == "y":
            self.filter_dataset(
                output_fp=output_dir / filename, min_threshold=100
            )
        else:
            logger.info("Filtering cancelled.")


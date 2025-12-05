"""
Balanced Sampling Pipeline with Fixed Samples Per CWE

This module performs balanced sampling on vulnerability datasets:
1. Filter CWEs with insufficient samples
2. Select exactly N samples per CWE (median-centered by token_count)
3. Match non-vulnerable samples by complexity/token distribution

Result: Perfectly CWE-balanced dataset with distribution-matched safe functions.
"""

import argparse
import logging
import pandas as pd
import numpy as np
import dataframe_image as dfi
import json

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast
from collections import Counter
from .logging_config import setup_logger


JsonlEntry: TypeAlias = dict[str, Any]

setup_logger()
logger = logging.getLogger(name=__name__)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="Balance dataset",
        description="Balanced sampling with fixed samples per CWE"
    )
    parser.add_argument(
        "--input_fp",
        "-i",
        type=str,
        required=True,
        help="Absolute path to the input dataset."
    )
    parser.add_argument(
        "--output_fp",
        "-o",
        type=str,
        required=True,
        help="Absolute path to the output location.",
    )
    parser.add_argument(
        "--min_samples_per_cwe",
        type=int,
        default=100,
        help="Minimum samples required for a CWE to be included (default: 100)"
    )
    parser.add_argument(
        "--target_samples_per_cwe",
        type=int,
        default=None,
        help="Target samples per CWE. If None, uses min_samples_per_cwe (default: None)"
    )
    parser.add_argument(
        "--export_stats",
        action="store_true",
        help="Export detailed CWE distribution statistics"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug CLI logs",
    )
    return parser


@dataclass
class CWEDistributionStats:
    """Statistics about CWE distribution."""

    total_cwes: int = 0
    total_samples: int = 0
    total_cwe_exposure: int = 0
    cwe_counts: dict[str, int] = field(default_factory=dict)
    gini_coefficient: float = 0.0
    entropy: float = 0.0

    def compute_metrics(self):
        """Compute distribution metrics from cwe_counts."""
        if not self.cwe_counts:
            return

        counts = np.array(list(self.cwe_counts.values()))
        self.total_cwe_exposure = int(counts.sum())
        self.total_cwes = len(counts)

        if self.total_cwe_exposure == 0:
            return

        # Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        self.gini_coefficient = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

        # Shannon entropy (higher = more balanced)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        self.entropy = -np.sum(probs * np.log2(probs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cwes": self.total_cwes,
            "total_samples": self.total_samples,
            "total_cwe_exposure": self.total_cwe_exposure,
            "gini_coefficient": round(self.gini_coefficient, 4),
            "entropy": round(self.entropy, 4),
            "cwe_counts": self.cwe_counts,
        }

    def summary(self) -> str:
        return (
            f"  Unique CWEs: {self.total_cwes}\n"
            f"  Vulnerable samples: {self.total_samples:,}\n"
            f"  Total CWE exposure: {self.total_cwe_exposure:,}\n"
            f"  Gini coefficient: {self.gini_coefficient:.4f} (0=equal, 1=unequal)\n"
            f"  Shannon Entropy: {self.entropy:.4f} bits"
        )


@dataclass 
class Balancer:
    """
    Balanced sampler with fixed samples per CWE.

    Pipeline:
    1. Filter CWEs below minimum threshold
    2. Select N samples per CWE (median-centered by token_count)
    3. Match non-vulnerable samples by complexity/token bins
    """

    input_fp: Path
    output_fp: Path

    min_samples_per_cwe: int = 100
    target_samples_per_cwe: int | None = None

    export_stats: bool = True

    df: pd.DataFrame = field(init=False)
    df_balanced: pd.DataFrame = field(init=False)
    stats_before: CWEDistributionStats = field(
        init=False, default_factory=CWEDistributionStats
    )
    stats_after: CWEDistributionStats = field(
        init=False, default_factory=CWEDistributionStats
    )

    random_state: int = 42
    n_bins: int = 90

    lower_quantile: float = 0.05
    upper_quantile: float = 0.99

    def __post_init__(self):
        self.target_samples_per_cwe = (
            self.min_samples_per_cwe
            if self.target_samples_per_cwe is None
            else self.target_samples_per_cwe
        )

    def read_jsonl_as_df(self):
        """Load JSONL dataset into DataFrame."""
        self.df = pd.read_json(self.input_fp, lines=True, encoding="utf-8")
        logger.info(f"Loaded {len(self.df):,} samples from {self.input_fp}")

    def save_df_to_jsonl(self, df: pd.DataFrame):
        """Save DataFrame to JSONL format."""
        if df.empty:
            raise ValueError("Passed empty dataframe to save. Abort")
        df.to_json(self.output_fp, orient="records", lines=True, force_ascii=False)
        logger.info(f"DataFrame saved to {self.output_fp}")

    def _normalize_cwe_list(self, cwe_field: list[str] | str) -> list[str]:
        """Normalize CWE field to a list of CWE identifiers."""
        if isinstance(cwe_field, list):
            return [cwe for cwe in cwe_field if cwe]
        elif isinstance(cwe_field, str) and cwe_field:
            return [cwe_field]
        return []

    def _compute_cwe_counts(self, df_vuln: pd.DataFrame) -> dict[str, int]:
        """
        Compute CWE occurrence counts using full attribution.
        Each sample contributes to ALL its CWEs."""
        cwe_counts: Counter[str] = Counter()
        for cwe_list in df_vuln["cwe"]:
            for cwe in self._normalize_cwe_list(cwe_list):
                cwe_counts[cwe] += 1
        return dict(cwe_counts)

    def _compute_cwe_stats(self, df: pd.DataFrame) -> CWEDistributionStats:
        """Compute CWE distribution statistics for vulnerable samples."""
        stats = CWEDistributionStats()

        vuln_df = df[df["target"] == 1]
        stats.total_samples = len(vuln_df)
        stats.cwe_counts = self._compute_cwe_counts(cast(pd.DataFrame, vuln_df))
        stats.compute_metrics()

        return stats

    def _get_samples_for_cwe(self, df_vuln: pd.DataFrame, cwe: str) -> pd.DataFrame:
        """Get all samples that contain the specified CWE."""
        mask = df_vuln["cwe"].apply(
            lambda cwe_list: cwe in self._normalize_cwe_list(cwe_list)
        )
        return cast(pd.DataFrame, df_vuln[mask])

    def _select_median_centered_samples(
        self, cwe_samples: pd.DataFrame, n: int
    ) -> pd.DataFrame:
        """
        Select N samples centered around the median token_count.

        Sorts by token_count, finds median position, selects N/2 before and N/2 after.
        This ensures representative samples (not outliers).

        Args:
            cwe_samples: DataFrame of samples for a specific CWE
            n: Number of samples to select

        Returns:
            DataFrame of selected samples
        """
        if len(cwe_samples) <= n:
            return cwe_samples

        sorted_samples = cwe_samples.sort_values("token_count").reset_index(drop=False)
        sorted_samples = sorted_samples.rename(columns={"index": "original_index"})

        median_pos = len(sorted_samples) // 2

        # Select N/2 before and N/2 after median
        half_n = n // 2
        start_pos = max(0, median_pos - half_n)
        end_pos = start_pos + n

        if end_pos > len(sorted_samples):
            end_pos = len(sorted_samples)
            start_pos = max(0, end_pos - n)

        selected = sorted_samples.iloc[start_pos:end_pos]

        return cwe_samples.loc[selected["original_index"]]

    def _select_vulnerable_samples(self, df_vuln: pd.DataFrame) -> pd.DataFrame:
        """
        Select samples to achieve target_samples_per_cwe for each qualifying CWE.

        Strategy (Full Attribution):
        1. Compute CWE counts and filter those below threshold
        2. For each CWE independently, identify the best N candidates (median-centered)
        3. Union all candidates - a multi-label sample selected for any of its CWEs
           will contribute to ALL its CWEs in the final count
        4. Return unique samples

        This ensures each CWE gets close to its target while multi-label samples
        efficiently serve multiple CWEs.

        Returns:
            DataFrame of selected vulnerable samples
        """
        cwe_counts = self._compute_cwe_counts(df_vuln)

        qualifying_cwes = {
            cwe: count
            for cwe, count in cwe_counts.items()
            if count >= self.min_samples_per_cwe
        }

        logger.info(
            f"CWEs with >= {self.min_samples_per_cwe} samples: {len(qualifying_cwes)}"
        )
        logger.info(f"CWEs filtered out: {len(cwe_counts) - len(qualifying_cwes)}")

        if not qualifying_cwes:
            raise ValueError(
                f"No CWEs have >= {self.min_samples_per_cwe} samples. "
                f"Maximum available: {max(cwe_counts.values()) if cwe_counts else 0}"
            )

        logger.info(f"\nSelecting {self.target_samples_per_cwe} samples per CWE:")

        cwe_candidate_indices: dict[str, set[Any]] = {}
        for cwe, original_count in sorted(qualifying_cwes.items(), key=lambda x: -x[1]):
            cwe_samples = self._get_samples_for_cwe(df_vuln, cwe)

            # Select median-centered samples
            n_to_select = min(self.target_samples_per_cwe, len(cwe_samples))
            selected = self._select_median_centered_samples(cwe_samples, n_to_select)

            cwe_candidate_indices[cwe] = set(selected.index)
            logger.debug(
                f"  {cwe}: {len(selected)} candidates from {original_count} samples"
            )

        all_selected_indices: set[Any] = set()
        for indices in cwe_candidate_indices.values():
            all_selected_indices.update(indices)

        df_selected = df_vuln.loc[list(all_selected_indices)]

        final_cwe_counts = self._compute_cwe_counts(df_selected)

        logger.info(f"\nCWE selection results:")
        for cwe in sorted(qualifying_cwes.keys()):
            target = self.target_samples_per_cwe
            actual = final_cwe_counts.get(cwe, 0)
            status = "✓" if actual >= target else f"({actual}/{target})"
            logger.info(f"  {cwe}: {actual} {status}")

        logger.info(
            f"\nTotal unique vulnerable samples selected: {len(all_selected_indices)}"
        )

        shortfalls = {
            cwe: (self.target_samples_per_cwe, final_cwe_counts.get(cwe, 0))
            for cwe in qualifying_cwes
            if final_cwe_counts.get(cwe, 0) < self.target_samples_per_cwe
        }
        if shortfalls:
            logger.warning(f"\n{len(shortfalls)} CWEs below target:")
            for cwe, (target, actual) in shortfalls.items():
                logger.warning(f"  {cwe}: {actual}/{target}")

        return df_selected

    def _stratified_sample_non_vulnerable(
        self, df_vuln: pd.DataFrame, df_non_vuln: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sample non-vulnerable functions matching the complexity/token distribution
        of vulnerable functions.
        """

        combined_for_bins = pd.concat(
            [
                df_vuln[["cyclomatic_complexity", "token_count"]],
                df_non_vuln[["cyclomatic_complexity", "token_count"]],
            ]
        )

        try:
            _, complexity_bins = pd.qcut(
                combined_for_bins["cyclomatic_complexity"],
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            _, token_bins = pd.qcut(
                combined_for_bins["token_count"],
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
        except ValueError as e:
            logger.warning(f"Could not create {self.n_bins} bins, reducing: {e}")
            # Fallback to fewer bins
            _, complexity_bins = pd.qcut(
                combined_for_bins["cyclomatic_complexity"],
                q=20,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            _, token_bins = pd.qcut(
                combined_for_bins["token_count"],
                q=20,
                labels=False,
                retbins=True,
                duplicates="drop",
            )

        complexity_bins = np.unique(complexity_bins)  # type: ignore
        token_bins = np.unique(token_bins)  # type: ignore

        # Apply bins
        df_vuln = df_vuln.copy()
        df_non_vuln = df_non_vuln.copy()

        df_vuln["complexity_bin"] = pd.cut(
            df_vuln["cyclomatic_complexity"],
            bins=complexity_bins,
            labels=False,
            include_lowest=True,
        )
        df_vuln["token_bin"] = pd.cut(
            df_vuln["token_count"],
            bins=token_bins,
            labels=False,
            include_lowest=True,
        )

        df_non_vuln["complexity_bin"] = pd.cut(
            df_non_vuln["cyclomatic_complexity"],
            bins=complexity_bins,
            labels=False,
            include_lowest=True,
        )
        df_non_vuln["token_bin"] = pd.cut(
            df_non_vuln["token_count"],
            bins=token_bins,
            labels=False,
            include_lowest=True,
        )

        # Drop NaN bins
        df_vuln = df_vuln.dropna(subset=["complexity_bin", "token_bin"])
        df_non_vuln = df_non_vuln.dropna(subset=["complexity_bin", "token_bin"])

        df_vuln["complexity_bin"] = df_vuln["complexity_bin"].astype(int)
        df_vuln["token_bin"] = df_vuln["token_bin"].astype(int)
        df_non_vuln["complexity_bin"] = df_non_vuln["complexity_bin"].astype(int)
        df_non_vuln["token_bin"] = df_non_vuln["token_bin"].astype(int)

        vuln_bin_counts = df_vuln.groupby(["complexity_bin", "token_bin"]).size()

        sampled_indices: list[Any] = []

        for (c_bin, t_bin), count in vuln_bin_counts.items():  # type: ignore
            non_vuln_in_bin = df_non_vuln[
                (df_non_vuln["complexity_bin"] == c_bin)
                & (df_non_vuln["token_bin"] == t_bin)
            ]

            num_to_sample = min(int(count), len(non_vuln_in_bin))
            if num_to_sample > 0:
                sampled = non_vuln_in_bin.sample(
                    n=num_to_sample, random_state=self.random_state
                )
                sampled_indices.extend(sampled.index.tolist())

        df_non_vuln_sampled = df_non_vuln.loc[sampled_indices]

        df_vuln = df_vuln.drop(columns=["complexity_bin", "token_bin"])
        df_non_vuln_sampled = df_non_vuln_sampled.drop(
            columns=["complexity_bin", "token_bin"]
        )

        return df_vuln, df_non_vuln_sampled

    def _run_pipeline(self):
        """Execute the full sampling pipeline."""
        assets_dir = self.output_fp.parent

        logger.info("=" * 60)
        logger.info("BALANCED SAMPLING WITH FIXED SAMPLES PER CWE")
        logger.info(f"Min samples per CWE: {self.min_samples_per_cwe}")
        logger.info(f"Target samples per CWE: {self.target_samples_per_cwe}")
        logger.info("=" * 60)

        # --- Compute initial stats ---
        self.stats_before = self._compute_cwe_stats(self.df)
        logger.info("\nCWE Distribution BEFORE sampling:")
        logger.info(self.stats_before.summary())

        # --- Filter by quantiles ---
        logger.info("\n[Step 1] Filtering by complexity/token quantiles...")

        min_tokens = self.df["token_count"].quantile(self.lower_quantile)
        max_tokens = self.df["token_count"].quantile(self.upper_quantile)
        min_complexity = max(
            3.0, self.df["cyclomatic_complexity"].quantile(self.lower_quantile)
        )
        max_complexity = self.df["cyclomatic_complexity"].quantile(self.upper_quantile)

        logger.info(f"Token count range: [{min_tokens:.0f}, {max_tokens:.0f}]")
        logger.info(f"Complexity range: [{min_complexity:.0f}, {max_complexity:.0f}]")

        original_count = len(self.df)
        self.df = cast(
            pd.DataFrame,
            self.df[
                (self.df["token_count"] >= min_tokens)
                & (self.df["token_count"] <= max_tokens)
                & (self.df["cyclomatic_complexity"] >= min_complexity)
                & (self.df["cyclomatic_complexity"] <= max_complexity)
            ].copy(),
        )
        logger.info(f"Removed {original_count - len(self.df):,} outlier samples")

        # --- Split by target ---
        self.df["target"] = self.df["target"].astype(int)
        df_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 1].copy())
        df_non_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 0].copy())

        logger.info(f"\nAfter filtering:")
        logger.info(f"  Vulnerable samples: {len(df_vuln):,}")
        logger.info(f"  Non-vulnerable samples: {len(df_non_vuln):,}")

        # --- Step 2: Select vulnerable samples (CWE-balanced, median-centered) ---
        logger.info(
            "\n[Step 2] Selecting vulnerable samples (median-centered per CWE)..."
        )
        df_vuln_selected = self._select_vulnerable_samples(df_vuln)

        # --- Step 3: Stratified sampling of non-vulnerable ---
        logger.info("\n[Step 3] Stratified sampling of non-vulnerable samples...")
        df_vuln_final, df_non_vuln_sampled = self._stratified_sample_non_vulnerable(
            df_vuln_selected, df_non_vuln
        )

        # --- Results ---
        num_vuln = len(df_vuln_final)
        num_non_vuln = len(df_non_vuln_sampled)

        logger.info(f"\nSampling results:")
        logger.info(f"  Vulnerable: {num_vuln:,}")
        logger.info(f"  Non-vulnerable: {num_non_vuln:,}")

        if num_vuln > 0:
            balance_ratio = num_non_vuln / num_vuln
            logger.info(f"  Balance ratio: {balance_ratio:.2%}")

        # --- Assemble final dataset ---
        self.df_balanced = pd.concat([df_vuln_final, df_non_vuln_sampled])
        self.df_balanced = self.df_balanced.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)

        # --- Compute final stats ---
        self.stats_after = self._compute_cwe_stats(self.df_balanced)

        # --- Export stats ---
        if self.export_stats:
            self._export_stats(assets_dir)

        # --- Summary ---
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLING COMPLETE")
        logger.info("=" * 60)

        logger.info(f"\nFinal dataset: {len(self.df_balanced):,} samples")
        logger.info(
            f"Target distribution:\n{self.df_balanced['target'].value_counts().to_string()}"
        )

        logger.info("\nCWE Distribution AFTER sampling:")
        logger.info(self.stats_after.summary())

        # Gini improvement
        if self.stats_before.gini_coefficient > 0:
            gini_change = (
                self.stats_after.gini_coefficient - self.stats_before.gini_coefficient
            )
            pct_change = abs(gini_change / self.stats_before.gini_coefficient) * 100
            direction = "improved" if gini_change < 0 else "worsened"
            logger.info(f"\nGini coefficient {direction} by {pct_change:.1f}%")
            logger.info(f"  Before: {self.stats_before.gini_coefficient:.4f}")
            logger.info(f"  After: {self.stats_after.gini_coefficient:.4f}")

    def _export_stats(self, assets_dir: Path):
        """Export statistics to files."""
        stats_file = assets_dir / "cwe_distribution_stats.json"

        output = {
            "settings": {
                "min_samples_per_cwe": self.min_samples_per_cwe,
                "target_samples_per_cwe": self.target_samples_per_cwe,
            },
            "before": self.stats_before.to_dict(),
            "after": self.stats_after.to_dict(),
        }

        with open(stats_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Statistics exported to {stats_file}")

        before = self.stats_before.cwe_counts
        after = self.stats_after.cwe_counts

        comparison = []
        for cwe in sorted(set(before.keys()) | set(after.keys())):
            comparison.append(
                {
                    "CWE": cwe,
                    "Before": before.get(cwe, 0),
                    "After": after.get(cwe, 0),
                }
            )

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values("After", ascending=False)
        comparison_df = comparison_df.set_index("CWE")

        comparison_df.to_csv(assets_dir / "cwe_comparison.csv")
        try:
            dfi.export(
                comparison_df.head(30),
                assets_dir / "cwe_distribution_comparison.png",
                table_conversion="matplotlib",
            )
        except Exception as e:
            logger.warning(f"Could not export comparison image: {e}")

    def process(self):
        """Run the full pipeline."""
        self.read_jsonl_as_df()
        self._run_pipeline()

        if self.df_balanced is not None and not self.df_balanced.empty:
            self.save_df_to_jsonl(self.df_balanced)
        else:
            logger.error("Pipeline resulted in empty dataset. Nothing saved.")


if __name__ == "__main__":
    args = get_parser().parse_args()

    balancer = Balancer(
        input_fp=Path(args.input_fp),
        output_fp=Path(args.output_fp),
        min_samples_per_cwe=args.min_samples_per_cwe,
        target_samples_per_cwe=args.target_samples_per_cwe,
        export_stats=args.export_stats,
    )

    balancer.process()

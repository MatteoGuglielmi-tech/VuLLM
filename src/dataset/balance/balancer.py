import argparse
import logging
import pandas as pd
import numpy as np
import dataframe_image as dfi

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast
from logging_config import setup_logger


JsonlEntry: TypeAlias = dict[str, Any]

setup_logger()
logger = logging.getLogger(name=__name__)


def get_parser():
    parser = argparse.ArgumentParser(prog="Balance dataset")
    parser.add_argument(
        "--input_fp",
        "-i",
        type=str,
        help="Absolute path to the raw dataset."
    )
    parser.add_argument(
        "--output_fp",
        "-o",
        type=str,
        help="Absolute path to the output location where to save the balanced dataset to.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Activate debug CLI logs",
    )
    return parser


@dataclass
class Balancer:

    input_fp: Path
    output_fp: Path
    df: pd.DataFrame = field(init=False)

    # defaults
    balance_tolerance: float = 0.95
    random_state: int = 42
    n_bins: int = 90  # 50 - 100

    # Define quantiles for filtering. e.g., remove bottom 5% and top 1%.
    lower_quantile = 0.05
    upper_quantile = 0.99

    def read_jsonl_as_df(self):
        self.df = pd.read_json(self.input_fp, lines=True, encoding="utf-8")

    def save_df_to_jsonl(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Passed empty dataframe to save. Abort")
        df.to_json(self.output_fp, orient="records", lines=True, force_ascii=False)

        logger.info(f"DataFrame saved to {self.output_fp}")

    def _perform_fallback_sampling(self, df_vuln: pd.DataFrame, df_non_vuln: pd.DataFrame):
        logger.warning("\n⚠️ Distribution matching failed. Using fallback.")
        num_to_sample = len(df_vuln)

        df_non_vuln_sampled = df_non_vuln.sample( n=num_to_sample, random_state=self.random_state)
        self._assemble_final_df(df_vuln, df_non_vuln_sampled)

    def _assemble_final_df(self, df_vuln: pd.DataFrame, df_non_vuln_sampled: pd.DataFrame):
        combined_df = pd.concat([df_vuln, df_non_vuln_sampled])
        # cols_to_drop = ["cyclomatic_complexity", "token_count", "complexity_bin", "token_bin"]
        # existing_cols_to_drop = [ col for col in cols_to_drop if col in combined_df.columns ]
        # self.df_balanced = cast(pd.DataFrame, combined_df.drop(columns=existing_cols_to_drop))
        # self.df_balanced = self.df_balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        self.df_balanced = combined_df.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)

    def _perform_stratified_sampling(self):
        """Performs stratified sampling to balance vulnerable and non-vulnerable functions."""

        assets_dir = self.output_fp.parent

        logger.info("Performing stratified sampling...")

        min_tokens = self.df["token_count"].quantile(self.lower_quantile)
        min_complexity = self.df["cyclomatic_complexity"].quantile(self.lower_quantile)

        max_tokens = self.df["token_count"].quantile(self.upper_quantile)
        max_complexity = self.df["cyclomatic_complexity"].quantile(self.upper_quantile)

        logger.info(f"Filtering between token counts: ({min_tokens:.1f}, {max_tokens:.1f})")
        logger.info(f"Filtering between complexities: ({min_complexity:.1f}, {max_complexity:.1f})")

        original_count = len(self.df)
        self.df = cast(pd.DataFrame, self.df[
            (self.df["token_count"] >= min_tokens) &
            (self.df["token_count"] <= max_tokens) &
            (self.df["cyclomatic_complexity"] >= min_complexity) &
            (self.df["cyclomatic_complexity"] <= max_complexity)
        ])
        logger.info(f"Removed {original_count - len(self.df)} samples. New size: {len(self.df)}")

        self.df["target"] = self.df["target"].astype("int")  # ensure type
        df_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 1].copy())
        df_non_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 0].copy())

        quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        df_vuln_stats = df_vuln[["cyclomatic_complexity", "token_count"]].describe(
            percentiles=quantiles
        )
        df_non_vuln_stats = df_non_vuln[
            ["cyclomatic_complexity", "token_count"]
        ].describe(percentiles=quantiles)

        dfi.export(
            cast(pd.DataFrame, df_vuln_stats),
            assets_dir / "vulnerable_stats_before.png" ,
            table_conversion="matplotlib",
        )
        dfi.export(
            cast(pd.DataFrame, df_non_vuln_stats),
            assets_dir / "non_vulnerable_stats_before.png",
            table_conversion="matplotlib",
        )

        logger.debug(f"Found {len(df_vuln)} vulnerable functions (target: 1).")
        logger.debug(f"Found {len(df_non_vuln)} non-vulnerable functions (target: 0).")

        try:
            # quantile cut: divide into a set number of equal-sized groups
            _, complexity_bins = pd.qcut(
                self.df["cyclomatic_complexity"],
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",  # if fails to create q bins, merge the problematic bins
            )
            _, token_bins = pd.qcut(
                self.df["token_count"],
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates="drop",
            )

            # Ensure bin edges are unique and monotonic for pd.cut, which can fail if qcut merges bins.
            complexity_bins = np.unique(cast(np.ndarray, complexity_bins))
            token_bins = np.unique(cast(np.ndarray, token_bins))

            # value cut: divides based on pre-defined bins
            # apply same bin edges to non-vulnerable data
            self.df["complexity_bin"] = pd.cut(
                self.df["cyclomatic_complexity"],
                bins=complexity_bins,
                labels=False,
                include_lowest=True,
            )
            self.df["token_bin"] = pd.cut(
                self.df["token_count"],
                bins=token_bins,
                labels=False,
                include_lowest=True,
            )

            self.df.dropna(subset=["complexity_bin", "token_bin"], inplace=True)
            self.df["complexity_bin"] = self.df["complexity_bin"].astype(int)
            self.df["token_bin"] = self.df["token_bin"].astype(int)


            df_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 1].copy())
            df_non_vuln = cast(pd.DataFrame, self.df[self.df["target"] == 0].copy())

            # count targets in each bin
            target_counts = df_vuln.groupby(["complexity_bin", "token_bin"]).size()

            sampled_indices = []
            for (c_bin, t_bin), count in target_counts.items():  # type: ignore
                non_vuln_in_bin = df_non_vuln[
                    (df_non_vuln["complexity_bin"] == c_bin)
                    & (df_non_vuln["token_bin"] == t_bin)
                ]
                num_to_sample = min(int(count), len(non_vuln_in_bin))
                sampled_indices.extend(
                    non_vuln_in_bin.sample(
                        n=num_to_sample, random_state=self.random_state
                    ).index
                )

            df_non_vuln_sampled = df_non_vuln.loc[sampled_indices]

            # check if really balanced
            num_vuln = len(df_vuln)
            num_sampled = len(df_non_vuln_sampled)
            if num_sampled < (num_vuln * self.balance_tolerance):
                logger.warning(
                    f"\n⚠️ Distribution matching produced a poorly balanced set ({num_sampled}/{num_vuln})."
                )
                self._perform_fallback_sampling(df_vuln, df_non_vuln)
            else:
                # If the balance is acceptable, proceed as normal.
                logger.info(f"Distributions match >= {self.balance_tolerance*100}%")
                self._assemble_final_df(
                    df_vuln=df_vuln, df_non_vuln_sampled=df_non_vuln_sampled
                )

        except ValueError:
            logger.warning("\n⚠️  Distribution matching failed due to incompatible data for binning.")
            self._perform_fallback_sampling(df_vuln, df_non_vuln)

        finally:
            df_vuln = cast(pd.DataFrame, self.df_balanced[self.df_balanced["target"] == 1].copy())
            df_non_vuln = cast(pd.DataFrame, self.df_balanced[self.df_balanced["target"] == 0].copy())

            df_vuln_stats = df_vuln[
                ["cyclomatic_complexity", "token_count"]
            ].describe(percentiles=quantiles)
            df_non_vuln_stats = df_non_vuln[
                ["cyclomatic_complexity", "token_count"]
            ].describe(percentiles=quantiles)

            dfi.export(
                cast(pd.DataFrame, df_vuln_stats),
                assets_dir / "vulnerable_stats_after.png",
                table_conversion="matplotlib",
            )
            dfi.export(
                cast(pd.DataFrame, df_non_vuln_stats),
                assets_dir / "non_vulnerable_stats_after.png",
                table_conversion="matplotlib",
            )

        logger.info("Balancing complete.")
        if self.df_balanced is not None:
            print("Final balanced dataset distribution:")
            print(self.df_balanced["target"].value_counts())
        else:
            logger.warning("Balancing resulted in an empty dataset.")

    def process(self):
        """Runs the full pipeline: load, extract features, and balance."""
        self.read_jsonl_as_df()
        self._perform_stratified_sampling()
        self.save_df_to_jsonl(df=self.df_balanced)


if __name__ == "__main__":
    args = get_parser().parse_args()
    balancer = Balancer(input_fp=Path(args.input_fp), output_fp=Path(args.output_fp))
    balancer.process()

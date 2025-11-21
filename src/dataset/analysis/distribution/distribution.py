import sys
import re
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from collections import Counter
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator
from typing import cast

from . import utils
from .logging_config import setup_logger
from .cli import get_parser
from .ui import rich_exception, rich_print, rich_status
from .status import CWEStatusAnalyzer


logger = logging.getLogger(name=__name__)

fpath_italic = "/usr/share/fonts/TTF/CascadiaCodeNFItalic.ttf"
fpath = "/usr/share/fonts/TTF/CascadiaCodeNF.ttf"
prop_title = fm.FontProperties(fname=fpath, size=20)
prop = fm.FontProperties(fname=fpath, size=14)
prop_it = fm.FontProperties(fname=fpath, size=20)

plt.style.use("ggplot")


class DataDistribution:
    def __init__(self, pth2jsonl: str, output_dir: str, mitre_file: Path) -> None:
        self.output_dir: Path = Path(output_dir)
        self.mitre_file: Path = mitre_file
        self.dataset_path: Path = Path(pth2jsonl)
        self.__data_dict: list[utils.JsonlEntry] = utils.read_jsonl(input_file_path=self.dataset_path)
        self.data_df: pd.DataFrame = pd.DataFrame(data=self.__data_dict)

    def _validate_4plots(self) -> None:
        expected_columns: list[str] = sorted([
            "project", "cwe", "cwe_desc", "func",
            "target", "cyclomatic_complexity", "token_count" 
        ])
        columns = sorted(self.data_df.columns.tolist())

        try:
            if expected_columns != columns:
                raise ValueError("Oh boy, columns mismatch!!")
        except ValueError:
            rich_exception(show_locals=True)
            sys.exit(1)

    def _pie_chart_target(self) -> None:
        n_colors = len(set(self.data_df["target"]))
        palette = sns.color_palette(palette="pastel", n_colors=n_colors)

        target_df = self.data_df[["target"]]
        class_counts = target_df.groupby(by="target").size()
        counts = class_counts.values.tolist()
        label_names = class_counts.index.values.tolist()

        def __func(pct, allvals) -> str:
            absolute = int(pct / 100.0 * np.sum(allvals))
            return "{:.1f}%\n({:d})".format(pct, absolute)

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ret = ax.pie(
            x=counts,
            labels=label_names,
            colors=palette,
            startangle=90,
            autopct=lambda pct: __func(pct, counts),
            textprops=dict(color="k"),
            explode=(0.005, 0.005),
            radius=0.8,
        )

        ax.set_aspect("equal")

        if len(ret) == 3:
            wedges, texts, autotexts = ret[0], ret[1], ret[2]
            plt.setp(autotexts, size=12, style="italic")
        else:
            wedges, texts = ret[0], ret[1]

        plt.setp(texts, size=11, fontproperties=prop_title, weight="bold")

        fig.suptitle(
            "Vulnerability IDs density distribution",
            fontsize=15,
            fontweight="bold",
            ha="center",
            y=0.95,
            fontproperties=prop_title,
        )

        names = ["Non-Vulnerable", "Vulnerable"]
        plt.legend(wedges, names, loc="upper right", facecolor="white")
        plt.tight_layout()

        fig_path: Path = self.output_dir / "pie_target.svg"
        fig_path.parent.mkdir(exist_ok=True, parents=True)

        try:
            plt.savefig(fig_path, dpi=300)
            # plt.show()

            logger.info(f"Distribution pie chart saved to `{fig_path}`")
        except Exception:
            rich_exception(show_locals=True)

    def _barplot_vulnerability(self) -> None:
        # filter non-vulnerable targets
        vulnerable_df: pd.DataFrame = self.data_df.loc[self.data_df["target"] != 0]
        n_vul: int = len(vulnerable_df)
        vulnerable_df = vulnerable_df.drop(labels=["target"], axis=1)

        logger.info(f"Dropped {n_vul - len(vulnerable_df)} vulnerable entries")

        # subsitute empty CWE field with None
        vulnerable_df.loc[:, "cwe"] = vulnerable_df["cwe"].replace(
            to_replace="", value="None", regex=True
        )

        # some of the functions have multiple vulnerability codes
        cwe_extended: list[str] = []
        for el in vulnerable_df["cwe"]:
            if isinstance(el, list):
                for e in el:
                    cwe_extended.append(e.replace(" ", "").strip())
            else:
                cwe_extended.append(el)

        del vulnerable_df
        class_freq: dict[str, int] = dict(Counter(cwe_extended))
        del cwe_extended

        vulnerable_df: pd.DataFrame = (
            pd.DataFrame(data=class_freq, index=pd.Series([0]))
            .rename(index={0: "freq"})
            .T
        )

        vulnerable_df.reset_index(inplace=True)
        vulnerable_df.rename(columns={"index": "cwe"}, inplace=True)
        vulnerable_df["cwe"] = sorted(
            vulnerable_df["cwe"].to_list(),
            key=lambda id: int(
                re.sub(pattern=r"\[.*\]", repl="", string=id)
                .strip()
                .replace("CWE-", "")
            ),
        )

        pal = sns.color_palette(palette="pastel", n_colors=len(vulnerable_df["cwe"]))

        fig, ax = plt.subplots(figsize=(25, 15))

        ax = sns.barplot(
            x="cwe",
            y="freq",
            hue="cwe",
            data=vulnerable_df,
            palette=pal,
            zorder=10,
            legend=True,
        )

        fig.suptitle(
            "Statistical distribution for vulnerability IDs.",
            fontsize=16,
            fontweight="bold",
            ha="center",
            y=0.97,
        )
        fig.text(
            x=0.5,
            y=0.94,
            s="Analysis shows a right-skewed distribution with several high-value outliers",
            fontsize=15,
            ha="center",
            color="gray",
        )

        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("CWE IDs", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_formatter("{x:.0f}")
        ax.set_ylim(0, int(vulnerable_df["freq"].max() + 25))

        ax.grid(
            True, which="minor", linestyle=":", alpha=0.6, axis="y", color="darkgray"
        )
        ax.grid(
            True, which="major", linestyle="-", alpha=0.8, axis="y", color="darkgray"
        )
        ax.grid(
            True, which="major", linestyle="--", alpha=0.4, axis="x", color="darkgray"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()

        labels = [labels[i] + f" [{v}]" for i, v in enumerate(vulnerable_df["freq"])]

        plt.legend(
            handles,
            labels,
            title="CWE IDs",
            # loc="upper right",
            loc="best",
            ncol=5,
            bbox_to_anchor=[0.9, 0.9],
            bbox_transform=fig.transFigure,
            facecolor="white",
            fontsize=12,
            title_fontsize=14,
        )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        try:
            plt.savefig(self.output_dir / "cwe_distr.png", dpi=300)
            # plt.show()

            logger.info(f"Distribution pie chart saved to `{self.output_dir / "cwe_distr.png"}`")
        except Exception:
            rich_exception(show_locals=True)

    def generate_4plots(self):

        def _save_stats(obj, *, filename: str, mode: str="w"):
            with open(file=self.output_dir / filename, mode=mode) as f:
                json.dump(obj, f, indent=2)

        self._validate_4plots()
        self.data_df["target"] = self.data_df["target"].astype("int")
        token_stats = self.data_df["token_count"].describe(
            percentiles=[0.1, 0.5, 0.9, 0.99]
        )
        _save_stats(token_stats.to_dict(), filename="stats.json")

        df_vuln = cast(pd.DataFrame, self.data_df[self.data_df["target"] == 1].copy())
        vul_stats = df_vuln["token_count"].describe(percentiles=[0.1, 0.5, 0.9, 0.99])
        _save_stats(vul_stats.to_dict(), filename="stats.json", mode="a")
        df_non_vuln = cast(pd.DataFrame, self.data_df[self.data_df["target"] == 0].copy())
        non_vul_stats = df_non_vuln["token_count"].describe(percentiles=[0.1, 0.5, 0.9, 0.99])
        _save_stats(non_vul_stats.to_dict(), filename="stats.json", mode="a")

        self.generate_violin(
            df=df_vuln,
            x_col_content="Token Count",
            y_col_name="token_count",
            title="Distribution of token counts in vulnerable functions",
            filename="distr_token_count_vul.svg",
        )
        self.generate_violin(
            df=df_vuln,
            x_col_content="Complexity",
            y_col_name="cyclomatic_complexity",
            title="Distribution of cyclomatic complexity scores across vulnerable functions",
            filename="distr_complexity_vul.svg",
        )
        self.generate_violin(
            df=df_non_vuln,
            x_col_content="Token Count",
            y_col_name="token_count",
            title="Distribution of token counts across non-vulnerable functions",
            filename="dist_token_count_non_vul.svg",
        )
        self.generate_violin(
            df=df_non_vuln,
            x_col_content="Complexity",
            y_col_name="cyclomatic_complexity",
            title="Distribution of cyclomatic complexity scores across non-vulnerable functions",
            filename="distr_complexity_non_vul.svg",
        )

    def generate_violin(
        self,
        df: pd.DataFrame,
        x_col_content: str,
        y_col_name: str,
        title: str,
        filename: str,
    ):
        fig, ax = plt.subplots(figsize=(10, 15))
        category_df = pd.DataFrame(
            [x_col_content] * df[y_col_name].size, columns=["Category"] # type: ignore
        )
        extended_df = pd.concat([df.copy(), category_df], axis=1)

        s: pd.Series = pd.Series(extended_df[y_col_name])
        mean_val = s.mean()
        median_val = s.median()
        std_val = s.std()
        max_val = s.max()
        min_val = s.min()
        # idx_max = s.idxmax()
        # print(df.loc[idx_max])

        sns.violinplot(
            x="Category",
            y=y_col_name,
            data=extended_df,
            ax=ax,
            inner=None,
            linewidth=1,
            # cut=0,  # trim violin to limits
        )

        sns.boxplot(
            x="Category",
            y=y_col_name,
            data=extended_df,
            ax=ax,
            width=0.15,
            boxprops={
                "facecolor": "#ffffff",
                "zorder": 3,
                "edgecolor": "#444444",
                "linewidth": 1.2,
            },
            showfliers=False,
            showcaps=True,
            whiskerprops={"linewidth": 1.5, "zorder": 3, "color": "#444444"},
            medianprops={"linewidth": 2.5, "zorder": 4, "color": "#E74C3C"},
        )

        # Plot Mean and Median with distinct, clear markers
        ax.plot(
            [0],
            [median_val],  # type: ignore
            marker="o",
            color="#ff6b6b",
            markersize=10,
            linestyle="None",
            label="Median",
        )
        ax.plot(
            [0],
            [mean_val],  # type: ignore
            marker="D",
            color="#e63946",
            markersize=8,
            linestyle="None",
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Mean",
        )

        # SET LOG SCALE & CUSTOMIZE AXES
        # ==============================================================================
        ax.set_xlabel("")
        ax.set_ylabel("Token Count", fontsize=14, fontweight="bold", labelpad=15)
        # ax.set_xticklabels([''])
        ax.tick_params(axis="x", length=0)  # Remove x-axis tick marks
        ax.tick_params(axis="y", which="major", length=6, width=1.2)

        # REFINE ANNOTATIONS AND TITLE
        # ==============================================================================
        fig.suptitle(
            title,
            fontsize=16,
            fontweight="bold",
            ha="center",
            y=0.97,
        )
        fig.text(
            x=0.5,
            y=0.94,
            s="Violin plot representing the underlying distribution",
            fontsize=15,
            ha="center",
            color="gray",
        )

        stats_text = (
            f"Mean: {mean_val:.2f}\n"
            f"Median: {median_val:.2f}\n"
            f"Std Dev: {std_val:.2f}\n"
            f"Max values: {max_val:.0f}\n"
            f"Min values: {min_val:.0f}"
        )

        print(f"Plotting: {title}\n{stats_text}")

        # place text box
        ax.text(
            0.97,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.5", fc="#fefefe", ec="#cccccc", lw=1, alpha=0.9
            ),
        )

        # refine grid and spines
        ax.grid(
            True, which="minor", linestyle=":", alpha=0.6, axis="y", color="darkgray"
        )
        ax.grid(
            True, which="major", linestyle="-", alpha=0.8, axis="y", color="darkgray"
        )
        ax.grid(
            True, which="major", linestyle="--", alpha=0.4, axis="x", color="darkgray"
        )

        ax.set_ylabel("Counts", fontsize=12)
        if y_col_name == "complexity":
            major_interval = 20
            minor_interval = 10
        else:
            major_interval = 500
            minor_interval = 250

        ax.yaxis.set_major_locator(MultipleLocator(major_interval))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(MultipleLocator(minor_interval))
        ax.yaxis.set_minor_formatter("{x:.0f}")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Source line at the bottom
        fig.text(
            0.08,
            0.02,
            "Source: DiverseVul dataset",
            fontsize=10,
            color="#666666",
            ha="left",
        )

        plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.90))

        try:
            plt.savefig(self.output_dir / filename, dpi=300)
            # plt.show()

            logger.info(f"Distribution pie chart saved to `{self.output_dir / filename}`")
        except Exception:
            rich_exception(show_locals=True)

    def deprecation_analysis(self):
        logger.info("Starting deprecation analysis ...")
        analyzer = CWEStatusAnalyzer(
            dataset_path=self.dataset_path,
            cwe_status_csv=self.mitre_file,
            output_dir=self.output_dir,
        )

        stats = analyzer.run_analysis()

        logger.info(f"Deprecated CWEs: {stats['deprecated']['count']}")
        logger.info(f"Unknown CWEs: {stats['unknown']['count']}")
        logger.info(
            f"Functions with deprecated labels: {stats['deprecated']['func_count']}"
        )

    def generate_all(self):
        with rich_status(
            description="Generating Pie chart for binary label distribution"
        ):
            self._pie_chart_target()

        with rich_status("Generating CWE IDs distribution (barplot)"):
            self._barplot_vulnerability()

        with rich_status("Generating violin plots after stratified sampling"):
            self.generate_4plots()

        self.deprecation_analysis()


if __name__ == "__main__":
    setup_logger()
    args = get_parser().parse_args()

    logger.debug("🚀 Starting routine...🚀")
    data = DataDistribution(
        pth2jsonl=args.source, output_dir=args.target, mitre_file=args.mitre_file
    )
    data.generate_all()

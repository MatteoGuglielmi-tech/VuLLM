import re
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from collections import Counter
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator

import utils
from simple_loader import Loader
from logging_config import setup_logger

# setup logger
setup_logger()
logger = logging.getLogger(name=__name__)


fpath_italic = "/usr/share/fonts/TTF/CascadiaCodeNFItalic.ttf"
fpath = "/usr/share/fonts/TTF/CascadiaCodeNF.ttf"
prop_title = fm.FontProperties(fname=fpath, size=20)
prop = fm.FontProperties(fname=fpath, size=14)
prop_it = fm.FontProperties(fname=fpath, size=20)

# sns.axes_style(style="darkgrid")
# sns.set_style(style="ticks")
# sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})
plt.style.use("ggplot")


def get_parser():
    parser = argparse.ArgumentParser(prog="Balance dataset")
    parser.add_argument("--input_file_path", type=str, help="Absolute path to the raw dataset.")
    return parser


class DataDistribution:
    def __init__(self, pth2jsonl: str) -> None:
        self.__data_dict: dict = utils._read_jsonl(input_file_path=pth2jsonl)
        self.data_df: pd.DataFrame = pd.DataFrame(data=self.__data_dict)
        self.data_df = self.data_df.drop(labels=[ "func", "hash", "size", "commit_id", "message", "project" ], axis=1)
        self.__nb_samples: int = len(self.data_df["cwe"])

    def _pie_chart_target(self) -> None:
        palette = sns.color_palette(palette="pastel", n_colors=len(self.data_df["target"]))

        target_df: pd.DataFrame = self.data_df.drop(labels=["cwe"], axis=1)
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
            radius=.8
        )

        ax.set_aspect("equal")
        # ax.set_xlim(-1.1, 1.1)
        # ax.set_ylim(-1.1, 1.1)

        if len(ret) == 3:
            wedges, texts, autotexts = ret[0], ret[1], ret[2]
            plt.setp(autotexts, size=12, style="italic")
        else:
            wedges, texts = ret[0], ret[1]

        plt.setp(texts, size=11, fontproperties=prop_title, weight="bold")

        fig.suptitle(
            "Vulnerability IDs density distribution", fontsize=15,
            fontweight="bold", ha="center", y=0.95, fontproperties=prop_title,
        )

        names = ["Non-Vulnerable", "Vulnerable"]
        plt.legend(wedges, names, loc="upper right", facecolor="white", font=prop)
        plt.tight_layout()
        plt.savefig("./assets/pie_target.png", dpi=300) # Save with high resolution
        plt.show()

        logger.info("Distribution pie chart saved to `./assets/pie_target.png`")


    def _barplot_vulnerability(self) -> None:
        # filter non-vulnerable targets
        vulnerable_df: pd.DataFrame = self.data_df.loc[self.data_df["target"] != 0]
        vulnerable_df = vulnerable_df.drop(labels=["target"], axis=1)

        # subsitute empty CWE field
        vulnerable_df.loc[:, "cwe"] = vulnerable_df["cwe"].replace(
            to_replace="", value="None", regex=True
        )

        # some of the functions have multiple vulnerability codes
        cwe_extended: list[str] = []
        for el in vulnerable_df["cwe"]:
            if isinstance(el, list):
                for e in el: cwe_extended.append(re.sub(pattern=r"\s*", repl="", string=e))
            else: cwe_extended.append(el)

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
            )
        )

        pal = sns.color_palette(palette="pastel", n_colors=len(vulnerable_df["cwe"]))

        # create pie chart
        fig, ax = plt.subplots(figsize=(25,15))

        ax = sns.barplot(
            data=vulnerable_df, x="cwe", y="freq", hue="cwe",
            palette=pal, zorder=10, legend=True,
        )

        fig.suptitle("Statistical distribution for vulnerability IDs.", fontsize=16,
            fontweight="bold", ha="center", y=0.97,
        )
        fig.text(x=0.5, y=0.95, s='Analysis shows a right-skewed distribution with several high-value outliers',
                fontsize=15, ha="center", color='gray')

        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("CWE IDs", font=fpath, fontsize=12)
        ax.set_ylabel("Count", font=fpath, fontsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_formatter("{x:.0f}")
        ax.set_ylim(0, int(vulnerable_df["freq"].max() + 25))

        ax.grid(True, which="minor", linestyle=":", alpha=0.6, axis="y", color="darkgray")
        ax.grid(True, which="major", linestyle="-", alpha=0.8, axis="y", color="darkgray")
        ax.grid(True, which="major", linestyle="--", alpha=0.4, axis="x", color="darkgray")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()

        labels = [labels[i] + f" [{v}]" for i, v in enumerate(vulnerable_df["freq"])]

        plt.legend( handles, labels, 
            title="CWE IDs",
            # loc="upper right",
            loc="best",
            ncol=5,
            bbox_to_anchor=[0.935, 0.9],
            bbox_transform=fig.transFigure,
            facecolor="white",
            fontsize=12,
            title_fontsize=14,

        )

        plt.tight_layout(rect=(0., 0., 1., 0.96))
        plt.savefig("./assets/cwe_distr.png", dpi=300) # Save with high resolution

        plt.show()

    def __debug(self):
        with open("./misc/debug.txt", "a") as f:
            for i, item in enumerate(self.data_df["cwe"].tolist()):
                print(f"{i} : {item}", file=f)

    def generate_all(self):
        with Loader("Generating pie char for binary targets"): self._pie_chart_target()
        # with Loader("Generating CWE IDs distribution (barplot)"): self._barplot_vulnerability()


def create_dirs():
    dirs = [Path("./misc"), Path("./assets")]
    for p in dirs: p.mkdir(exist_ok=True)
    logger.info(msg=f"Created destination directorie")


if __name__ == "__main__":
    create_dirs()
    args = get_parser().parse_args()

    logger.debug("🚀 Starting routine...🚀")
    data = DataDistribution(pth2jsonl=args.input_file_path)
    data.generate_all()

    logger.info("✅ Process completed successfully! ✅")

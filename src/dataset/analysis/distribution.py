import os
import re
from collections import Counter
from contextlib import suppress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager as fm
from matplotlib.ticker import MultipleLocator

import utils
from animate import Loader
from log import logger

fpath_italic = "/usr/share/fonts/TTF/CascadiaCodeNFItalic.ttf"
fpath = "/usr/share/fonts/TTF/CascadiaCodeNF.ttf"
prop_title = fm.FontProperties(fname=fpath, size=20)
prop = fm.FontProperties(fname=fpath, size=14)
prop_it = fm.FontProperties(fname=fpath, size=20)

sns.axes_style(style="darkgrid")
sns.set_style(style="ticks")
sns.set_theme(style="ticks", rc={"axes.spines.right": False, "axes.spines.top": False})


class DataDistribution:
    def __init__(self, pth2json: str) -> None:
        self.__data_dict: dict = utils.read_json(pth=pth2json)
        self.data_df: pd.DataFrame = pd.DataFrame(data=self.__data_dict).T
        # don't care about func field
        self.data_df = self.data_df.drop(labels=["func"], axis=1)

        self.__nb_samples: int = len(self.data_df["cwe"])

    def pie_chart_target(self) -> None:
        palette = sns.color_palette(
            palette="pastel", n_colors=len(self.data_df["target"])
        )

        target_df: pd.DataFrame = self.data_df.drop(labels=["cwe"], axis=1)

        class_counts = target_df.groupby(by="target").size()
        counts = class_counts.values.tolist()
        label_names = class_counts.index.values.tolist()

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(5, 2))
        ret = ax.pie(  # not typed since linter goes crazy
            x=counts,
            labels=label_names,
            colors=palette,
            startangle=90,
            autopct=lambda pct: self.__func(pct, counts),
            textprops=dict(color="k"),
            explode=(0.005, 0.005),
        )

        if len(ret) == 3:
            wedges, texts, autotexts = ret[0], ret[1], ret[2]
            plt.setp(autotexts, size=15, style="italic")
        else:
            wedges, texts = ret[0], ret[1]

        plt.setp(texts, size=15, fontproperties=prop_title, weight="bold")

        plt.title(
            f"Vulnerability codes distribution\nTotal number of samples : {self.__nb_samples}",
            loc="center",
            fontproperties=prop_title,
        )

        names = ["Non-Vulnerable", "Vulnerable"]

        plt.legend(wedges, names, loc="upper right")

        plt.tight_layout()
        plt.show()
        fig.savefig(
            "./assets/pie_target.png", dpi=300, bbox_inches="tight", pad_inches=0.1
        )

    def __func(self, pct, allvals):
        absolute = int(pct / 100.0 * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    def pie_chart_vulnerability(self) -> None:
        # filter non-vulnerable targets
        vulnerable_df: pd.DataFrame = self.data_df.loc[self.data_df["target"] != "0"]
        # at this point, target is not necessary anymore
        vulnerable_df = vulnerable_df.drop(labels=["target"], axis=1)

        # subsitute empty CWE field
        vulnerable_df.loc[:, "cwe"] = vulnerable_df["cwe"].replace(
            to_replace="", value="None", regex=True
        )

        # some of the functions have multiple vulnerability codes
        cwe_extended: list[str] = []
        for el in vulnerable_df["cwe"]:
            if isinstance(el, list):
                for e in el:
                    cwe_extended.append(re.sub(pattern=r"\s*", repl="", string=e))
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

        # create pie chart
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(25, 10)

        ax = sns.barplot(
            data=vulnerable_df,
            x="cwe",
            y="freq",
            hue="cwe",
            palette=sns.color_palette(
                palette="pastel", n_colors=len(vulnerable_df["cwe"])
            ),
            zorder=10,
            legend=True,
        )
        plt.grid(True, zorder=0)

        fig.suptitle(
            "Statistical distribution for all thickness values in a single respiratory cycle."
        )

        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("CWE")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_minor_formatter("{x:.0f}")
        ax.set_ylim(0, int(vulnerable_df["freq"].max() + 25))
        ax.grid(True, zorder=100, which="minor", linestyle="--", alpha=0.5)
        ax.grid(True, zorder=100, which="major", linestyle="-", alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()

        labels = [labels[i] + f" [{v}]" for i, v in enumerate(vulnerable_df["freq"])]

        plt.legend(
            handles,
            labels,
            title="CWE of vulnerable functions",
            loc="upper right",
            ncol=5,
            bbox_to_anchor=[0.935, 0.9],
            bbox_transform=fig.transFigure,
        )

        plt.tight_layout()
        plt.show()
        ax.figure.figure.savefig(
            "./assets/cwe_distr.png", dpi=300, bbox_inches="tight", pad_inches=0.1
        )

    def __debug(self):
        with open("./misc/debug.txt", "a") as f:
            for i, item in enumerate(self.data_df["cwe"].tolist()):
                print(f"{i} : {item}", file=f)


def create_dirs():
    dirs = ["./misc", "./assets"]
    for d in dirs:
        with suppress(FileExistsError):
            os.mkdir(path=d)
            continue
        logger.info(msg=f"Created {d} directory")


if __name__ == "__main__":
    create_dirs()

    with Loader(desc="Plotting distribution "):
        data = DataDistribution(pth2json="../../../DiverseVul/DiverseVul.json")
        data.pie_chart_target()
        data.pie_chart_vulnerability()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge

fpath_italic = "/usr/share/fonts/TTF/CascadiaCodeNFItalic.ttf"
fpath = "/usr/share/fonts/TTF/CascadiaCodeNF.ttf"
prop_title = fm.FontProperties(fname=fpath, size=20)
prop = fm.FontProperties(fname=fpath, size=14)
prop_it = fm.FontProperties(fname=fpath, size=20)

plt.rcParams.update(
    {
        "lines.color": "white",
        "patch.edgecolor": "white",
        "text.color": "black",
        "axes.facecolor": "white",
        "axes.edgecolor": "lightgray",
        "figure.facecolor": "lightgray",
        "figure.edgecolor": "black",
        "savefig.facecolor": "lightgray",
        "savefig.edgecolor": "black",
    }
)


class DataDistribution:
    def __init__(self, pth2json: str) -> None:
        self.__data_dict: dict = utils.read_json(pth=pth2json)
        self.data_df: pd.DataFrame = pd.DataFrame(data=self.__data_dict).T

        self.__nb_samples: int = len(self.data_df["cwe"])

    def pie_chart_target(self) -> None:
        colors = sns.color_palette("pastel")[0:1]

        class_counts = self.data_df.groupby(by="target").size()
        counts = class_counts.values.tolist()
        label_names = class_counts.index.values.tolist()

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(5, 2))
        wedges, texts, autotexts = ax.pie(
            x=counts,
            labels=label_names,
            colors=colors,
            startangle=90,
            autopct=lambda pct: self.__func(pct, counts),
            textprops=dict(color="k"),
            explode=(0.005, 0.005),
        )

        plt.setp(autotexts, size=15, style="italic")
        plt.setp(texts, size=15, fontproperties=prop_title, weight="bold")

        plt.title(
            f"Vulnerability codes distribution\nTotal number of samples : {self.__nb_samples}",
            loc="center",
            fontproperties=prop_title,
        )

        plt.tight_layout()
        plt.show()

    def __func(self, pct, allvals):
        absolute = int(pct / 100.0 * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    def pie_chart_vulnerability(self) -> None:
        # | str only for linting issues
        label_names: list[str] | str = list(set(self.data_df["cwe"]))
        colors = sns.color_palette("pastel")[0 : len(label_names)]
        explode = tuple([0.005 for _ in range(len(label_names))])

        # filter non-vulnerable targets
        working_copy = self.data_df.copy()
        vulnerable_df: pd.DataFrame = working_copy.loc[working_copy["target"] != "0"]

        # subsitute empty CWE field with Unk identifier
        cwe_series = vulnerable_df["cwe"]
        cwe_series = cwe_series.replace(to_replace="", value="UNK", regex=True)
        vulnerable_df["cwe"] = cwe_series

        class_counts = vulnerable_df.groupby(by="cwe").size()
        # get label names and frequency
        label_names = class_counts.index.values.tolist()
        counts = class_counts.values.tolist()

        # create pie chart
        _, ax = plt.subplots(figsize=(5, 2))
        _, texts, autotexts = ax.pie(
            x=counts,
            labels=label_names,
            colors=colors,
            startangle=90,
            autopct=lambda pct: self.__func(pct, counts),
            textprops=dict(color="k"),
            explode=explode,
        )

        plt.setp(autotexts, size=15, style="italic")
        plt.setp(texts, size=15, fontproperties=prop_title, weight="bold")

        plt.title(
            f"Vulnerability codes distribution\nTotal number of samples : {self.__nb_samples}",
            loc="center",
            fontproperties=prop_title,
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    data = DataDistribution(pth2json="./divfix.json")
    # data.pie_chart_target()
    data.pie_chart_vulnerability()

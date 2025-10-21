import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff

from abc import abstractmethod
from typing import Iterable, cast
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import export_graphviz

from scipy.stats import randint
from loader_config import Loader


logger = logging.getLogger(__name__)


@dataclass
class RandomForestPipeline:
    """A base class for running a Random Forest classification pipeline."""

    source_path: Path
    assets_folder: Path
    test_size: float = 0.2
    n_estimators: int = 100
    n_iters: int = 10
    n_folds: int = 5
    seed: int = 42
    hp_tuning: bool = False

    model: RandomForestClassifier = field(init=False)

    encoder: LabelEncoder | None = None

    def _load_dataset(self) -> pd.DataFrame:
        dataset = load_dataset("json", data_files=str(self.source_path), split="train")
        return pd.DataFrame(data=dataset)

    def _confusion_matrix(self, y_test: Iterable, y_pred: Iterable, target_column: str):
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

        if target_column == "cwe":
            plt.figure(figsize=(30, 30))
        else:
            plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for Random Forest Classifier")

        self.assets_folder.mkdir(parents=True, exist_ok=True)
        pathname: Path = self.assets_folder / f"{target_column}_confusion_matrix.png"
        plt.savefig(pathname, dpi=300, bbox_inches="tight")
        logger.info(f"🗃️ Confusion matrix for {target_column} classification saved to: {pathname} 🗃️")

    def _heatmap(self, y_test: np.ndarray, y_pred: np.ndarray):
        class_names = sorted(list(set(y_test) | set(y_pred)))

        cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=class_names)

        fig = ff.create_annotated_heatmap(
            z=cm.tolist(),
            x=list(class_names),
            y=list(class_names),
            colorscale="Viridis",
            showscale=True,
        )

        fig.update_layout(
            title_text='<b>Interactive Confusion Matrix</b>',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            xaxis=dict(tickangle=-45)
        )

        # for i in range(len(fig.layout.annotations)):
        #     fig.layout.annotations[i].text = ''
        #
        fig.show()

        fig.write_html("confusion_matrix.html")

        try:
            fig.write_image("confusion_matrix.png", scale=2)
            print("Successfully saved confusion_matrix.png")
        except ValueError as e:
            print(f"Could not save static image. Please install kaleido: pip install kaleido\nError: {e}")


    def _classification_report(
        self, y_test: Iterable, y_pred: Iterable, target_column: str
    ):
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0.) # type: ignore
        df_report = pd.DataFrame(report_dict).transpose()

        df_report = df_report.round(3)
        df_report.loc["accuracy", ["precision", "recall", "f1-score"]] = np.nan
        df_report.loc[["macro avg", "weighted avg"], "support"] = df_report.loc[
            "macro avg", "support"
        ].astype(int)

        if target_column == "cwe":
            fig, ax = plt.subplots(figsize=(30, len(df_report) * 0.8))
        else:
            fig, ax = plt.subplots(figsize=(10, len(df_report) * 0.8))
        ax.axis("off")

        table = ax.table(
            cellText=df_report.values,  # type: ignore
            colLabels=df_report.columns,  # type: ignore
            rowLabels=df_report.index,  # type: ignore
            loc="center",
            cellLoc="center",
            colColours=["#f2f2f2"] * len(df_report.columns),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.2)

        plt.title(f"Classification Report for {target_column} classification", fontsize=16, pad=20)

        self.assets_folder.mkdir(parents=True, exist_ok=True)
        pathname: Path = (
            self.assets_folder / f"{target_column}_classification_report.png"
        )
        plt.savefig(pathname, bbox_inches="tight", dpi=300)
        plt.close(fig)

        logger.info(f"🗃️ Classification report for {target_column} classification saved to: {pathname} 🗃️")

    # def display_trees(self, x_train):
    #     for i in range(3):
    #         tree = self.model.estimators_[i]
    #         dot_data = export_graphviz(
    #             tree,
    #             feature_names=x_train.columns,
    #             filled=True,
    #             max_depth=2,
    #             impurity=False,
    #             proportion=True,
    #         )
    #         graph = graphviz.Source(dot_data)  # type: ignore
    #         display(graph)

    def hyperparamter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> dict[str, int]:
        param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}

        _, counts = np.unique(y_train, return_counts=True)
        min_count = counts.min()
        rf = RandomForestClassifier()

        rand_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=self.n_iters, cv=min(min_count, self.n_folds)
        )

        rand_search.fit(X_train, y_train)
        best_hp: dict[str, int] = rand_search.best_params_
        logger.info(f"🧨 Best hyperparameters: {best_hp} 🧨")

        return best_hp

    @abstractmethod
    def _preprocess_data(
        self, df: pd.DataFrame, feature_column: str, target_column: str
    ):
        """Placeholder for preprocessing. Must be implemented by child classes."""

        raise NotImplementedError("Subclasses must implement this method.")

    def run(self, feature_column: str, target_column: str):
        """Runs the full ML pipeline."""

        logger.info(f"🚀 --- Running pipeline for target: {target_column} --- 🚀 ")

        df = self._load_dataset()
        logger.info("🏗️ Dataset loaded. 🏗️")

        X, y = self._preprocess_data(
            df=df.copy(), feature_column=feature_column, target_column=target_column
        )
        logger.info("🚧 Preprocessing completed. 🚧")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )
        logger.info("🔪 Data splitting completed. 🔪")

        with Loader(
            desc_msg=f"Tokenizing {feature_column}",
            end_msg="⚙️ Tokenization (BoW) completed ⚙️",
            logger=logger,
        ):
            count_vectorizer = CountVectorizer()
            X_train_count = count_vectorizer.fit_transform(X_train)
            X_test_count = count_vectorizer.transform(X_test)

        if self.hp_tuning:
            with Loader(
                desc_msg=f"Started hyper-paramters tuning for {target_column} RFC",
                end_msg="🔧 HP tuning ended. 🔧",
                logger=logger,
            ):
                # cast to avoid annoying diagnostics
                best_hps = self.hyperparamter_tuning(
                    X_train=cast(np.ndarray, X_train_count),
                    y_train=cast(np.ndarray, y_train),
                )

            # apply best params
            self.model = RandomForestClassifier(
                n_estimators=best_hps.get("n_estimators", self.n_estimators),
                max_depth=best_hps.get("max_depth"),
                random_state=self.seed,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators, random_state=self.seed
            )

        with Loader(
            desc_msg=f"Fitting model for {target_column}",
            end_msg="💪 Model training completed. 💪",
            logger=logger,
        ):
            self.model.fit(X_train_count, y_train)

        y_pred = self.model.predict(X_test_count)

        # Decode if necessary
        if self.encoder:
            y_test = cast(np.ndarray, self.encoder.inverse_transform(y_test))
            y_pred = cast(np.ndarray, self.encoder.inverse_transform(y_pred))
            self._heatmap(y_test=y_test, y_pred=y_pred)
        else:
            self._confusion_matrix(y_test=y_test, y_pred=y_pred, target_column=target_column)

        logger.debug(f"Accuracy score for {target_column}: {round(accuracy_score(y_test, y_pred), 3)}")
        self._classification_report(y_test=y_test, y_pred=y_pred, target_column=target_column)


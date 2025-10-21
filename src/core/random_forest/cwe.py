import pandas as pd
from typing import cast
from base import RandomForestPipeline
from sklearn.preprocessing import LabelEncoder
from loader_config import Loader

class CWEModel(RandomForestPipeline):
    """Pipeline for the multi-class CWE target."""

    def _preprocess_data(self, df: pd.DataFrame, feature_column: str, target_column: str):
        df_exploded = df.explode(column=target_column)

        vc: pd.Series = df_exploded['cwe'].value_counts()
        # nested casting to avoid linter complaining
        cwe_to_keep = cast(pd.Series, cast(pd.DataFrame, vc[vc > 1]).index)
        df_filtered = df_exploded[df_exploded['cwe'].isin(cwe_to_keep)].copy()

        # 3. Label Encode the target
        self.encoder = LabelEncoder()
        X = df_filtered[feature_column]

        with Loader("Encoding CWE-IDs"):
            y = self.encoder.fit_transform(y=df_filtered[target_column])

        return X, y



from base import RandomForestPipeline
import pandas as pd

class BinaryModel(RandomForestPipeline):
    """Pipeline for the simple binary target."""

    def _preprocess_data(self, df: pd.DataFrame, feature_column: str, target_column: str):
        X = df[feature_column]
        y = df[target_column]

        return X, y

"""
Load the latest model and make predictions.

Used by the API endpoint. Keeps the model in memory after initial load.
"""

import json
import os

import joblib
import pandas as pd

from app.feature_engineering import engineer_features

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


class Predictor:
    """Loads the latest model and provides a prediction interface."""

    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or MODEL_DIR
        self.model = None
        self.feature_cols = None
        self.load_latest()

    def load_latest(self) -> None:
        """Load the model pointed to by latest.txt."""
        latest_path = os.path.join(self.model_dir, "latest.txt")
        with open(latest_path) as f:
            model_filename = f.read().strip()

        model_path = os.path.join(self.model_dir, model_filename)
        self.model = joblib.load(model_path)

        feature_path = os.path.join(self.model_dir, "feature_cols.json")
        with open(feature_path) as f:
            self.feature_cols = json.load(f)

    def predict(self, df: pd.DataFrame) -> list[float]:
        """Predict share_15_54 for a DataFrame with raw data.csv columns.

        Args:
            df: DataFrame with at minimum: timeslot_datetime_from,
                channel_id, and all ch{id}__f_{1..12} columns.

        Returns:
            List of predicted share values.
        """
        df_eng, _ = engineer_features(df)
        X = df_eng[self.feature_cols]
        return self.model.predict(X).tolist()

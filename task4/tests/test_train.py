"""Tests for the training pipeline."""

import json
import os
import tempfile

import numpy as np
import pandas as pd

from app.feature_engineering import CHANNEL_IDS
from tests.test_feature_engineering import _make_sample_df


class TestTrainingPipeline:
    def test_train_produces_model_file(self):
        """Verify training produces a .pkl file, metadata, and latest pointer."""
        from app.train import save_model, train_model

        df = _make_sample_df(n_timeslots=50)
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        date_col: list = []
        for d in dates:
            for _ in CHANNEL_IDS:
                date_col.append(d)
        df["timeslot_datetime_from"] = date_col[: len(df)]

        model, feature_cols, metrics = train_model(df)

        assert model is not None
        assert len(feature_cols) > 0
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["mae"] >= 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_model(model, feature_cols, metrics, model_dir=tmpdir)
            assert os.path.exists(path)
            assert os.path.exists(os.path.join(tmpdir, "latest.txt"))
            assert os.path.exists(os.path.join(tmpdir, "feature_cols.json"))

            # Verify metadata is valid JSON.
            meta_files = [
                f for f in os.listdir(tmpdir) if f.startswith("metadata_")
            ]
            assert len(meta_files) == 1
            with open(os.path.join(tmpdir, meta_files[0])) as f:
                meta = json.load(f)
            assert "mae" in meta["metrics"]

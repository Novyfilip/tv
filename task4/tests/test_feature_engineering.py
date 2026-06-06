"""Tests for the feature engineering pipeline."""

import numpy as np
import pandas as pd

from app.feature_engineering import (
    CATEGORICAL_FEATURES,
    CHANNEL_IDS,
    NUMERIC_FEATURES,
    engineer_features,
)


def _make_sample_row(channel_id: int = 3) -> dict:
    """Create a minimal valid row matching data.csv schema."""
    row = {
        "main_ident": 12345,
        "timeslot_datetime_from": "2023-06-15T20:00:00",
        "channel_id": channel_id,
        "share_15_54": 15.0,
        "share_15_54_3mo_mean": 14.0,
    }
    rng = np.random.RandomState(42)
    for ch in CHANNEL_IDS:
        for i in range(1, 13):
            feat = f"f_{i}"
            col = f"ch{ch}__{feat}"
            if feat == "f_10":
                row[col] = "Komedie" if ch != 9 else np.nan
            elif feat == "f_11":
                row[col] = "Cesko" if ch != 9 else np.nan
            else:
                row[col] = float(rng.rand() * 100)
    return row


def _make_sample_df(n_timeslots: int = 10) -> pd.DataFrame:
    """Create a sample DataFrame with one row per channel per timeslot."""
    rows = []
    for _ in range(n_timeslots):
        for ch in CHANNEL_IDS:
            row = _make_sample_row(channel_id=ch)
            rows.append(row)
    return pd.DataFrame(rows)


class TestFeatureEngineering:
    def test_returns_dataframe_and_feature_list(self):
        df = _make_sample_df()
        result_df, feature_cols = engineer_features(df)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0

    def test_temporal_features_created(self):
        df = _make_sample_df()
        result_df, _ = engineer_features(df)
        for col in ["hour", "day_of_week", "month", "is_weekend", "is_prime_time"]:
            assert col in result_df.columns, f"Missing: {col}"

    def test_own_features_created(self):
        df = _make_sample_df()
        result_df, _ = engineer_features(df)
        for feat in NUMERIC_FEATURES:
            assert f"own_{feat}" in result_df.columns, f"Missing: own_{feat}"

    def test_competitor_features_created(self):
        df = _make_sample_df()
        result_df, _ = engineer_features(df)
        for feat in NUMERIC_FEATURES:
            assert f"comp_mean_{feat}" in result_df.columns
            assert f"comp_max_{feat}" in result_df.columns

    def test_prime_time_flag_correct(self):
        df = _make_sample_df(n_timeslots=1)

        df["timeslot_datetime_from"] = "2023-06-15T20:00:00"
        result_df, _ = engineer_features(df)
        assert (result_df["is_prime_time"] == 1).all()

        df["timeslot_datetime_from"] = "2023-06-15T10:00:00"
        result_df, _ = engineer_features(df)
        assert (result_df["is_prime_time"] == 0).all()

    def test_no_target_leakage(self):
        df = _make_sample_df()
        _, feature_cols = engineer_features(df)
        assert "share_15_54" not in feature_cols
        assert "share_15_54_3mo_mean" not in feature_cols

    def test_handles_nan_in_categoricals(self):
        df = _make_sample_df()
        result_df, _ = engineer_features(df)
        assert "own_f_10_encoded" in result_df.columns


class TestEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame([_make_sample_row()])
        result_df, feature_cols = engineer_features(df)
        assert len(result_df) == 1
        assert len(feature_cols) > 0

    def test_all_channels_present_in_features(self):
        df = _make_sample_df()
        _, feature_cols = engineer_features(df)
        channel_cols = [c for c in feature_cols if c.startswith("channel_")]
        assert len(channel_cols) > 0

"""
Feature engineering for Nova TV share prediction.

Transforms raw data.csv rows into model-ready features.
Single source of truth — used by both train.py and predict.py.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CHANNEL_IDS = [3, 4, 9, 54]
NUMERIC_FEATURES = ["f_1", "f_2", "f_3", "f_4", "f_5",
                    "f_6", "f_7", "f_8", "f_9", "f_12"]
CATEGORICAL_FEATURES = ["f_10", "f_11"]


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply all feature transformations to a raw DataFrame.

    Args:
        df: DataFrame with columns matching data.csv schema.
            Must include: timeslot_datetime_from, channel_id,
            and all ch{id}__f_{1..12} columns.

    Returns:
        df: DataFrame with engineered features added (copy of input).
        feature_cols: Ordered list of feature column names for model input.
    """
    df = df.copy()
    df["timeslot_datetime_from"] = pd.to_datetime(df["timeslot_datetime_from"])

    # -- Temporal features ------------------------------------------------
    df["hour"]          = df["timeslot_datetime_from"].dt.hour
    df["day_of_week"]   = df["timeslot_datetime_from"].dt.dayofweek
    df["month"]         = df["timeslot_datetime_from"].dt.month
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype(int)
    df["is_prime_time"] = df["hour"].isin([19, 20, 21, 22]).astype(int)

    # -- Own vs competitor features (vectorised) --------------------------
    # Explicitly convert source columns to float64 so pandas doesn't infer
    # mixed dtypes (some csv cells are numeric, others string — newer pandas
    # versions raise TypeError on mixed assignments).
    for ch in CHANNEL_IDS:
        for feat in NUMERIC_FEATURES:
            col = f"ch{ch}__{feat}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for feat in NUMERIC_FEATURES:
        for ch in CHANNEL_IDS:
            mask = df["channel_id"] == ch
            df.loc[mask, f"own_{feat}"] = df.loc[mask, f"ch{ch}__{feat}"]
            other_chs = [c for c in CHANNEL_IDS if c != ch]
            other_cols = [f"ch{c}__{feat}" for c in other_chs]
            df.loc[mask, f"comp_mean_{feat}"] = df.loc[mask, other_cols].mean(axis=1)
            df.loc[mask, f"comp_max_{feat}"]  = df.loc[mask, other_cols].max(axis=1)

    for feat in CATEGORICAL_FEATURES:
        for ch in CHANNEL_IDS:
            mask = df["channel_id"] == ch
            df.loc[mask, f"own_{feat}"] = df.loc[mask, f"ch{ch}__{feat}"]

    # -- Encode categoricals ----------------------------------------------
    for feat in ["own_f_10", "own_f_11"]:
        df[feat] = df[feat].fillna("MISSING")
        le = LabelEncoder()
        df[f"{feat}_encoded"] = le.fit_transform(df[feat])

    # -- One-hot encode channel_id ----------------------------------------
    dummies = pd.get_dummies(df["channel_id"], prefix="channel", dtype=int)
    # Ensure all four channel columns exist, even if some channels are
    # absent from the input (common at inference time — single-channel predictions).
    for ch in CHANNEL_IDS:
        col = f"channel_{ch}"
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)

    # -- Final feature list -----------------------------------------------
    feature_cols = (
        ["hour", "day_of_week", "month", "is_weekend", "is_prime_time"]
        + [c for c in df.columns if c.startswith("channel_")]
        + [f"own_{feat}" for feat in NUMERIC_FEATURES]
        + [f"comp_mean_{feat}" for feat in NUMERIC_FEATURES]
        + [f"comp_max_{feat}" for feat in NUMERIC_FEATURES]
        + ["own_f_10_encoded", "own_f_11_encoded"]
    )

    return df, feature_cols

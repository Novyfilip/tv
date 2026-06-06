"""
Train the LightGBM model and save a versioned artifact.

Usage:
    python -m app.train --data-url <URL> [--model-dir models/]

If --data-url is not provided, falls back to local data/data.csv.
"""

import argparse
import json
import logging
import os
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.feature_engineering import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def download_data(url: str | None = None, local_path: str = "data/data.csv") -> pd.DataFrame:
    """Download dataset from URL, or fall back to local file."""
    if url:
        logger.info("Downloading data from %s", url)
        df = pd.read_csv(url, low_memory=False)
        os.makedirs("data", exist_ok=True)
        df.to_csv(local_path, index=False)
        logger.info("Saved local copy to %s", local_path)
    else:
        logger.info("Loading data from %s", local_path)
        df = pd.read_csv(local_path, low_memory=False)
    return df


def train_model(df: pd.DataFrame):
    """Train LightGBM on the full dataset.

    Returns (model, feature_cols, metrics dict).
    """
    df, feature_cols = engineer_features(df)

    # Time-based split: last 30 days as validation.
    df["timeslot_datetime_from"] = pd.to_datetime(df["timeslot_datetime_from"])
    cutoff = df["timeslot_datetime_from"].max() - pd.Timedelta(days=30)
    train_mask = df["timeslot_datetime_from"] < cutoff
    val_mask   = df["timeslot_datetime_from"] >= cutoff

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "share_15_54"]
    X_val   = df.loc[val_mask, feature_cols]
    y_val   = df.loc[val_mask, "share_15_54"]

    logger.info("Training on %d rows, validating on %d rows",
                len(X_train), len(X_val))

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    y_pred = model.predict(X_val)
    metrics = {
        "mae":            float(mean_absolute_error(y_val, y_pred)),
        "rmse":           float(np.sqrt(mean_squared_error(y_val, y_pred))),
        "r2":             float(r2_score(y_val, y_pred)),
        "n_train":        int(len(X_train)),
        "n_val":          int(len(X_val)),
        "best_iteration": int(model.best_iteration_),
    }

    logger.info("Validation: MAE=%.3f  RMSE=%.3f  R²=%.3f",
                metrics["mae"], metrics["rmse"], metrics["r2"])

    return model, feature_cols, metrics


def save_model(model, feature_cols, metrics, model_dir: str | None = None) -> str:
    """Save model with timestamp-based versioning.

    Creates:
        models/model_YYYYMMDD_HHMMSS.pkl
        models/feature_cols.json
        models/metadata_YYYYMMDD_HHMMSS.json
        models/latest.txt  (points to current version)
    """
    model_dir = model_dir or MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename    = f"model_{timestamp}.pkl"
    metadata_filename = f"metadata_{timestamp}.json"

    # Model.
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # Feature columns.
    feature_path = os.path.join(model_dir, "feature_cols.json")
    with open(feature_path, "w") as f:
        json.dump(feature_cols, f)

    # Metadata.
    metadata = {
        "timestamp":  timestamp,
        "model_file": model_filename,
        "metrics":    metrics,
    }
    metadata_path = os.path.join(model_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", metadata_path)

    # Latest pointer.
    latest_path = os.path.join(model_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(model_filename)
    logger.info("Latest pointer updated -> %s", model_filename)

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Nova TV share prediction model",
    )
    parser.add_argument(
        "--data-url", type=str, default=None,
        help="URL to download updated dataset from",
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR,
        help="Directory to save model artifacts",
    )
    args = parser.parse_args()

    df = download_data(url=args.data_url)
    model, feature_cols, metrics = train_model(df)
    save_model(model, feature_cols, metrics, model_dir=args.model_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()

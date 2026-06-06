"""
REST API for Nova TV share prediction.

Endpoints:
    POST /predict    — predict share_15_54 for given timeslots and channels
    GET  /health     — health check
    GET  /model-info — current model metadata
"""

import json
import os
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.predict import Predictor

app = FastAPI(
    title="Nova TV Share Prediction API",
    description="Predict audience share (15-54 demographic) for Nova TV channels",
    version="1.0.0",
)

predictor = Predictor()


class PredictionRequest(BaseModel):
    """Request body for the predict endpoint."""
    timeslot_datetime_from: List[str]   # ISO-format timestamps
    channel_id: List[int]               # Channel IDs (3, 4, 9, or 54)

    model_config = {
        "json_schema_extra": {
            "example": {
                "timeslot_datetime_from": [
                    "2024-01-10T20:00:00",
                    "2024-01-10T20:00:00",
                ],
                "channel_id": [3, 54],
            }
        }
    }


class PredictionItem(BaseModel):
    timeslot_datetime_from: str
    channel_id: int
    predicted_share_15_54: float | None
    note: str | None = None


class PredictionResponse(BaseModel):
    predictions: List[PredictionItem]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict share_15_54 for given timeslots and channel IDs.

    The stored dataset (data/data.csv) is used as a feature lookup.
    For each requested timeslot + channel, the closest matching row
    is found and its features are used for prediction.

    In production this would join against a live programming schedule.
    """
    if len(request.timeslot_datetime_from) != len(request.channel_id):
        raise HTTPException(
            status_code=400,
            detail="timeslot_datetime_from and channel_id must have equal length",
        )

    try:
        data_path = os.environ.get("DATA_PATH", "data/data.csv")
        full_df = pd.read_csv(data_path, low_memory=False)
        full_df["timeslot_datetime_from"] = pd.to_datetime(
            full_df["timeslot_datetime_from"]
        )

        results: List[PredictionItem] = []
        for ts, ch in zip(request.timeslot_datetime_from, request.channel_id):
            ts_dt = pd.to_datetime(ts)

            mask = full_df["channel_id"] == ch
            channel_df = full_df[mask].copy()
            channel_df["time_diff"] = (
                channel_df["timeslot_datetime_from"] - ts_dt
            ).abs()
            closest = channel_df.nsmallest(1, "time_diff")

            if closest.empty:
                results.append(PredictionItem(
                    timeslot_datetime_from=ts,
                    channel_id=ch,
                    predicted_share_15_54=None,
                    note="No matching data found",
                ))
            else:
                pred = predictor.predict(closest)
                results.append(PredictionItem(
                    timeslot_datetime_from=ts,
                    channel_id=ch,
                    predicted_share_15_54=round(pred[0], 3),
                ))

        return PredictionResponse(predictions=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model."""
    model_dir = predictor.model_dir
    latest_path = os.path.join(model_dir, "latest.txt")
    with open(latest_path) as f:
        model_filename = f.read().strip()

    timestamp = model_filename.replace("model_", "").replace(".pkl", "")
    metadata_path = os.path.join(model_dir, f"metadata_{timestamp}.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata

    return {"model_file": model_filename, "metadata": "not available"}

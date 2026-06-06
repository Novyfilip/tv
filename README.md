# Nova TV — Share Prediction Service

Predicts TV audience share (age 15-54 demographic) for Nova TV channels using
a LightGBM gradient-boosted decision tree model.

## Quick Start

### Prerequisites

- Docker and Docker Compose, OR
- Python 3.11+ with pip

### Option A: Docker (recommended)

```bash
# Build and start the service
docker-compose up --build

# The API is now available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Option B: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Copy data and the pre-trained model from task2
cp ../data.csv data/
cp ../models/lgb_model_task2.pkl models/model_20240101_000000.pkl
echo "model_20240101_000000.pkl" > models/latest.txt
cp ../models/feature_cols.json models/

# Start the API server
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### `POST /predict`

Predict `share_15_54` for given timeslots and channels.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timeslot_datetime_from": ["2024-01-10T20:00:00"],
    "channel_id": [3]
  }'
```

Response:

```json
{
  "predictions": [
    {
      "timeslot_datetime_from": "2024-01-10T20:00:00",
      "channel_id": 3,
      "predicted_share_15_54": 21.543,
      "note": null
    }
  ]
}
```

The endpoint looks up programming features from the stored dataset.
For timeslots not found in the data, `predicted_share_15_54` is null
with a note explaining why.

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### `GET /model-info`

Returns the current model's version timestamp and validation metrics.

## Daily Retraining

```bash
# Retrain from a new data URL
./scripts/retrain.sh https://example.com/updated_data.csv

# Retrain from local data
./scripts/retrain.sh

# Rollback to a previous model
./scripts/rollback.sh model_20240115_120000.pkl
```

The retrain script:

1. Downloads updated data (if URL provided)
2. Trains a new LightGBM model version
3. Promotes the new model on success
4. Rolls back to the previous version on failure

## Model Versioning

Models are saved with timestamps: `models/model_YYYYMMDD_HHMMSS.pkl`

```
models/
├── model_20240101_000000.pkl       # Initial model (from task2)
├── model_20240115_120000.pkl       # Retrained model
├── metadata_20240115_120000.json   # Metrics for that version
├── feature_cols.json               # Feature column list (same for all versions)
└── latest.txt                      # Pointer: "model_20240115_120000.pkl"
```

`latest.txt` always points to the currently active model version.
Metadata JSON files contain validation metrics, row counts, and best iteration.

## Running Tests

```bash
# Requires a trained model in models/
# Install test dependencies first:
pip install pytest httpx

pytest tests/ -v
```

## Project Structure

```
task4/
├── app/                         # Application code
│   ├── api.py                   # FastAPI endpoints
│   ├── train.py                 # Training pipeline
│   ├── predict.py               # Prediction logic
│   └── feature_engineering.py   # Shared feature transforms
├── scripts/                     # Orchestration
│   ├── retrain.sh               # Daily retraining
│   └── rollback.sh              # Model rollback
├── tests/                       # Test suite
├── models/                      # Versioned model artifacts
├── data/                        # Dataset storage
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Design Decisions

- **LightGBM** over neural networks: trains in seconds on CPU, handles missing
  values natively (critical for Channel 9's absent metadata), explainable via SHAP.
  No GPU required for training or inference.
- **Timestamp-based versioning** over git-based: simpler, each model is a
  self-contained .pkl file with accompanying metadata JSON.
- **FastAPI** over Flask: automatic OpenAPI docs, async support, built-in
  request validation via Pydantic.
- **Shell script orchestration** over Airflow/Prefect: appropriate for a
  single-model daily job. Would migrate to a workflow engine if the pipeline
  grew to multiple models or complex dependencies.
- **Nearest-timeslot feature lookup** for predictions: the API finds the closest
  matching timeslot in the stored dataset to get programming features. In production
  this would join against a live programming schedule database.

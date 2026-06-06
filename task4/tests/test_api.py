"""Tests for the prediction API.

Requires a trained model in models/.
Copy from task2 first:
    cp ../models/lgb_model_task2.pkl models/model_20240101_000000.pkl
    echo "model_20240101_000000.pkl" > models/latest.txt
    cp ../models/feature_cols.json models/
"""

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestModelInfoEndpoint:
    def test_model_info_returns_data(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_file" in data or "timestamp" in data


class TestPredictEndpoint:
    def test_predict_returns_predictions(self):
        response = client.post("/predict", json={
            "timeslot_datetime_from": ["2024-01-10T20:00:00"],
            "channel_id": [3],
        })
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 1

    def test_predict_mismatched_lengths_returns_400(self):
        response = client.post("/predict", json={
            "timeslot_datetime_from": ["2024-01-10T20:00:00", "2024-01-10T21:00:00"],
            "channel_id": [3],
        })
        assert response.status_code == 400

    def test_predict_multiple_channels(self):
        response = client.post("/predict", json={
            "timeslot_datetime_from": [
                "2024-01-10T20:00:00",
                "2024-01-10T20:00:00",
            ],
            "channel_id": [3, 54],
        })
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 2

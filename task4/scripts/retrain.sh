#!/bin/bash
# retrain.sh — Daily model retraining orchestration
#
# Usage:
#   ./scripts/retrain.sh [DATA_URL]
#
# If DATA_URL is provided, downloads fresh data from that URL.
# Otherwise, retrains on local data/data.csv.
#
# Flow:
#   1. Download updated data (if URL provided)
#   2. Train a new model version
#   3. Validate: if training fails, roll back to previous model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${PROJECT_DIR}/models"
DATA_URL="${1:-}"

echo "=== Nova TV Model Retraining ==="
echo "Timestamp: $(date -Iseconds)"
echo "Model dir: ${MODEL_DIR}"

# Remember current model for potential rollback.
PREVIOUS_MODEL=""
if [ -f "${MODEL_DIR}/latest.txt" ]; then
    PREVIOUS_MODEL=$(cat "${MODEL_DIR}/latest.txt")
    echo "Current model: ${PREVIOUS_MODEL}"
fi

# Run training.
echo ""
echo "--- Training new model ---"
if [ -n "${DATA_URL}" ]; then
    python -m app.train --data-url "${DATA_URL}" --model-dir "${MODEL_DIR}"
else
    python -m app.train --model-dir "${MODEL_DIR}"
fi

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
    if [ -n "${PREVIOUS_MODEL}" ]; then
        echo "Rolling back to ${PREVIOUS_MODEL}"
        echo "${PREVIOUS_MODEL}" > "${MODEL_DIR}/latest.txt"
    fi
    exit 1
fi

echo ""
echo "--- Retraining complete ---"
NEW_MODEL=$(cat "${MODEL_DIR}/latest.txt")
echo "New model:      ${NEW_MODEL}"
echo "Previous model: ${PREVIOUS_MODEL:-none}"
echo ""
echo "To rollback manually: ./scripts/rollback.sh ${PREVIOUS_MODEL:-<model_filename>}"

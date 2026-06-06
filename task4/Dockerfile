FROM python:3.11-slim

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ... the rest of your Dockerfile (COPY requirements.txt, RUN pip install, etc.)

# Install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
RUN chmod +x scripts/*.sh

# Create runtime directories.
RUN mkdir -p models data

# Copy pre-trained model as initial version.
COPY models/ models/

# Expose API port.
EXPOSE 8000

# Default: run the API server.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

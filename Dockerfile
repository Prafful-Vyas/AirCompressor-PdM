FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies first so this layer is cached unless deps change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY api ./api
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

# Model artifacts are loaded from MLflow at startup (see src/predict.py).
# Mount a tracking store (e.g. -v ./mlruns:/app/mlruns, or set
# MLFLOW_TRACKING_URI to a remote server) when running this image.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

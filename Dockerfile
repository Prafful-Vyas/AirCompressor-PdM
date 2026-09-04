# ---- builder: resolves deps with uv, kept out of the final image ----
FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install dependencies first so this layer is cached unless deps change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# ---- runtime: just the venv + app code, running as a non-root user ----
FROM python:3.13-slim AS runtime

RUN useradd --system --create-home --home-dir /app --uid 1000 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app api ./api
COPY --chown=app:app src ./src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

USER app

EXPOSE 8000

# Model artifacts are loaded from MLflow at startup (see src/predict.py).
# Mount a tracking store (e.g. -v ./mlruns:/app/mlruns, or set
# ACPDM_MLFLOW_TRACKING_URI to a remote server) when running this image.
# /health is a pure liveness check -- it stays 200 even before any model
# has been trained/promoted, so the check only confirms the process is up.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

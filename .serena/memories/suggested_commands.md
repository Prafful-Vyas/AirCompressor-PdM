# Suggested Commands (Windows / PowerShell dev machine)

All Python commands go through `uv run` — do not invoke `python`/`pytest`/`ruff` directly
unless already inside an activated `.venv`.

- Install deps: `uv sync`
- Train + track: `uv run python -m src.train` then `mlflow ui` (http://localhost:5000)
- Run API (dev): `uv run uvicorn api.main:app --reload` → docs at http://localhost:8000/docs
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Format check only (what CI runs): `uv run ruff format --check .`
- Install git hooks: `pre-commit install` (runs ruff check --fix + ruff format on commit,
  config in `.pre-commit-config.yaml`)
- Docker: `docker build -t air-compressor-api .` then
  `docker run -p 8000:8000 -v "$(pwd)/mlruns:/app/mlruns" air-compressor-api`
  (Windows PowerShell: use `${PWD}` instead of `$(pwd)` for the volume mount)

No Windows-specific variants needed for `git`/`ls`/`grep` — this repo is developed with a
POSIX-like shell (Git Bash) available even on Windows.
# Suggested Commands (Windows / PowerShell dev machine)

**Known local bug:** on this machine, `uv run <console-script>` (e.g. `uv run
pytest`, `uv run ruff ...`, `uv run mypy ...`) fails with `error: uv trampoline
failed to canonicalize script path` — caused by the space in the `Pulkit Vyas`
directory name breaking uv's trampoline `.exe` shims on Windows. Workaround:
always invoke via `uv run python -m <module>` instead (e.g. `uv run python -m
pytest`), which bypasses the trampoline entirely. `uv run python ...` itself
(no module) works fine. This is local-machine-specific — CI runs on ubuntu with
no spaces in the path, so plain `uv run pytest` etc. work fine there; don't
"fix" the CI workflow to work around this.

- Install deps: `uv sync`
- Train + track: `uv run python -m src.train` then `mlflow ui` (http://localhost:5000)
- Promote a trained model to serving: `uv run python -m src.promote list <target>`,
  `... promote <target> <version>`, `... current <target>` (see `mem:mlops_conventions`)
- Run API (dev): `uv run python -m uvicorn api.main:app --reload` → docs at
  http://localhost:8000/docs (needs `X-API-Key` header on `/predict`, see `.env.example`)
- Tests + coverage: `uv run python -m pytest` (gate is `--cov-fail-under=70`, set via
  `pyproject.toml` addopts, not a CLI flag)
- Lint: `uv run python -m ruff check .`
- Format: `uv run python -m ruff format .`
- Format check only (what CI runs): `uv run python -m ruff format --check .`
- Type check: `uv run python -m mypy src api`
- Dependency vuln scan: `uv export --no-hashes --format requirements-txt -o
  requirements-audit.txt` then `uvx pip-audit -r requirements-audit.txt --strict`
  (see `mem:mlops_conventions` for the documented `--ignore-vuln` list)
- Install git hooks: `uvx pre-commit install` (ruff + gitleaks; config in
  `.pre-commit-config.yaml`). Validate without installing: `uvx pre-commit run --all-files`
- Docker: `docker build -t air-compressor-api .` then `docker run -p 8000:8000 -e
  ACPDM_API_KEY=... -v "$(pwd)/mlruns:/app/mlruns" air-compressor-api`
- Docker Compose (API + real MLflow server): `cp .env.example .env` (set
  `ACPDM_API_KEY`), then `docker compose up --build -d`

No Windows-specific variants needed for `git`/`ls`/`grep` — this repo is developed with a
POSIX-like shell (Git Bash) available even on Windows.
# Task Completion Checklist

Before considering a coding task done, run (matches CI in `.github/workflows/ci.yml`;
use `uv run python -m <module>` form, see `mem:suggested_commands` for why):

1. `uv run python -m ruff check .`
2. `uv run python -m ruff format --check .` (or without `--check` to actually fix)
3. `uv run python -m mypy src api`
4. `uv run python -m pytest` (enforces `--cov-fail-under=70` via pyproject.toml)

CI additionally runs `uvx pip-audit` (dependency vuln scan) and a `docker build` smoke
test — only worth running locally if you touched dependencies or the Dockerfile.

If touching `src/train.py`, `src/predict.py`, or the fault-column relationships, also
sanity-check against `mem:domain_model` (e.g. don't silently change which sensors are
in `SENSOR_COLS`/`NON_NEGATIVE_COLS` without checking `docs/ARCHITECTURE.md`'s
correlation rationale). If touching model loading/serving, remember registering a
model via `train.py` does NOT make it servable — see `mem:mlops_conventions`.
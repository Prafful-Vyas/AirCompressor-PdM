# Task Completion Checklist

Before considering a coding task done, run (matches CI in `.github/workflows/ci.yml`):

1. `uv run ruff check .`
2. `uv run ruff format --check .` (or `uv run ruff format .` to actually fix)
3. `uv run pytest`

All three must pass — CI runs exactly these on every push/PR to `main`. If touching
`src/train.py` or the fault-column relationships, also sanity-check against
`mem:domain_model` (e.g. don't silently change which sensors are in `SENSOR_COLS`
without checking `docs/ARCHITECTURE.md`'s correlation rationale).
# Code Conventions

- Ruff config (`pyproject.toml`): line-length 88, target py313, rules
  `E, F, I, B, UP` (pycodestyle, pyflakes, isort, bugbear, pyupgrade).
  `notebooks/` and `docs/` are excluded from ruff. Ruff's B008 flags `Depends(fn)` as
  a default-arg function call in some cases (inconsistently — flags `Depends(get_settings)`
  but not `Depends(some_instance)`); fix by calling the function inside the dependency
  body instead of via nested `Depends`, not by suppressing the rule.
- One class per pipeline stage, named `<Stage>er`/`<Stage>or`
  (`DataIngestor`, `FeatureEngineer`, `Trainer`, `Predictor`), each with a module-level
  `logger = logging.getLogger(__name__)` and structured log calls rather than prints.
  `src/promote.py` breaks this pattern deliberately — it's a CLI, not a pipeline stage,
  so it's a set of module-level functions (`list_versions`, `promote`, `show_current`)
  dispatched from `main()`, not a class.
- Column/experiment/domain constants live only in `src/config.py` (`mem:core`); runtime
  /infra config (API keys, CORS, tracking URI) lives only in `src/settings.py`. Never
  redefine either locally in train.py, predict.py, ingestion.py, or api/main.py.
- Preprocessing + model are always bundled into a single sklearn `Pipeline` object
  (built by `build_preprocessing_pipeline()` / `Trainer._build_pipeline()`) so the fitted
  artifact logged to MLflow is self-contained and callable directly on raw feature input.
- Docstrings are used sparingly and only to explain *why* (e.g. `config.py`'s module
  docstring explains why constants are centralized), not to restate the signature.
- **FastAPI/Starlette gotcha**: the default `RequestValidationError` handler echoes the
  raw invalid input value back in the response body via `jsonable_encoder`. If that
  value is a NaN/Infinity float, Starlette's `JSONResponse` (which sets
  `allow_nan=False`) raises and turns a clean 422 into an unhandled 500. `api/main.py`
  registers a custom handler that omits the raw value (`loc`/`msg`/`type` only) to
  avoid this — don't revert to the default handler without re-solving this.
- httpx's `TestClient.post(json=...)` refuses to serialize NaN/Infinity client-side
  (unlike stdlib `json.dumps`, which allows them by default) — tests that need to send
  a NaN payload must build the body with `json.dumps` themselves and post via
  `content=`/`headers={"Content-Type": "application/json"}` instead of `json=`.
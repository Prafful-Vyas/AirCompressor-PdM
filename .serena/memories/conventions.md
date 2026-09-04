# Code Conventions

- Ruff config (`pyproject.toml`): line-length 88, target py313, rules
  `E, F, I, B, UP` (pycodestyle, pyflakes, isort, bugbear, pyupgrade).
  `notebooks/` and `docs/` are excluded from ruff.
- One class per pipeline stage, named `<Stage>er`/`<Stage>or`
  (`DataIngestor`, `FeatureEngineer`, `Trainer`, `Predictor`), each with a module-level
  `logger = logging.getLogger(__name__)` and structured log calls rather than prints.
- Column/experiment constants live only in `src/config.py` (`mem:core`) — never redefine
  `SENSOR_COLS`/`TARGET_COLS`/etc. locally in train.py, predict.py, or api/main.py.
- Preprocessing + model are always bundled into a single sklearn `Pipeline` object
  (built by `build_preprocessing_pipeline()` / `Trainer._build_pipeline()`) so the fitted
  artifact logged to MLflow is self-contained and callable directly on raw feature input.
- Docstrings are used sparingly and only to explain *why* (e.g. `config.py`'s module
  docstring explains why constants are centralized), not to restate the signature.
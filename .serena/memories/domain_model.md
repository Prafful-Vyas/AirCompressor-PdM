# Domain Model (see docs/ARCHITECTURE.md for full derivation)

- Dataset: `data/raw/aircompressor.csv`, 1000 rows, 26 columns. 4 fault targets
  (`bearings`, `wpump`, `radiator`, `exvalve`) are **mutually exclusive** by construction
  (each row sums to 0 or 1 across the 4 targets; pairwise correlation between any two
  targets is exactly -0.25, derivable from p=0.2 marginal fault rate + mutual exclusivity).
  Implication: 4 independent binary classifiers (current approach) each implicitly also
  learn "not one of the other 3 faults" as part of what class 0 means.
- `acmotor` is constant (`"Stable"`) across all 1000 rows — zero variance, not usable as
  a feature or signal.
- "Heat cluster": `outlet_temp`, `oil_tank_temp`, `water_inlet_temp`, `water_outlet_temp`
  pairwise correlate 0.96–0.98 (same underlying compression-heat event). A *divergence*
  in this cluster (not absolute temperature) is the real anomaly signal, e.g. radiator
  faults = water/oil temps rising while `water_flow` drops.
- Ground/head accelerometer axis pairs are near-duplicate signals (`gaccz`↔`haccz`
  r=0.998, `gaccx`↔`haccx` r=0.99) — this is why `SENSOR_COLS` (`src/config.py`) only
  includes `gaccx`/`haccx`, not the full 6-axis set, and why bearing wear is better
  isolated via rolling std / FFT than raw vibration level (raw level is confounded with
  operating load: `gaccz`/`haccz` correlate ~0.95-0.97 with `torque`/`outlet_pressure_bar`).
- `SENSOR_COLS` in `src/config.py` deliberately samples **one representative column per
  physical subsystem** (drive train, compression, water loop, vibration) rather than all
  20 numeric columns, to avoid feeding near-duplicate signals into the model — don't
  "complete" this list without re-reading `docs/ARCHITECTURE.md` §5 first.
- Reported metrics (README table) are from a single run and drift as training data
  grows — treat as illustrative, not a regression baseline to assert against in tests.
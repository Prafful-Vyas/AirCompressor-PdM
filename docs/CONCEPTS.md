# Concepts Used in This Project

📚 **Docs:** [README](../README.md) · [Architecture](ARCHITECTURE.md) · [Concepts](CONCEPTS.md)

This document explains every concept, technique, and tool used across the codebase —
the production pipeline in `src/`/`api/`, and the notebooks in `notebooks/`. The pipeline
(`src/ingestion.py`, `preprocessing.py`, `features.py`, `train.py`, `predict.py`) is built
from the approach worked out in **`PdM-Approach.ipynb`** (section 8); `NN-Approach.ipynb`
(mirrored in `docs/doc.md`, section 9) is a separate, supplementary experiment. It's meant
as a glossary + "why this was used here" reference.

---

## 1. Domain Concepts

### Predictive Maintenance (PdM)
Instead of fixing equipment on a fixed schedule (preventive maintenance) or after it
breaks (reactive maintenance), predictive maintenance uses sensor data to predict
**when/whether a component will fail** so it can be repaired just in time. This project
predicts failure of four air compressor components: `bearings`, `wpump` (water pump),
`radiator`, and `exvalve` (exhaust valve).

### Sensor Features (the raw dataset)
The dataset (`data/raw/aircompressor.csv`) contains readings from a real air compressor:
- **RPM, motor_power, torque** — how hard the motor is working.
- **outlet_pressure_bar, air_flow, noise_db, outlet_temp** — output-side compressed air characteristics.
- **wpump_outlet_press, water_inlet/outlet_temp, wpump_power, water_flow** — cooling water loop.
- **oilpump_power, oil_tank_temp** — lubrication system.
- **gaccx/y/z (ground acceleration), haccx/y/z (head acceleration)** — vibration sensors in 3 axes; vibration is one of the earliest indicators of bearing wear.
- **bearings/wpump/radiator/exvalve/acmotor** — target/status columns (0 = OK, 1 = failure, except `acmotor` which is categorical like "Stable").

Physically, degrading bearings vibrate more and less predictably before they fail —
this is why vibration + rolling statistics are central to the feature engineering.

---

## 2. Software Architecture Concepts

### Modular Pipeline Design
Rather than one long script (or just notebooks), logic is split into single-purpose
modules: `ingestion.py` → `preprocessing.py` → `features.py` → `train.py` → `predict.py`.
Each stage takes the previous stage's output as input. This makes each part testable,
reusable, and independently swappable (e.g., you could change the model in `train.py`
without touching how data is loaded).

### Object-Oriented Design (Classes)
`DataIngestor`, `FeatureEngineer`, `Trainer` are classes that bundle related state
(e.g., `window_sizes`, `raw_data_path`) with the methods that operate on it, instead of
passing many loose arguments between free functions.

### Logging (`logging` module)
Every module configures a `logger` via `logging.basicConfig(...)` and uses
`logger.info/warning/error` instead of `print()`. This gives timestamped, leveled,
filterable output — essential once code runs unattended (e.g., in a Docker container or
scheduled job) where you can't just watch a terminal.

### `if __name__ == "__main__":` guard
Each module has a demo/test block under this guard. It only runs when the file is
executed directly (`python src/features.py`), not when it's imported by another module
(e.g., `train.py` importing `FeatureEngineer`). This lets each file double as both a
library component and a runnable smoke test.

---

## 3. Data Ingestion (`src/ingestion.py`)

### Defensive file loading
`DataIngestor.load_data()` checks `os.path.exists()` before reading and wraps the CSV
read in a `try/except`, raising a clear `FileNotFoundError` instead of a cryptic pandas
error. This is a basic but important production habit: fail loudly and specifically.

### Data summary / sanity check
`get_data_summary()` logs column names and `isnull().sum()` (missing values per column)
right after loading — a cheap first check to catch obviously broken data early.

---

## 4. Data Preprocessing (`src/preprocessing.py`)

### Scikit-learn `Pipeline`
`build_preprocessing_pipeline()` chains transformation steps (`sanity_check` → `imputer`
→ `scaler`) into a single `Pipeline` object. A pipeline guarantees the *exact same*
sequence of transformations is applied to training and test/inference data, and lets you
call `.fit_transform()` / `.transform()` as one unit instead of manually replaying steps.

### Custom Transformers (`BaseEstimator`, `TransformerMixin`)
`SensorSanityChecker` subclasses these two sklearn base classes to become a pipeline-
compatible step with its own domain logic (e.g., clamping physically impossible negative
pressure to 0). `TransformerMixin` gives it a free `.fit_transform()` by combining `fit`
and `transform`; `BaseEstimator` gives it sklearn-standard `get_params`/`set_params`.

### Imputation (`SimpleImputer(strategy="median")`)
Fills missing sensor values. **Median** (not mean) is used because sensor data is often
skewed by spikes/outliers, and the median is robust to those — a mean would be dragged
off-center by a handful of extreme readings. For a sorted sample $x_{(1)} \le \dots \le x_{(n)}$:

$$
\text{median}(x) =
\begin{cases}
x_{\left(\frac{n+1}{2}\right)} & n \text{ odd} \\[4pt]
\dfrac{1}{2}\left(x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}\right) & n \text{ even}
\end{cases}
$$

Unlike the mean $\bar{x} = \frac{1}{n}\sum_i x_i$, every term of which shifts when one
extreme value changes, the median only depends on the middle observation(s) — a single
huge sensor spike can't drag it away from the bulk of the data.

### Feature Scaling — `RobustScaler` vs `StandardScaler`
Both rescale numeric features so no single sensor (e.g., `motor_power` in the hundreds)
dominates a model just because of its raw magnitude compared to one like `torque`.
**StandardScaler** centers on the *mean* and scales by *standard deviation* (the **z-score**):

```math
z = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{n}\sum_i x_i, \qquad \sigma = \sqrt{\frac{1}{n}\sum_i (x_i-\mu)^2}
```

This assumes roughly normal data, and is sensitive to outliers because both $\mu$ and
$\sigma$ are themselves skewed by outliers.

**RobustScaler** centers on the *median* and scales by the **interquartile range (IQR)**:

```math
x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}, \qquad \text{IQR}(x) = Q_3 - Q_1
```

where $Q_1$/$Q_3$ are the 25th/75th percentiles. Since $Q_1$, $Q_3$, and the median all
come from the middle 50% of the data, a handful of extreme sensor spikes barely move
them — far less affected than $\mu$/$\sigma$. This is why the pipeline uses
`RobustScaler` for failing-machine sensor data (`preprocessing.py`), while the
exploratory notebooks use the simpler `StandardScaler` for PCA input (PCA's math
specifically requires mean-centered, unit-variance input — see §9).

---

## 5. Feature Engineering (`src/features.py`)

### Rolling Window Statistics
`create_rolling_features()` computes, for each sensor and each window size $w$ (5, 10, 20
timesteps), a **rolling mean** and **rolling standard deviation**
(`df[col].rolling(window=window).mean()/.std()`) over the trailing window ending at row $t$:

$$
\bar{x}_t^{(w)} = \frac{1}{w}\sum_{i=t-w+1}^{t} x_i,
\qquad
s_t^{(w)} = \sqrt{\frac{1}{w-1}\sum_{i=t-w+1}^{t}\left(x_i - \bar{x}_t^{(w)}\right)^2}
$$

- **Rolling mean** $\bar{x}_t^{(w)}$ smooths out noise and reveals a shifting baseline
  (e.g., pressure slowly drifting up).
- **Rolling std** $s_t^{(w)}$ captures increasing *volatility/instability* — a bearing
  about to fail often vibrates more erratically, not just more strongly, so the variance
  itself is a predictive signal, not just the average.

### Lag Features
`create_lag_features()` uses `df[col].shift(lag)` to add the sensor's value from $k$
timesteps ago as a new column:

$$
x^{\text{lag}_k}_t = x_{t-k}, \qquad k \in \{1, 2\}
$$

This lets a row-based model (like RandomForest, which has no built-in memory) "see"
recent history and implicitly learn a rate-of-change/delta between the current and prior
states — e.g. $x_t - x^{\text{lag}_1}_t$ approximates the discrete derivative of the
sensor signal, even though the model never computes that subtraction explicitly; it just
has both raw values available to split on.

### Backward Fill (`bfill()`)
Rolling and lag operations leave `NaN` values at the start of the series (there's no
"previous 20 rows" for row 5). `bfill()` fills those leading NaNs with the next valid
observation so no rows are dropped and downstream models don't choke on missing values.

---

## 6. Model Training & Validation (`src/train.py`)

### Random Forest Classifier
An ensemble of many decision trees (`n_estimators=100`), each trained on a random subset
of data/features, with the final prediction being a majority vote. Chosen here as a
strong, low-effort baseline for tabular sensor data — it handles non-linear relationships
and mixed-scale features well and needs little tuning.

### Time-Series Cross-Validation (`TimeSeriesSplit`)
Ordinary k-fold cross-validation shuffles data randomly, which would let a model
"see the future" (train on data from *after* the test period) — a form of **data
leakage** that inflates validation scores unrealistically for time-ordered sensor data.
`TimeSeriesSplit(n_splits=5)` instead creates folds where training data always comes
*before* test data chronologically, giving a much more honest estimate of how the model
would perform on unseen future readings.

### Train/Test Split (chronological)
For the final reported score, the last 20% of rows (`split_idx = int(len(df) * 0.8)`) are
held out as a test set — again preserving time order rather than randomly shuffling, for
the same data-leakage reason as above.

### Class Imbalance & the F1-Score
Failures are rare events (e.g., only ~200/1000 rows show bearing issues in the source
data — an 80/20 split). **Accuracy** $= \frac{TP+TN}{TP+TN+FP+FN}$ is misleading here — a
model that always predicts "OK" would score 80% accuracy while being useless, because
accuracy doesn't distinguish *which* class the errors fall on.

Given the confusion-matrix counts — true/false positives/negatives ($TP$, $FP$, $TN$,
$FN$) for the failure class — define:

$$
\text{Precision} = \frac{TP}{TP+FP}, \qquad
\text{Recall} = \frac{TP}{TP+FN}
$$

**F1-score** is their harmonic mean, which stays low unless *both* are reasonably high
(unlike an arithmetic average, it punishes a model that's great at one and terrible at
the other):

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

This is why `train.py` logs `f1_score(..., average="weighted")` instead of accuracy.

In predictive maintenance specifically, **recall** on the failure class matters more than
**precision**: missing a real failure (false negative) risks unplanned downtime, while a
false alarm (false positive) just costs an unnecessary inspection. `PdM-Approach.ipynb`
shows this trade-off directly — its logistic regression baseline gets recall ≈ 0.89 on
the failure class but precision only ≈ 0.45, meaning it catches most real failures at the
cost of many extra (but cheap) warnings.

### `classification_report`
Prints per-class precision, recall, F1, and support in one table — used at the end of
`train()` for a fuller picture beyond the single F1 number.

---

## 7. Experiment Tracking — MLflow (`src/train.py`)

### Runs, Params, and Metrics
`mlflow.start_run()` opens a tracked "experiment run." Inside it:
- `mlflow.log_param(...)` records configuration (e.g., `model_type`, `features_count`) —
  things that describe *how* the model was built.
- `mlflow.log_metric(...)` records results (e.g., `f1_score`, per-fold F1s, `mean_cv_f1`) —
  things that describe *how well* it worked.

This turns ad-hoc "I think this run was better" comparisons into a searchable,
reproducible history — you can later compare any two runs' params/metrics side by side
via `mlflow ui`.

### Model Logging (`mlflow.sklearn.log_model`)
Serializes the trained model itself as an MLflow artifact tied to the run, so a specific
model version can be reloaded later for inference (see `src/predict.py`, currently a
stub) without retraining.

---

## 8. `PdM-Approach.ipynb` — The Notebook Behind the Pipeline

This is the **primary** notebook: it works through the problem the same way the
production pipeline (`src/ingestion.py` → `preprocessing.py` → `features.py` →
`train.py` → `predict.py`) is structured, and its findings directly explain *why* those
modules are built the way they are. (`NN-Approach.ipynb`, covered in section 9, is a
separate, secondary experiment and was **not** what the pipeline was built from.)

### Diagnostic EDA — Healthy vs. Faulty Comparison
Rather than generic `.describe()` on the whole dataset, the notebook splits rows by
label (`healthy_df = df[df['bearings'] == 0]`, `faulty_df = df[df['bearings'] == 1]`)
and compares each sensor's average between the two groups (noise, oil temperature,
vibration, motor power). This is a simple but effective form of **univariate feature
selection**: it tells you *which* sensors actually shift when a component is failing
before you build any model. The notebook finds noise, vibration, and motor power all
increase with faulty bearings, while oil tank temperature does not — which is why
`train.py`'s `sensors` list centers on `rpm`, `motor_power`, `torque`,
`outlet_pressure_bar`, `noise_db`, `outlet_temp`, `gaccx`, `haccx` (vibration- and
load-related signals) rather than every column in the dataset.

### Engineered Vibration Magnitude (Vector Norm)
```python
df['ground_acc_mag'] = np.sqrt(df['gaccx']**2 + df['gaccy']**2 + df['gaccz']**2)
```
The three ground-acceleration axes (`gaccx/y/z`) are combined into a single scalar via
the **Euclidean norm** (vector magnitude):

$$
\|\vec{a}\| = \sqrt{a_x^2 + a_y^2 + a_z^2}
$$

— the same math as computing the length of a 3D vector. This captures overall vibration
*intensity* regardless of which direction it's occurring in, collapsing 3 correlated
columns into 1 more interpretable one. This
specific composite feature is not yet ported into `src/features.py`, which instead
engineers rolling statistics on each raw axis independently — see the gap noted at the
end of this section.

### Predictive ("Failure Soon") Labeling
```python
df['bearing_failure_soon'] = df['bearings'].rolling(window=5).max().shift(-5)
```
This is the conceptual core of turning a *diagnostic* problem ("is it broken right now?")
into a *predictive* one ("will it break soon?"). For fault indicator $f_t \in \{0,1\}$ and
horizon $h=5$, the new label is:

$$
y_t = \max(f_{t+1}, f_{t+2}, \dots, f_{t+h})
$$

`rolling(window=5).max()` computes $\max(f_{t-h+1}, \dots, f_t)$ (a *trailing* max);
`.shift(-5)` then pulls that value $h$ steps **backward** in time so it lines up as a
*leading* max at row $t$ instead — i.e. row $t$ ends up labeled with whether a failure
occurs anywhere in the next 5 rows. Rows left with `NaN`
(near the end of the series, where there's no full future window) are dropped via
`dropna()`. This is the same `.shift()` mechanism `src/features.py`'s `create_lag_features`
uses for **backward**-looking lag features — here it's applied to the *target* and
pointed *forward* instead, which is what makes the resulting model genuinely predictive
rather than just diagnostic.

### Baseline Model: Logistic Regression + Chronological Split
The notebook's first model is `LogisticRegression` — a linear classifier that models the
probability of failure by passing a weighted sum of the features through the **sigmoid**
function:

$$
P(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

$\sigma(\cdot)$ squashes any real number into $(0, 1)$, so the linear combination
$w^\top x + b$ can be read as a probability of failure. Training fits $w, b$ by
minimizing **log loss** (binary cross-entropy) between predicted probabilities $\hat y_i$
and true labels $y_i$ over $n$ training rows — the same loss used by the PyTorch models
in §9:

$$
L = -\frac{1}{n}\sum_{i=1}^{n} \Big[ y_i \log(\hat y_i) + (1-y_i)\log(1-\hat y_i) \Big]
$$

trained with
`train_test_split(..., shuffle=False)` to keep rows in time order (a manual precursor to
`train.py`'s more rigorous `TimeSeriesSplit` and chronological 80/20 split, see section 6).
`train.py` later upgrades from this linear baseline to a `RandomForestClassifier` to
capture non-linear interactions between sensors.

### FFT (Fast Fourier Transform) for Vibration Analysis
The notebook's key insight, called out explicitly in its markdown: *"Bearings don't just
increase vibration, they vibrate at specific frequencies when damaged."* A rolling
mean/std (as in `features.py`) describes a signal's **time-domain** behavior (its level
and volatility) but says nothing about *which frequencies* the vibration is oscillating
at — that requires moving to the **frequency domain**, which is what FFT is for.

- **Time domain vs. frequency domain** — a time-domain signal is "value vs. time"
  (what the sensor literally reads at each instant). A frequency-domain signal is
  "energy vs. frequency" (how much the signal oscillates at each rate). FFT converts one
  into the other.
**Computing the FFT** — `np.fft.fft(signal)` computes the **Discrete Fourier
Transform** of a length-$N$ window $x_0, \dots, x_{N-1}$:

```math
X_k = \sum_{n=0}^{N-1} x_n \, e^{-i 2\pi k n / N}, \qquad k = 0, 1, \dots, N-1
```

Each complex coefficient $X_k$ represents how strongly frequency $k$ is present in the
window. `np.abs(X_k)` converts it to a magnitude, giving the **amplitude spectrum**
$|X_k|$.

- **Symmetry / using half the spectrum** — for a real-valued input signal (as all sensor
  readings are), the FFT output is conjugate-symmetric ($X_{N-k} = \overline{X_k}$, so
  $|X_{N-k}| = |X_k|$), so the notebook keeps only the first half
  (`fft_vals[:len(fft_vals)//2]`) — the second half is redundant.
**High-frequency energy feature** — `np.mean(fft_vals[int(len(fft_vals)*0.5):])`
averages the amplitude across the *upper* half of the frequency spectrum:

```math
E_{\text{high}} = \frac{1}{N/2 - N/4}\sum_{k=N/4}^{N/2-1} |X_k|
```

The hypothesis: damaged bearings push more vibration energy into higher frequencies
than smooth, healthy rotation.

- **Lesson learned — global FFT is wrong; use sliding windows.** Computing FFT once over
  the *entire* signal produces one single number reused for every row (as the notebook
  demonstrates — the feature comes out identical for both classes and is useless).
  Bearing degradation is a **local-in-time** phenomenon, so FFT has to be recomputed on
  small, overlapping windows of recent readings, not the whole history at once.
- **Windowed / sliding FFT** — the fix loops a fixed `window_size` ($N=20$ rows) across
  the signal, computes $X_k$ on just that slice, and assigns the resulting feature to the
  *last* row of the window (`df.loc[df.index[i], 'fft_high_freq_energy'] = ...`). This is
  structurally identical to the rolling-window pattern already in
  `src/features.py::create_rolling_features` (§5) — same sliding-window idea, just with
  an FFT instead of `.mean()`/`.std()` as the aggregation.
**Spectral spread feature** — `np.std(fft_vals)` per window measures how *dispersed*
the vibration energy is across frequencies (a wide, noisy spectrum vs. one concentrated
at a single peak):

```math
\sigma_{\text{spread}} = \sqrt{\frac{1}{N/2}\sum_{k=0}^{N/2-1}\left(|X_k| - \overline{|X|}\right)^2}
```

The notebook finds this rises slightly before failure, indicating the vibration
spectrum becomes less stable as a bearing degrades.

**Frequency bins (`np.fft.fftfreq`)** — converts raw FFT bin *index* $k$ into an actual
frequency in Hz, given an assumed sampling rate $f_s$ and window length $N$:

```math
f_k = \frac{k \cdot f_s}{N}, \qquad k = 0, 1, \dots, \tfrac{N}{2}-1
```

used only for plotting a human-readable "amplitude vs. frequency (Hz)" chart comparing
normal vs. soon-to-fail vibration.

- **Interpreting small feature gains** — adding the spectral-spread FFT feature to the
  logistic regression model improved accuracy only marginally (≈0.73 → ≈0.74) while
  recall on failures stayed ≈0.89. The notebook's explicit takeaway: a **small, stable**
  improvement from a new feature is a *good* sign (the original time-domain features were
  already strong); a sudden large jump from one added feature is usually a red flag for
  **data leakage or overfitting**, not a genuine breakthrough.

### Gap Between the Notebook and the Current Pipeline
`src/features.py` currently implements only the generic rolling mean/std and lag features
described in section 5. It does **not yet** port over two things explored in
`PdM-Approach.ipynb`: the engineered vibration-magnitude (vector-norm) feature, and the
windowed-FFT frequency-domain features (high-frequency energy, spectral spread). Adding
an FFT-based step to `FeatureEngineer` would be the natural next increment to bring the
production pipeline fully in line with the notebook's approach.

---

## 9. `NN-Approach.ipynb` — Supplementary Experiment (Not Part of the Pipeline)

This second notebook (mirrored in `docs/doc.md`) explores a different, more classical
ML/stats toolkit as a separate comparison — it is **not** what `src/` implements, but is
worth understanding since it lives in the same repo.

### Exploratory Data Analysis (EDA)
Checking `.isnull().sum()`, `.dtypes`, and `.groupby(...).size()` (class counts) are
standard first steps to understand a new dataset's shape, types, and quirks before
modeling.

### Stratified Train/Test Split
`train_test_split(..., stratify=y)` ensures the train and test sets have the *same
proportion* of failure vs. non-failure rows as the full dataset. Without this, a random
split of an already-imbalanced dataset could accidentally put almost all failure cases
into just one side.

### Standardization (`StandardScaler`)
Rescales each feature to zero mean and unit variance. This is a near-mandatory
preprocessing step before PCA (below), because PCA is sensitive to feature scale — a
feature measured in the thousands (like `motor_power`) would otherwise dominate the
variance calculation over one measured in single digits (like `torque`).

### Principal Component Analysis (PCA)
A dimensionality-reduction technique that finds new axes (**principal components**) which
are linear combinations of the original features, ordered by how much variance in the
data they explain. The notebook does this two ways.

**Manually**, via the covariance matrix and its **eigenvalues/eigenvectors**. For
standardized data $X$ (rows = samples, columns = features), the covariance matrix is:

```math
\Sigma = \frac{1}{n-1} X^\top X
```

(`np.cov`). Its eigenvectors $v_i$ and eigenvalues $\lambda_i$ satisfy:

```math
\Sigma v_i = \lambda_i v_i
```

(`np.linalg.eig`) — each eigenvector $v_i$ gives the *direction* of a new axis
(principal component), and its eigenvalue $\lambda_i$ gives how much variance the data
has along that direction. Sorting eigen-pairs by $\lambda_i$ descending and keeping the
top 2 gives the 2 most informative combined features; a sample $x$ is projected onto
component $i$ by $x \cdot v_i$ (a dot product).

**Via `sklearn.decomposition.PCA`** — the standard, production-ready equivalent of the
same math.

**Explained variance ratio** (`var_exp`, `cum_var_exp`) quantifies how much of the total
information is preserved by keeping only the top $m$ components out of $p$ total:
```math
\text{explained\_variance\_ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{p}\lambda_j},
\qquad
\text{cumulative}_m = \sum_{i=1}^{m}\frac{\lambda_i}{\sum_{j=1}^{p}\lambda_j}
```
the notebook notes the first two components already explain ~80% of the variance,
meaning most sensor readings move together (are correlated) rather than varying
independently.

**Loading vectors** (`get_most_contributing_features`) identify *which original sensors*
contribute most to each principal component — useful for interpreting an otherwise
abstract "PC1, PC2..." axis back in terms of real sensors (e.g., PC1 is dominated by
`gaccy`/`gaccz`, i.e., vibration).

### Handling Class Imbalance — SMOTE-Tomek
A combined resampling strategy.

**SMOTE** (Synthetic Minority Oversampling Technique) creates new *synthetic* examples
of the minority (failure) class by interpolating between existing minority samples,
rather than just duplicating them. For a minority sample $x_i$ and one of its $k$
nearest minority neighbors $x_{zi}$, a synthetic point is generated as:

```math
x_{\text{new}} = x_i + \lambda \cdot (x_{zi} - x_i), \qquad \lambda \sim \text{Uniform}(0,1)
```

i.e. a random point on the line segment between the two real minority samples.

**Tomek Links** then removes borderline/ambiguous pairs of opposite-class samples that
sit right next to each other — formally, $(x_i, x_j)$ from different classes form a
Tomek link if no other sample $x_k$ satisfies $d(x_i,x_k) < d(x_i,x_j)$ or
$d(x_j,x_k) < d(x_i,x_j)$ (neither is closer to any third point than they are to each
other) — cleaning up the decision boundary.

Together (`SMOTETomek`), this both grows the minority class and sharpens the boundary
between classes, aiming for a better-trained classifier on imbalanced data.

### Neural Network Classifier (PyTorch)
A small feed-forward network built from scratch to compare against classical models:
- `nn.Linear` layers compute an affine transform of the input: $z = Wx + b$, a
  fully-connected weight matrix $W$ plus bias $b$.
`Sigmoid` (hidden layer) and `Softmax` (output layer) are **activation functions**:

```math
\sigma(z) = \frac{1}{1+e^{-z}}, \qquad
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
```

Sigmoid introduces non-linearity so the network can learn more than straight lines;
softmax turns the 2 output values into class probabilities that sum to 1.

`CrossEntropyLoss` measures how wrong the predicted class probabilities $\hat y$ are
versus the true one-hot label $y$ — the standard loss for classification:

```math
L = -\sum_{c} y_c \log(\hat y_c)
```

`Adam` optimizer updates each weight $\theta$ using running averages of the gradient
($m_t$, first moment) and its square ($v_t$, second moment), giving each parameter its
own adaptive step size:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \qquad
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
```

where $g_t$ is the current gradient and $\hat m_t, \hat v_t$ are bias-corrected versions
of $m_t, v_t$ — this generally converges faster/more reliably than plain SGD
($\theta_t = \theta_{t-1} - \eta g_t$).

- **Epoch** = one full pass over the training data; **batch** = a small chunk of data
  processed per weight update (`DataLoader`/`TensorDataset` handle batching/shuffling).
- The **training loop** pattern (`forward → loss → loss.backward() → optimizer.step() →
  optimizer.zero_grad()`) is the standard PyTorch recipe: compute predictions, measure
  error, backpropagate gradients (via the chain rule, $\frac{\partial L}{\partial \theta}$
  for every parameter $\theta$), apply the update, then clear gradients before the next
  batch.

### ROC Curve & AUC
The **Receiver Operating Characteristic** curve plots, at every possible decision
threshold, the **True Positive Rate** against the **False Positive Rate**:
$$
\text{TPR} = \frac{TP}{TP+FN} \;(\text{= Recall}), \qquad
\text{FPR} = \frac{FP}{FP+TN}
$$
showing the trade-off between catching more real failures and raising more false alarms.
**AUC** (Area Under the Curve) condenses this into one number:
$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx
$$
— 0.5 means no better than random guessing, 1.0 means a perfect classifier — used here to
compare the plain-data model against the SMOTE-Tomek resampled one.

---

## 10. Deployment & Tooling

### FastAPI + Uvicorn (`api/main.py`, currently a stub)
FastAPI is a Python web framework for building APIs, typically used here to expose a
`/predict` endpoint that takes sensor readings and returns a failure prediction.
**Uvicorn** is the ASGI server that actually runs a FastAPI app (`uvicorn api.main:app`).
FastAPI's appeal for ML serving is automatic request validation (via Pydantic) and
auto-generated interactive docs (Swagger UI at `/docs`).

### Docker Containerization (`Dockerfile`)
Packages the app and its exact dependencies into a portable image
(`FROM python:3.10-slim`, install deps, run Uvicorn) so it runs identically on any
machine — the developer's laptop, a CI runner, or a production server — without "works on
my machine" issues.

### `uv` & `pyproject.toml`
`uv` is a fast Python package/project manager (an alternative to `pip`/`poetry`).
`pyproject.toml` declares the project's dependencies (`mlflow`, `scikit-learn`,
`ipykernel`) and required Python version; `uv.lock` pins exact resolved versions for
reproducible installs (`uv sync`).

### `.python-version`
Pins the Python interpreter version for the project so tools like `uv`/`pyenv`
automatically select the right one.

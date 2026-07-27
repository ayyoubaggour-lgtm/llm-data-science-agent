"""
run.py — Single-file pipeline. One command to generate, run, and fix code.

Usage:
    python run.py --data data/fatigue_data.json --request "fit a polynomial and plot it"
    python run.py --data data/fatigue_data.json --request "..." --max-fixes 3
"""

import os
import re
import sys
import json
import argparse
import subprocess
import requests
import time
import psutil
import threading


class PerformanceMonitor:
    """
    Samples CPU/RAM for this pipeline process and (if running) the Ollama
    process on a background thread, at a fixed interval, until stop() is
    called. Ollama CPU/RAM is tracked separately since it's the dominant
    consumer, in addition to the combined pipeline+Ollama totals.
    """

    SAMPLE_INTERVAL = 0.5  # seconds between samples

    def __init__(self):
        self.pipeline_process = psutil.Process(os.getpid())

        self.ollama_process = None
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if "ollama" in (p.info["name"] or "").lower():
                    self.ollama_process = psutil.Process(p.info["pid"])
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.running = False
        self.thread = None
        self.start_time = None

        # Combined (pipeline + ollama) samples used for the headline numbers
        self.cpu_samples = []
        self.memory_samples = []

        # Ollama-only samples, since that's the process that actually
        # consumes most of the CPU/RAM and you want it broken out
        self.ollama_cpu_samples = []
        self.ollama_memory_samples = []

        # Prime cpu_percent() for both processes. The first call after a
        # process object is created always returns 0.0/meaningless — psutil
        # needs a baseline to diff against on the *next* call.
        self.pipeline_process.cpu_percent()
        if self.ollama_process:
            try:
                self.ollama_process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.ollama_process = None

    def _sample_once(self):
        try:
            python_cpu = self.pipeline_process.cpu_percent()
            python_memory = self.pipeline_process.memory_info().rss / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            python_cpu, python_memory = 0.0, 0.0

        ollama_cpu, ollama_memory = 0.0, 0.0
        if self.ollama_process:
            try:
                ollama_cpu = self.ollama_process.cpu_percent()
                ollama_memory = self.ollama_process.memory_info().rss / 1024**2
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Ollama process died/restarted mid-run; stop tracking it
                self.ollama_process = None

        self.cpu_samples.append(python_cpu + ollama_cpu)
        self.memory_samples.append(python_memory + ollama_memory)
        self.ollama_cpu_samples.append(ollama_cpu)
        self.ollama_memory_samples.append(ollama_memory)

    def collect(self):
        while self.running:
            self._sample_once()
            time.sleep(self.SAMPLE_INTERVAL)

    def start(self):
        self.start_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

        # Guard against an empty sample list (e.g. an extremely fast run)
        # so max()/sum() below never crash
        if not self.cpu_samples:
            self._sample_once()

        return {
            "runtime_seconds": round(time.time() - self.start_time, 3),
            "peak_RAM_MB": round(max(self.memory_samples), 2),
            "average_CPU_percent": round(
                sum(self.cpu_samples) / len(self.cpu_samples), 2
            ),
            "maximum_CPU_percent": round(max(self.cpu_samples), 2),
            "ollama_peak_RAM_MB": round(max(self.ollama_memory_samples), 2),
            "ollama_average_CPU_percent": round(
                sum(self.ollama_cpu_samples) / len(self.ollama_cpu_samples), 2
            ),
            "ollama_maximum_CPU_percent": round(max(self.ollama_cpu_samples), 2),
        }


# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL          = "gemma4:e4b"
OLLAMA_URL     = "http://localhost:11434/api/chat"
TIMEOUT        = 600
MAX_FIXES      = 5
RESULTS_DIR    = "./results"
PLOT_PATH      = f"{RESULTS_DIR}/output_plot.png"
SCRIPT_PATH    = f"{RESULTS_DIR}/final_model.py"
TEMP_PATH      = "./temp/runtime.py"

# Single monitor instance shared for the whole pipeline run. Do NOT create a
# fresh PerformanceMonitor per LLM call — each one re-scans every running
# process to find Ollama, which is pure overhead, and its samples were never
# even read before.
monitor = PerformanceMonitor()

llm_tokens_total = 0
repair_iterations = 0

SYSTEM_GENERATOR = (
    "You generate scientific Python code. "
    "Return ONLY raw Python code. No markdown, no backticks, no explanation."
)

SYSTEM_FIXER = (
    "You are a debugging expert. "
    "Return ONLY the complete corrected Python script. "
    "No markdown, no backticks, no explanation, no truncation."
)

BASE_RULES = """
HARD RULES — follow every one exactly, no exceptions:

DATA LOADING
- Load data only from the file path provided in the schema — never hardcode values
- Use the exact loading code provided in the schema block, copy it verbatim, then select
  columns from the resulting DataFrame per the COLUMN SELECTION RULES in the schema
- Any numeric column used in computation must be a numpy array with dtype=float
- Column names come from the schema — never use placeholder names like col_x or col_y
- Never cast a categorical column straight to float — encode it first if a numeric
  method needs it (see COLUMN SELECTION RULES in the schema)

LIBRARIES
- Allowed: numpy, pandas, matplotlib, scipy, sklearn, json, os
- No other libraries, no pip installs
- Every library actually used in the script MUST be imported at the top — never call
  a function like os.makedirs(...) without first having `import os` at the top of
  the script; double-check every module.function() call has a matching import
- The "./results" output directory is ALREADY CREATED before this script runs —
  never call os.makedirs("./results") or os.makedirs(RESULTS_DIR) yourself, and
  therefore never import os purely for that purpose; just write directly to
  "./results/<filename>" (e.g. plt.savefig("./results/output_plot.png", ...))

CODE STRUCTURE
- Write all code flat, top to bottom — no helper functions, no def statements
- Every step must be inline in the script body
- Script must run completely with no user input
- Never call plt.show()
- If (and only if) matplotlib is used, set the backend before any matplotlib import:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt

VARIABLE CONSISTENCY
- Once a variable name is defined (e.g. a grid like x_range, a results array, a column
  list), it must keep the SAME shape/length and meaning everywhere it is used for the
  rest of the script — never redefine the same name later with a different length or
  a different set of points (e.g. do not create x_range with 200 points early on and
  then reuse the name x_range for a 300-point grid in a later plot)
- If a script genuinely needs two different grids/arrays for two different purposes
  (e.g. one grid for the uncertainty band, a different one for a second plot), give
  them clearly distinct names (e.g. x_range_fit vs x_range_final) — never reuse one
  name for two different shapes
- Before using any array in plt.plot/plt.fill_between/etc., the arrays being passed
  together must have matching lengths — double check this explicitly rather than
  assuming an earlier-computed array still matches

ERROR HANDLING
- Never wrap code in try/except purely to suppress an error and print a placeholder
  message like "Could not run X, skipping" — this hides the real problem instead of
  solving it and is treated as a FAILED script even if it exits with code 0
- If a computation can fail for a legitimate data reason (e.g. a categorical column has
  only one level so ANOVA/group comparison isn't meaningful), detect that condition
  explicitly BEFORE running the computation and print a clear, specific explanation of
  why it was skipped — never rely on catching the exception after the fact
- try/except is only acceptable around a narrow, specific operation where the failure
  mode is anticipated and explained — never as a catch-all around a whole analysis step

DATA LEAKAGE / RESULT VALIDITY
- When building a feature matrix X for any predictive model (regression, classification,
  etc.), the target/response column must be explicitly EXCLUDED — build X by dropping the
  target column by name (e.g. X = df.drop(columns=[target_col])), never by an inclusion
  list that could accidentally leave the target in
- Before reporting any fitted model's results, verify the target column name does not
  appear anywhere in the feature/coefficient list. If it does, this is a bug — fix the
  feature selection, do not report the result
- Treat a suspiciously perfect result (R² > 0.999, a coefficient of ~1.0 paired with all
  others near 0, zero residual error, or 100% classification accuracy) as a signal of a
  likely bug (usually target leakage or a degenerate/constant feature) rather than a
  genuine finding — print an explicit warning identifying the likely cause and, if the
  cause is target leakage, fix it and re-run rather than reporting the inflated number
- Weak or null results (low R², non-significant p-values, near-zero correlations) are
  valid, real findings — report them plainly and do not manufacture a stronger
  relationship than the data actually shows

DATA INTEGRITY
- Print the number of rows/samples used at the START of the analysis (immediately
  after loading) and again at any point rows are dropped or filtered (e.g. dropna,
  duplicate removal, outlier removal, encoding failures)
- If the row count used in the final analysis differs from the row count reported
  by the schema, print an explicit line stating how many rows were dropped and why
  (e.g. "Dropped 1 row: missing value in 'void_content_pct'") — never let rows
  silently disappear with no explanation
- Never use a mismatched or unexplained row count in a downstream computation
  (e.g. cluster counts that don't sum to the schema's record count) — investigate
  and explain it before reporting results

ID / INDEX COLUMNS (applies to every task)
- Before building any feature matrix (for modeling, clustering, or correlation),
  drop columns that are just row identifiers, not real measurements — e.g. a
  column named "Id"/"id"/"index"/"row_number", or any column that is simply a
  sequential integer 1..N with no measurement meaning. Including it as a feature
  adds meaningless signal that can distort distances, correlations, or coefficients
- Print which column(s) were dropped for this reason, if any

MISSING VALUES (applies to every task, not just ones that mention modeling)
- Before ANY numeric computation that cannot tolerate NaN — this includes
  correlation matrices, model.fit() of any kind (including regression used
  incidentally, e.g. for residual-based outlier detection), distance/clustering
  calculations, or statistical tests — check every column involved for NaN with
  df[col].isna().sum()
- If any NaNs are found, impute (numeric: fillna with that column's median) or
  drop the affected rows before the computation, and print what was done and how
  many values/rows were affected — do this unconditionally, do not assume a
  column is complete just because the task isn't explicitly a "prediction" task
- This check applies even if the user's request never mentions "missing values",
  "NaN", or "regression" by name — e.g. plain EDA/summary requests still need it
  if any statistic, plot, or auxiliary model touches a column with gaps

PRINTED OUTPUT
- Print at least one line summarizing what the script did

- Never read axis labels or metadata from the data file at runtime — use the
  column names and request description to write axis labels directly as strings
  in any plt.xlabel() and plt.ylabel() calls

PRINTED TABLES (e.g. comparing metrics across multiple models)
- Never reuse one .format()/f-string template for both the header row (strings)
  and the data rows (numbers). A template containing a numeric spec like
  {:<10.4f} will raise "Unknown format code 'f' for object of type 'str'" if
  applied to a text header — use a plain {:<10} (no numeric spec) for header
  cells, and only apply .4f-style specs to the actual numeric row values
- Simplest safe pattern: define the column width once, then build the header
  with plain string formatting (f"{'Model':<15}{'Accuracy':<10}...") and each
  data row separately with numeric formatting (f"{name:<15}{acc:<10.4f}...")
  rather than sharing a single template between both
""".strip()

PLOT_RULES = """
ADDITIONAL RULES FOR REQUESTS THAT ASK FOR A PLOT / VISUALIZATION — follow exactly:

PLOTTING
- Set backend before any matplotlib import:
      import matplotlib
      matplotlib.use("Agg")
      import matplotlib.pyplot as plt
- plt.xlabel(), plt.ylabel(), and plt.title() must use values from the schema and request
- Y-axis label must describe the actual response variable — never say "log scale"
- Save with: plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")
- Never call plt.show()

PLOTTING DISTRIBUTIONS ACROSS MULTIPLE COLUMNS (e.g. "each feature", "all columns")
- Never call ax.hist() / plt.hist() on a boolean-dtype column or array — this raises
  TypeError/KeyError in numpy (numpy.bool has no signed/unsigned mapping for
  histogram binning). This includes one-hot-encoded dummy columns (0/1 from
  pd.get_dummies) and any boolean outlier-flag masks
- Before looping over columns to plot, build the loop list explicitly with:
      numeric_cols = df.select_dtypes(include=["number"]).columns
  and exclude any column whose dtype is bool — check with df[col].dtype == bool
  or df[col].dtype == "boolean" and skip it, or cast with .astype(int) first if it
  truly needs to be shown
- If one-hot/dummy columns were created for this analysis, plot them as bar counts
  (value_counts().plot(kind="bar")) instead of a histogram, since a 0/1 histogram
  is not a meaningful distribution plot anyway
""".strip()

NO_PLOT_RULES = """
ADDITIONAL RULES — THIS REQUEST DOES NOT ASK FOR A PLOT OR VISUALIZATION:

- Do NOT import matplotlib, do NOT create any figure, plot, chart, or boxplot,
  and do NOT write to output_plot.png — none of that was requested
- Communicate every result as printed text/numbers only (tables, summary stats,
  test statistics, coefficients, etc.)
- If you are tempted to add a plot "for clarity", don't — text output only
""".strip()

FITTING_RULES = """
ADDITIONAL RULES FOR FITTING / MODELING REQUESTS — follow every one exactly, no exceptions:

CHOOSING LINEAR SPACE VS LOG SPACE — decide this FIRST, before fitting anything
- Print min(y) and inspect the shape of y vs x before deciding.
- Only fit in log space (log_y = np.log(y)) if BOTH of the following hold:
    1. min(y) > 0 strictly (log space is undefined/blows up at y <= 0)
    2. the response is expected to grow or decay monotonically and exponentially
       (e.g. fatigue cycles-to-failure, radioactive decay, population growth) —
       NOT a response that rises to a peak and then falls (e.g. stress-strain,
       load-deflection curves, dose-response curves), and NOT a response that
       includes zero or negative values anywhere
- If either condition fails, fit directly in LINEAR SPACE instead:
      coeffs    = np.polyfit(x, y, degree)
      y_hat     = np.polyval(coeffs, x)
      rss       = np.sum((y - y_hat) ** 2)
- Never silently attempt log space "just in case" — an unjustified log transform on
  data containing a zero, a negative value, or a peak/rollover shape will produce a
  numerically degenerate fit (coefficients blow up, np.exp() of them can reach
  physically impossible magnitudes like 1e100+). Treat this as a bug, not a valid result.
- State explicitly in the printed output which space was used and why
  (e.g. "Fitting in linear space: y contains a zero value and is non-monotonic")

POLYNOMIAL FITTING WITH AIC
- Compute AIC from residuals in whichever space was chosen above — never from data
  range or max values:
      for degree in [1, 2, 3]:
          coeffs = np.polyfit(x, y_fit, degree)     # y_fit = log_y or y, per the rule above
          y_hat  = np.polyval(coeffs, x)
          rss    = np.sum((y_fit - y_hat) ** 2)
          aic    = len(x) * np.log(rss / len(x)) + 2 * (degree + 1)
          print(f"  degree={degree}  AIC={aic:.2f}")
- Select the degree with the lowest AIC
- HARD CAP: never test or select a degree higher than 4, regardless of dataset size —
  do NOT use max(1, n//4) or any formula that scales degree up with more data points.
  In-sample AIC on noisy real-world data will keep decreasing as degree increases and
  will otherwise pick an overfit, wildly oscillating curve. If AIC keeps improving all
  the way to degree 4, stop there and report degree 4 — do not extend the search.
- After selecting the best degree, compute and print R² in the ORIGINAL (linear) units
  of y regardless of which space was fit in, so fit quality is always interpretable
- If the best-fitting model still has R² < 0.8, print an explicit note that the fit
  is weak and report it plainly — do not present a poor fit as if it were adequate
- SANITY CHECK ON CURVE SHAPE: after fitting, evaluate the fitted curve (y_hat) on a
  fine grid across the x range and check its first difference (np.diff). If the data
  itself is monotonic (all np.diff(y) values share the same sign, allowing small
  measurement noise) but the fitted curve's np.diff changes sign more than twice, this
  is a strong sign of overfitting/oscillation. In that case, drop back to the lowest
  degree that is still monotonic and print an explicit note explaining the fallback
  (e.g. "Degree 9 fit was non-monotonic despite monotonic data — falling back to a
  lower degree.").

ELASTIC-REGION / YOUNG'S MODULUS RULES (apply whenever the request involves a
stress-strain curve, load-deflection curve, or explicitly asks for a modulus/stiffness)
- NEVER fit Young's Modulus (or any "initial slope"/stiffness value) using the entire
  dataset. Physical elastic-region behavior only holds for the small initial portion of
  the curve; the rest is plastic deformation, plateau, or post-yield behavior with a
  completely different (usually near-zero or negative) local slope.
- Identify the elastic region programmatically, expanding from the first two points:
      elastic_n = 2
      for n in range(2, len(x)):
          coeffs_e = np.polyfit(x[:n], y[:n], 1)
          y_hat_e  = np.polyval(coeffs_e, x[:n])
          ss_res   = np.sum((y[:n] - y_hat_e) ** 2)
          ss_tot   = np.sum((y[:n] - np.mean(y[:n])) ** 2)
          r2_e     = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
          if r2_e >= 0.995:
              elastic_n = n + 1   # this many points still fit linearly to R² >= 0.995
          else:
              break                # stop at the first point that breaks linearity
      x_elastic = x[:elastic_n]
      y_elastic = y[:elastic_n]
- Fit Young's Modulus (the slope) using ONLY x_elastic / y_elastic — never the full
  x / y arrays:
      coeffs_elastic = np.polyfit(x_elastic, y_elastic, 1)
      youngs_modulus = coeffs_elastic[0]
- Print the strain range actually used for the elastic fit, e.g.:
      print(f"Elastic region used: {elastic_n} of {len(x)} points, "
            f"strain range [{x_elastic.min():.5f}, {x_elastic.max():.5f}]")
- PLAUSIBILITY CHECK: if the request concerns metallic materials, a computed modulus
  outside roughly 1e3 to 5e5 (in the units of the data, typically MPa) is almost
  certainly wrong — print an explicit warning flagging that the elastic-region
  detection may have failed or the units may differ from expectations. Do not silently
  present an implausible modulus as if it were correct.
- YIELD POINT: never report yield strength as simply np.max(y) — that is the ultimate
  strength, not the yield point, and conflating the two is a common, serious error.
  Instead use the standard 0.2% offset method:
      offset_line_full = coeffs_elastic[0] * (x - 0.002) + coeffs_elastic[1]
      diff = y - offset_line_full
      sign_changes = np.where(np.diff(np.sign(diff)))[0]
      if len(sign_changes) > 0:
          yield_idx = sign_changes[0]
          yield_stress = y[yield_idx]
          yield_strain = x[yield_idx]
      else:
          yield_stress = None   # no crossing found — report this explicitly, do not
          yield_strain = None   # fall back silently to max(y)
      if yield_stress is not None:
          print(f"Yield point (0.2% offset method): stress={yield_stress:.2f}, "
                f"strain={yield_strain:.5f}")
      else:
          print("WARNING: could not identify a yield point via the 0.2% offset method "
                "— the curve may not show a clear yield transition in this dataset.")

RESIDUAL BOOTSTRAP UNCERTAINTY BAND
- Use residual bootstrap — never case bootstrap (never resample x,y pairs)
- Work in whichever space (linear or log) was chosen above; variable names and shapes
  must match. Example shown for the LINEAR-space case (the common case for data with
  zeros or a peak, e.g. stress-strain):
      base_at_x  = np.polyval(coeffs, x)          # shape (n,) — fit at original x points
      residuals  = y - base_at_x                  # shape (n,) — residuals in linear space
      x_range    = np.linspace(x.min(), x.max(), 200)
      boot_preds = np.zeros((500, 200))
      rng        = np.random.default_rng(seed=42)
      for i in range(500):
          resampled     = rng.choice(residuals, size=len(x), replace=True)
          y_boot        = base_at_x + resampled    # shape (n,) — add to base at original x
          c             = np.polyfit(x, y_boot, best_degree)
          boot_preds[i] = np.polyval(c, x_range)   # shape (200,) — predict on grid
      lower = np.percentile(boot_preds, 3,  axis=0)
      upper = np.percentile(boot_preds, 97, axis=0)
      mean  = np.polyval(coeffs, x_range)
  If (and only if) log space was justified per the rule above, use log_y in place of y
  and exponentiate lower/upper/mean at the end (np.exp(...)), clipping only as a final
  safety step (np.clip(..., 0, y.max() * 2)) — never as a way to hide an already-bad fit
- residuals are always shape (n,) computed at original x — never at x_range
- Before plotting, sanity-check the band: if np.max(upper) is not within a reasonable
  multiple (e.g. 10x) of y.max(), or if any value in mean/lower/upper is inf or nan,
  this indicates a degenerate fit — print an explicit warning and fall back to linear
  space rather than plotting physically meaningless values

FITTING PLOT ELEMENTS
- Always include all of these in the plot, in addition to the base plotting rules:
      plt.scatter(x, y, color="steelblue", zorder=5, label="Data")
      plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% band")
      plt.plot(x_range, mean, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")
      plt.legend()
      plt.grid(True, linestyle="--", alpha=0.6)
- Let matplotlib auto-scale the axes from the actual data/fit values — never manually
  force axis limits to hide an out-of-range fit

FITTING PRINTED OUTPUT
- Print which space (linear or log) was used and why
- Print AIC for each degree tested
- Print selected degree and its AIC
- Print fitted coefficients
- Print R² in original (linear) units
- Print at least one key numeric result (R², slope, or model summary)
""".strip()

ML_REGRESSION_RULES = """
ADDITIONAL RULES FOR MULTI-FEATURE PREDICTIVE MODELING REQUESTS (predicting a target
column from several feature columns using sklearn) — follow every one exactly:

THIS IS NOT SINGLE-VARIABLE CURVE FITTING
- Do NOT use np.polyfit / np.polyval, AIC degree selection, or residual bootstrap bands —
  those rules are for fitting one x to one y and do not apply here
- Use ALL feature columns relevant to the request (not just the first two numeric columns)

TRAIN/TEST SPLIT
- from sklearn.model_selection import train_test_split
- Split with a fixed random_state (e.g. 42) and a reasonable test size (e.g. 0.2)
- Fit the model on the training set only; evaluate metrics on the held-out test set

MISSING VALUES
- Before fitting, check every feature column used for NaN with df[col].isna().sum()
- Never pass a matrix containing NaN into any sklearn estimator — LinearRegression,
  Ridge, and RandomForestRegressor all raise ValueError on NaN input; only
  HistGradientBoostingRegressor tolerates NaN natively, and only use that if the
  request specifically calls for it
- If any NaNs are found, handle them explicitly and print how many rows/values were
  affected and what was done (see DATA INTEGRITY rule for the exact wording):
    - Simplest, safest default: impute numeric NaNs with that column's median
      (e.g. df[col] = df[col].fillna(df[col].median())) — do this BEFORE the
      train/test split so both splits use the same fill value
    - Alternatively drop affected rows if the request implies a clean-data analysis,
      but then explicitly print how many rows were dropped
- Do this check for every numeric feature column, not just the ones you expect to
  have gaps — verify rather than assume a column is complete

MODEL
- Use sklearn (e.g. LinearRegression, Ridge, or RandomForestRegressor) as fits the request
- Encode categorical columns (e.g. pd.get_dummies) before fitting; never feed strings
  directly into an sklearn regressor
- Build X by dropping the target column by name — never by an inclusion list

METRICS
- from sklearn.metrics import r2_score, mean_squared_error
- Compute R² with r2_score(y_test, y_pred)
- Compute RMSE as np.sqrt(mean_squared_error(y_test, y_pred)) — do not assume
  mean_squared_error has a `squared` argument, since sklearn versions vary
- Print both R² and RMSE clearly labeled

FEATURE IMPORTANCE
- Linear models: report the fitted coefficients paired with their feature names,
  sorted by absolute value, as the importance ranking
- Tree/ensemble models: report `.feature_importances_` paired with feature names,
  sorted descending
- Print the top features by importance explicitly by name — do not just print a
  bare array; the reader needs to know which feature each value belongs to
""".strip()

CLASSIFICATION_RULES = """
ADDITIONAL RULES FOR CLASSIFICATION REQUESTS (predicting a discrete label/category,
e.g. species, class, yes/no) — follow every one exactly:

THIS IS NOT REGRESSION
- Do NOT use LinearRegression, Ridge, RandomForestRegressor, r2_score, or
  mean_squared_error/RMSE — those are for continuous numeric targets, not
  discrete class labels
- Do NOT treat the target as numeric or attempt to predict it as a continuous value

TRAIN/TEST SPLIT
- from sklearn.model_selection import train_test_split
- Use stratify=y in the split so class proportions are preserved in both sets,
  especially important for small or imbalanced datasets
- Use a fixed random_state (e.g. 42)

MODEL CHOICE AND MULTICLASS SAFETY
- If the request asks to compare algorithms, pick at least two of: LogisticRegression,
  DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, SVC, GaussianNB
- For LogisticRegression with 3+ classes, do NOT set solver='liblinear' (it only
  supports one-vs-rest and raises a ValueError for true multiclass); either omit the
  solver argument to use the default, which handles multiclass natively, or set
  solver='lbfgs' explicitly
- Encode the target labels if they are strings (e.g. LabelEncoder), and encode any
  categorical feature columns the same way as for regression tasks

METRICS
- from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
- For precision/recall/F1 on a multiclass target, pass average='macro' (or
  average='weighted' if the request implies class imbalance matters) — omitting
  `average` on a multiclass target raises an error or silently returns per-class
  arrays instead of a single number
- Print accuracy, precision, recall, and F1 clearly labeled, for each model compared
- Print the confusion matrix as a labeled array or table (row/column labels = class
  names) so it's readable without cross-referencing label encoding
- When comparing multiple models, print a clear side-by-side summary (e.g. one line
  per model with all four metrics) so the comparison is easy to read directly
""".strip()

CLUSTERING_RULES = """
ADDITIONAL RULES FOR CLUSTERING REQUESTS — follow every one exactly, no exceptions:

CHOOSING THE NUMBER OF CLUSTERS
- Never pick the number of clusters (K) as an arbitrary fixed number with no justification
- Justify K using a data-driven method: compute either the elbow method (inertia/WCSS
  across a range of K, e.g. K=2..8) or silhouette score across the same range (or both)
- Print the metric values tested (e.g. inertia or silhouette score per K) and print which
  K was selected and why (e.g. "K=4 selected: highest silhouette score of 0.31")
- If a natural elbow/peak isn't clear, say so explicitly rather than silently picking a
  number — report the ambiguity as part of the result
- Never override the number chosen by the elbow/silhouette method with a different
  number justified only by vague language like "domain knowledge" or "common
  heuristic" — if elbow and silhouette disagree with each other, pick whichever you
  find more convincing and say why in concrete terms (e.g. "the elbow at K=3 shows
  the steepest drop in WCSS, even though silhouette peaks at K=2"); do not invent a
  reason to select a K that neither method's own numbers actually support

FEATURE SELECTION FOR CLUSTERING
- Decide explicitly and intentionally whether categorical features are included
  (one-hot/encoded) or excluded, and print a one-line justification for that choice
  based on the request — don't silently drop or silently include them without saying so
- Scale/standardize all numeric features before clustering (e.g. StandardScaler) since
  K-Means is distance-based and unscaled features of different magnitudes will dominate
- Explicitly drop any ID/index-like column before fitting — a column named things like
  "Id", "id", "index", "row_number", or one that is just a sequential 1..N integer with
  no measurement meaning, must never be fed into the distance calculation. It carries
  no real signal but still affects distances once scaled, which can visibly distort
  which points get grouped together. Print which column(s) were dropped for this reason

REPORTING RESULTS
- Report cluster sizes as counts AND verify they sum to the total number of rows used
  in the fit (see DATA INTEGRITY rule) — investigate and explain any mismatch
- When printing centroid values, state clearly whether they are in scaled or original
  units, since scaled centroids are not directly interpretable as real-world values

COMPARING CLUSTERS TO TRUE LABELS (if the request has a ground-truth label column,
e.g. species/category, and asks to compare clusters against it)
- Cluster IDs from KMeans (0, 1, 2, ...) are arbitrary and have no inherent
  correspondence to the true label values — NEVER pass raw cluster IDs and raw
  true labels directly into accuracy_score/precision_score/recall_score/f1_score.
  Doing so produces a meaningless number (often near 0%) even when the clustering
  is qualitatively excellent, because cluster "0" might line up with true label
  "2" and the naive comparison scores that as entirely wrong
- Build a contingency table (pd.crosstab(true_labels, cluster_ids)) first, then
  map each cluster to whichever true label is its majority/mode within that
  cluster, and ONLY THEN compute accuracy/precision/recall/F1 on the remapped
  labels if the request calls for those metrics
- Alternatively (and often preferred, since it doesn't require this remapping),
  report clustering-appropriate agreement metrics instead: adjusted_rand_score
  and/or normalized_mutual_info_score from sklearn.metrics, which are invariant
  to the arbitrary numbering of cluster IDs and don't require alignment
- Either way, always print the contingency table/confusion matrix itself (rows =
  true label, columns = cluster ID) so the reader can see the actual overlap
  regardless of which summary metric is used
- When building any printed table header dynamically, use an f-string with the
  actual class/cluster names inserted (e.g. f"True \\\\ Cluster" is invalid syntax —
  use a plain string or escape correctly) and make sure the number of header
  columns exactly matches the number of data columns in each row
""".strip()

# Keywords that signal single-x/y CURVE fitting (polyfit/AIC/bootstrap band)
FITTING_KEYWORDS = [
    "fit", "fitting", "curve", "polynomial", "aic", "bootstrap",
    "uncertainty", "confidence", "extrapolat", "forecast", "trend",
]

# Keywords that signal a CLASSIFICATION task (predicting a discrete label/category)
# rather than a continuous numeric target
CLASSIFICATION_KEYWORDS = [
    "classify", "classifier", "classification", "class label", "species",
    "category", "categories", "accuracy", "precision", "recall", "f1",
    "f1-score", "confusion matrix", "logistic regression", "decision tree",
    "svm", "support vector", "knn", "k-nearest", "naive bayes",
]

# Keywords that signal multi-feature ML regression/prediction (sklearn, R²/RMSE,
# feature importance) rather than single-variable curve fitting
ML_REGRESSION_KEYWORDS = [
    "regression", "predict", "prediction", "r2", "r²", "rmse", "mse",
    "feature importance", "important feature", "important variable",
    "random forest", "sklearn", "train test", "train/test",
]


def wants_fitting(user_request):
    text = user_request.lower()
    return any(kw in text for kw in FITTING_KEYWORDS) and not wants_ml_regression(user_request) \
        and not wants_classification(user_request)


def wants_classification(user_request):
    text = user_request.lower()
    # Clustering takes precedence: comparing unsupervised clusters against true
    # labels is not itself a classification task, even though words like
    # "species"/"labels" overlap with the classification keyword set
    if wants_clustering(user_request):
        return False
    return any(kw in text for kw in CLASSIFICATION_KEYWORDS)


def wants_ml_regression(user_request):
    text = user_request.lower()
    # Classification takes precedence: predicting a species/category/label is not
    # regression even though "predict" appears in both keyword sets
    if wants_classification(user_request):
        return False
    return any(kw in text for kw in ML_REGRESSION_KEYWORDS)


# Keywords that signal the request wants clustering / unsupervised grouping
CLUSTERING_KEYWORDS = [
    "cluster", "clustering", "k-means", "kmeans", "k means", "dbscan",
    "hierarchical clustering", "unsupervised group", "segment", "segmentation",
]


def wants_clustering(user_request):
    text = user_request.lower()
    return any(kw in text for kw in CLUSTERING_KEYWORDS)


# Keywords that signal the request wants a visualization/plot output
PLOT_KEYWORDS = [
    "plot", "graph", "chart", "visuali", "figure", "scatter", "histogram",
    "draw", "show me", "display the data",
]


def wants_plot(user_request):
    text = user_request.lower()
    return any(kw in text for kw in PLOT_KEYWORDS)

# ── LLM CALL ─────────────────────────────────────────────────────────────────
def call_ollama(history, system, model=MODEL, think=False, num_predict=None):
    if num_predict is None:
        num_predict = 8000 if think else 5000  # thinking tokens eat into the budget first
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}] + history,
            "stream": False,
            "keep_alive": "30m",
            "think": think,
            "options": {"temperature": 0.2, "num_predict": num_predict, "num_ctx": 8192},
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    reply = data.get("message", {}).get("content", "").strip()
    done  = data.get("done_reason", "?")
    toks  = data.get("eval_count", 0)
    global llm_tokens_total
    llm_tokens_total += toks

    print(f"  [llm] done={done} tokens={toks}")
    return reply


def strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def strip_special_tokens(text):
    # Strip gemma/llama-style special tokens like <unused56>, <|endoftext|>, <end_of_turn>, etc.
    text = re.sub(r"<unused\d+>", "", text)
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(r"<end_of_turn>", "", text)
    return text.strip()


# ── SCHEMA ────────────────────────────────────────────────────────────────────
def read_schema(data_path):
    data_path = data_path.replace("\\", "/")
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".json":
        with open(data_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            list_key = next((k for k, v in raw.items() if isinstance(v, list)), None)
            records  = raw[list_key] if list_key else []
            meta     = {k: v for k, v in raw.items() if not isinstance(v, list)}
            fmt      = f"wrapped JSON — load with: dataset['{list_key}']"
        else:
            records, meta, fmt = raw, {}, "flat JSON list"
        cols = list(records[0].keys()) if records else []

    elif ext == ".csv":
        import pandas as pd
        df      = pd.read_csv(data_path)
        records = df.to_dict(orient="records")
        cols    = list(df.columns)
        meta    = {}
        fmt     = "CSV"

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    n = len(records)

    # Detect numeric vs categorical per column (don't assume col[0]/col[1] are x/y)
    def is_numeric(col):
        for r in records[:50]:
            v = r.get(col)
            if v is None:
                continue
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)):
                continue
            try:
                float(v)
            except (TypeError, ValueError):
                return False
        return True

    col_types = {col: ("numeric" if is_numeric(col) else "categorical") for col in cols}

    # Sample a few values per column so the model knows what it's working with
    col_info = []
    for col in cols:
        vals = [r[col] for r in records if r.get(col) is not None][:5]
        col_info.append(f"  '{col}' ({col_types[col]}): sample values = {vals}")

    # Provide generic loading code — the model selects whichever columns the
    # REQUEST actually needs, by exact name, and must only cast numeric
    # columns to float. Categorical columns must never be cast to float directly.
    if ext == ".json" and isinstance(raw, dict):
        load_code = (
            f'import pandas as pd\n'
            f'with open("{data_path}", encoding="utf-8") as f:\n'
            f'    dataset = json.load(f)\n'
            f'records = dataset["{list_key}"]\n'
            f'# Build a DataFrame so you can select whichever columns the request needs\n'
            f'df = pd.DataFrame(records)'
        )
    elif ext == ".json":
        load_code = (
            f'import pandas as pd\n'
            f'with open("{data_path}", encoding="utf-8") as f:\n'
            f'    records = json.load(f)\n'
            f'# Build a DataFrame so you can select whichever columns the request needs\n'
            f'df = pd.DataFrame(records)'
        )
    else:
        load_code = (
            f'import pandas as pd\n'
            f'df = pd.read_csv("{data_path}")'
        )

    schema_block = f"""FILE: {data_path}
FORMAT: {fmt}
RECORDS: {n}
COLUMNS (name, type, sample values):
{chr(10).join(col_info)}
TOP_LEVEL_FIELDS (not columns, do not load these as data): {meta}

DATA LOADING — copy this exactly, do not substitute column names:
{load_code}

COLUMN SELECTION RULES:
- Select only the columns relevant to the REQUEST below, by their exact names from the list above.
- Only cast 'numeric' columns to float (e.g. np.array(df["col"], dtype=float)).
- Never cast a 'categorical' column directly to float — if a categorical column is needed
  for a numeric method (e.g. PCA, regression, clustering), encode it first
  (e.g. pd.get_dummies(df["col"]) or pd.factorize(df["col"])[0]) before using it numerically.
- If the request implies a simple x/y relationship and doesn't name columns explicitly,
  default to the first two numeric columns in the list above as x and y."""

    return n, cols, schema_block

# ── CODE EXECUTION ────────────────────────────────────────────────────────────
def run_script(code):
    os.makedirs(os.path.dirname(TEMP_PATH), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(TEMP_PATH, "w", encoding="utf-8") as f:
        f.write(code)
    result = subprocess.run(
        [sys.executable, os.path.abspath(TEMP_PATH)],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode == 0, result.stdout, result.stderr


# Phrases that indicate a script caught an exception and printed a placeholder
# instead of actually handling/fixing the underlying problem
SUPPRESSED_ERROR_PATTERNS = [
    r"could not run.*skip",
    r"error.*skip(?:ping|ped)",
    r"failed.*skip(?:ping|ped)",
    r"unable to (?:run|compute|perform).*skip",
    r"\(error[:\s]",
    r"exception.*ignor",
]


def detect_suppressed_error(stdout):
    """Return the matching line if stdout looks like it hid a real exception
    behind a placeholder message, else None."""
    for line in stdout.splitlines():
        low = line.lower()
        for pat in SUPPRESSED_ERROR_PATTERNS:
            if re.search(pat, low):
                return line.strip()
    return None


# Keywords that indicate a line is reporting a model fit/accuracy metric
FIT_METRIC_KEYWORDS = ["r-squared", "r²", "rsquared", "r^2", " r2 ", "accuracy"]
_NUMBER_RE = re.compile(r"(\d+\.\d+)%?")


def detect_suspicious_result(stdout):
    """Return the matching line if stdout reports a suspiciously perfect fit
    (near-1.0 R-squared, 100% accuracy, etc.) that the script itself did not
    already flag as a likely leakage/bug warning."""
    low_full = stdout.lower()
    if "leakage" in low_full or "likely a bug" in low_full or "suspicious" in low_full:
        return None  # script already flagged it itself — that's the desired behavior
    for line in stdout.splitlines():
        low = f" {line.lower()} "
        if not any(kw in low for kw in FIT_METRIC_KEYWORDS):
            continue
        is_pct = "%" in line or "accuracy" in low
        for match in _NUMBER_RE.findall(line):
            val = float(match)
            threshold = 99.9 if is_pct else 0.999
            if val >= threshold:
                return line.strip()
    return None


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def pipeline(data_path, user_request, max_fixes=MAX_FIXES):

    monitor.start()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Remove stale outputs from a previous run so success reporting below
    # only ever reflects what THIS run actually produced
    for stale in (PLOT_PATH, SCRIPT_PATH):
        if os.path.exists(stale):
            os.remove(stale)

    # Schema
    print("\n=== SCHEMA ===")
    n, cols, schema_block = read_schema(data_path)
    print(f"  {n} records, columns: {cols}")

    # Decide whether this request needs fitting/modeling rules, clustering rules,
    # and/or a plot
    do_fitting = wants_fitting(user_request)
    do_classification = wants_classification(user_request)
    do_ml_regression = wants_ml_regression(user_request)
    do_clustering = wants_clustering(user_request)
    # A plot is implied by curve-fitting requests; ML regression/classification
    # requests don't require one unless the user also used plot language
    do_plot = wants_plot(user_request) or do_fitting
    active_rules = BASE_RULES
    if do_plot:
        active_rules += "\n\n" + PLOT_RULES
    else:
        active_rules += "\n\n" + NO_PLOT_RULES
    if do_fitting:
        active_rules += "\n\n" + FITTING_RULES
    if do_classification:
        active_rules += "\n\n" + CLASSIFICATION_RULES
    elif do_ml_regression:
        active_rules += "\n\n" + ML_REGRESSION_RULES
    if do_clustering:
        active_rules += "\n\n" + CLUSTERING_RULES
    print(f"  Plot rules: {'ON' if do_plot else 'OFF'}   Fitting rules: {'ON' if do_fitting else 'OFF'}"
          f"   Classification rules: {'ON' if do_classification else 'OFF'}"
          f"   ML regression rules: {'ON' if do_ml_regression and not do_classification else 'OFF'}"
          f"   Clustering rules: {'ON' if do_clustering else 'OFF'}")

    rule_note = (
        f"plotting {'required — include matplotlib output as before' if do_plot else 'FORBIDDEN — do not use matplotlib at all, text/printed output only'}, "
        f"fitting/AIC/bootstrap {'required' if do_fitting else 'not required'}, "
        f"classification (stratified split, accuracy/precision/recall/F1, confusion matrix) {'required' if do_classification else 'not required'}, "
        f"ML regression (train/test split, R²/RMSE, feature importance) {'required' if do_ml_regression and not do_classification else 'not required'}, "
        f"clustering (elbow/silhouette K justification) {'required' if do_clustering else 'not required'}"
    )

    # Build generation prompt
    gen_prompt = (
        f"Write a single standalone Python script for this request:\n\n"
        f"REQUEST: {user_request}\n\n"
        f"DATA:\n{schema_block}\n\n"
        f"{active_rules}\n\n"
        f"Return ONLY the raw Python script."
    )

    # Generate
    print("\n=== GENERATING ===")
    history = [{"role": "user", "content": gen_prompt}]
    raw = call_ollama(history, SYSTEM_GENERATOR)
    code = strip_special_tokens(strip_fences(raw))
    history.append({"role": "assistant", "content": code})
    print(f"  Generated {len(code.splitlines())} lines.")

    if len(code.splitlines()) < 5:
        print("  ✗ Response too short — check model/connection.")
        monitor.stop()
        return

    # Run
    print("\n=== RUN ATTEMPT 1 ===")
    ok, stdout, stderr = run_script(code)

    suppressed = detect_suppressed_error(stdout) if ok else None
    if suppressed:
        ok = False
        stderr = (
            f"Script exited with code 0 but appears to have silently caught an "
            f"exception and printed a placeholder instead of handling it:\n"
            f"  \"{suppressed}\"\n"
            f"This violates the ERROR HANDLING rule. Do not use a broad try/except "
            f"to hide the failure — detect the actual data condition causing it and "
            f"handle it explicitly, or fix the underlying computation."
        )

    suspicious = detect_suspicious_result(stdout) if ok else None
    if suspicious:
        ok = False
        stderr = (
            f"Script exited with code 0 but reported a suspiciously perfect result "
            f"without explanation:\n"
            f"  \"{suspicious}\"\n"
            f"This is almost always target leakage (the target column ended up in the "
            f"feature matrix) or a degenerate/constant feature. This violates the "
            f"DATA LEAKAGE / RESULT VALIDITY rule. Check that the target column was "
            f"explicitly dropped from X before fitting, fix the feature selection, "
            f"and re-run. Do not report the inflated number."
        )

    if ok:
        print("  ✓ Success.")
        if stdout.strip():
            print(stdout.strip())
    else:
        print(f"  ✗ Error:\n{stderr.strip()}")

    # Fix loop
    # NOTE: we deliberately do NOT keep appending to a single growing `history`
    # list across fix attempts. Each fix call gets a fresh, minimal 3-message
    # history (original prompt + current code + current error) instead of the
    # full multi-turn back-and-forth. This keeps every call well under num_ctx
    # regardless of how many fix attempts have happened, which avoids the model
    # silently losing track of earlier variable definitions (e.g. redefining
    # x_range with a different length halfway through a long script) once the
    # conversation would otherwise have exceeded the context window.
    global repair_iterations
    attempt = 0
    while not ok and attempt < max_fixes:
        attempt += 1
        repair_iterations += 1
        print(f"\n=== FIX ATTEMPT {attempt} ===")

        fix_prompt = (
            f"This script failed with the error below. Fix it.\n\n"
            f"ERROR:\n{stderr.strip()}\n\n"
            f"Return the complete corrected script, keeping all original functionality "
            f"and following the same rules as before ({rule_note})."
        )
        fix_history = [
            {"role": "user", "content": gen_prompt},
            {"role": "assistant", "content": code},
            {"role": "user", "content": fix_prompt},
        ]
        raw_fix = call_ollama(fix_history, SYSTEM_FIXER)
        fixed   = strip_special_tokens(strip_fences(raw_fix))

        # Reject empty or heavily truncated responses
        if not fixed or (len(code.splitlines()) > 10 and
                         len(fixed.splitlines()) < len(code.splitlines()) * 0.6):
            print("  ⚠ Response empty or truncated — retrying.")
            continue

        code = fixed
        print(f"  Received {len(code.splitlines())} lines. Re-running...")

        ok, stdout, stderr = run_script(code)

        suppressed = detect_suppressed_error(stdout) if ok else None
        if suppressed:
            ok = False
            stderr = (
                f"Script exited with code 0 but appears to have silently caught an "
                f"exception and printed a placeholder instead of handling it:\n"
                f"  \"{suppressed}\"\n"
                f"This violates the ERROR HANDLING rule. Do not use a broad try/except "
                f"to hide the failure — detect the actual data condition causing it and "
                f"handle it explicitly, or fix the underlying computation."
            )

        suspicious = detect_suspicious_result(stdout) if ok else None
        if suspicious:
            ok = False
            stderr = (
                f"Script exited with code 0 but reported a suspiciously perfect result "
                f"without explanation:\n"
                f"  \"{suspicious}\"\n"
                f"This is almost always target leakage (the target column ended up in the "
                f"feature matrix) or a degenerate/constant feature. This violates the "
                f"DATA LEAKAGE / RESULT VALIDITY rule. Check that the target column was "
                f"explicitly dropped from X before fitting, fix the feature selection, "
                f"and re-run. Do not report the inflated number."
            )

        if ok:
            print("  ✓ Fixed.")
            if stdout.strip():
                print(stdout.strip())
        else:
            print(f"  ✗ Still failing:\n{stderr.strip()}")

    # Save
    print("\n=== SAVE ===")
    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Script → {SCRIPT_PATH}")

    if ok:
        print("\n✓ DONE")
        if os.path.exists(PLOT_PATH):
            print(f"  Plot   → {PLOT_PATH}")
    else:
        print("\n⚠ Pipeline finished without a passing script — saved best attempt.")

    perf = monitor.stop()

    print("\n=== PERFORMANCE ===")
    print(f"Runtime: {perf['runtime_seconds']} seconds")
    print(f"Peak RAM: {perf['peak_RAM_MB']} MB") #ollama + pipeline
    print(f"Average CPU: {perf['average_CPU_percent']} %") #ollama + pipeline
    print(f"Maximum CPU: {perf['maximum_CPU_percent']} %") #ollama + pipeline
    print(f"Ollama peak RAM: {perf['ollama_peak_RAM_MB']} MB")
    print(f"LLM tokens: {llm_tokens_total}")
    print(f"Repair iterations: {repair_iterations}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             required=True,  help="Path to data file (JSON or CSV)")
    parser.add_argument("--request",          required=True,  help="What to do with the data")
    parser.add_argument("--max-fixes",        type=int, default=MAX_FIXES)
    args = parser.parse_args()

    pipeline(args.data, args.request, max_fixes=args.max_fixes)
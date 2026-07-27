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

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL          = "gemma4:e4b"
REVIEWER_MODEL = "phi4-mini"  # Microsoft, domestic — genuinely different model family from the generator
# plain "phi4-mini" is the instruct model and does NOT support Ollama's "think" param
# (returns HTTP 400 if you try). Only reasoning-tagged variants like
# "phi4-mini-reasoning" or "phi4-reasoning" support it. Set this True only if you
# switch REVIEWER_MODEL to one of those.
REVIEWER_SUPPORTS_THINK = False
OLLAMA_URL     = "http://localhost:11434/api/chat"
TIMEOUT        = 600
MAX_FIXES      = 2
MAX_REVIEW_FIXES = 3  # separate budget for review-driven fix attempts
RESULTS_DIR    = "./results"
PLOT_PATH      = f"{RESULTS_DIR}/output_plot.png"
SCRIPT_PATH    = f"{RESULTS_DIR}/final_model.py"
REVIEW_PATH    = f"{RESULTS_DIR}/review_log.json"
TEMP_PATH      = "./temp/runtime.py"

SYSTEM_GENERATOR = (
    "You generate scientific Python code. "
    "Return ONLY raw Python code. No markdown, no backticks, no explanation."
)

SYSTEM_FIXER = (
    "You are a debugging expert. "
    "Return ONLY the complete corrected Python script. "
    "No markdown, no backticks, no explanation, no truncation."
)

SYSTEM_REVIEWER = (
    "You are a skeptical senior statistician and code reviewer. You will be shown a "
    "user's data-analysis request, the schema of their data, a Python script that was "
    "written to satisfy the request, and that script's printed output. Your job is to "
    "find real problems, not to be agreeable.\n\n"
    "Check specifically for:\n"
    "1. STATISTICAL METHOD APPROPRIATENESS — is the test/method used actually correct "
    "for this data type and this request (e.g. not running a t-test on >2 groups, not "
    "assuming normality without checking, not treating ordinal/categorical data as "
    "continuous, using the right test for the study design)?\n"
    "2. NUMERICAL PLAUSIBILITY — do the printed numbers look internally consistent and "
    "plausible given the data described in the schema (not necessarily re-deriving them, "
    "but sanity-checking magnitudes, signs, ranges, and whether stated conclusions match "
    "the stated numbers)?\n"
    "3. DATA LEAKAGE — could the target/response variable have leaked into the "
    "predictors, even subtly (e.g. a near-duplicate or derived column, a feature that "
    "encodes the same information as the target)? Suspiciously strong results (even if "
    "not literally R²=1.0, e.g. R²=0.95 on inherently noisy physical data) deserve "
    "scrutiny here.\n"
    "4. INTERPRETATION SOUNDNESS — does the prose explanation/conclusion in the output "
    "actually follow from the numbers, or does it overstate/understate/misstate what "
    "was found?\n\n"
    "Return ONLY a single JSON object, no markdown fences, no other text, in exactly "
    "this shape:\n"
    '{"verdict": "PASS" or "FLAG", "concerns": ["specific concern 1", ...], '
    '"statistical_method_ok": true/false, "leakage_risk": true/false, '
    '"interpretation_ok": true/false}\n'
    "Use \"FLAG\" if there is any concern worth a human's attention, even a minor one — "
    "list it in concerns rather than silently passing. Use \"PASS\" only if you have no "
    "real concerns. Keep each concern to one specific, actionable sentence."
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

PRINTED OUTPUT
- Print at least one line summarizing what the script did

- Never read axis labels or metadata from the data file at runtime — use the
  column names and request description to write axis labels directly as strings
  in any plt.xlabel() and plt.ylabel() calls
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

POLYNOMIAL FITTING WITH AIC
- Always fit in log space when the response variable represents counts, cycles, or
  quantities that must stay positive: log_y = np.log(y)
- Compute AIC from residuals — never from data range or max values:
      for degree in [1, 2]:
          coeffs    = np.polyfit(x, log_y, degree)
          log_y_hat = np.polyval(coeffs, x)
          rss       = np.sum((log_y - log_y_hat) ** 2)
          aic       = len(x) * np.log(rss / len(x)) + 2 * (degree + 1)
          print(f"  degree={degree}  AIC={aic:.2f}")
- Select the degree with the lowest AIC
- Cap degree at max(1, n//4) to prevent overfitting on small datasets

RESIDUAL BOOTSTRAP UNCERTAINTY BAND
- Use residual bootstrap — never case bootstrap (never resample x,y pairs)
- Follow this exactly, variable names and shapes must match:
      base_at_x  = np.polyval(coeffs, x)          # shape (n,) — fit at original x points
      residuals  = log_y - base_at_x              # shape (n,) — log space residuals
      x_range    = np.linspace(x.min(), x.max(), 200)
      boot_preds = np.zeros((500, 200))
      rng        = np.random.default_rng(seed=42)
      for i in range(500):
          resampled     = rng.choice(residuals, size=len(x), replace=True)
          log_y_boot    = base_at_x + resampled    # shape (n,) — add to base at original x
          c             = np.polyfit(x, log_y_boot, best_degree)
          boot_preds[i] = np.polyval(c, x_range)   # shape (200,) — predict on grid
      lower = np.clip(np.exp(np.percentile(boot_preds, 3,  axis=0)), 0, None)
      upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y.max() * 2)
      mean  = np.exp(np.polyval(coeffs, x_range))
- residuals are always shape (n,) computed at original x — never at x_range

FITTING PLOT ELEMENTS
- Always include all of these in the plot, in addition to the base plotting rules:
      plt.scatter(x, y, color="steelblue", zorder=5, label="Data")
      plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% band")
      plt.plot(x_range, mean, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")
      plt.legend()
      plt.grid(True, linestyle="--", alpha=0.6)

FITTING PRINTED OUTPUT
- Print AIC for each degree tested
- Print selected degree and its AIC
- Print fitted coefficients
- Print at least one key numeric result (R², slope, or model summary)
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

FEATURE SELECTION FOR CLUSTERING
- Decide explicitly and intentionally whether categorical features are included
  (one-hot/encoded) or excluded, and print a one-line justification for that choice
  based on the request — don't silently drop or silently include them without saying so
- Scale/standardize all numeric features before clustering (e.g. StandardScaler) since
  K-Means is distance-based and unscaled features of different magnitudes will dominate

REPORTING RESULTS
- Report cluster sizes as counts AND verify they sum to the total number of rows used
  in the fit (see DATA INTEGRITY rule) — investigate and explain any mismatch
- When printing centroid values, state clearly whether they are in scaled or original
  units, since scaled centroids are not directly interpretable as real-world values
""".strip()

# Keywords that signal the request wants a fit/model/trend, not just a raw-data plot
FITTING_KEYWORDS = [
    "fit", "fitting", "model", "modeling", "modelling", "trend", "regression",
    "predict", "prediction", "curve", "polynomial", "aic", "bootstrap",
    "uncertainty", "confidence", "extrapolat", "forecast",
]


def wants_fitting(user_request):
    text = user_request.lower()
    return any(kw in text for kw in FITTING_KEYWORDS)


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
    toks  = data.get("eval_count", "?")
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


# ── REVIEWER (LLM-based sanity check) ──────────────────────────────────────────
def run_reviewer(user_request, schema_block, code, stdout):
    """Ask a (potentially separate) LLM to critique the passing script for
    statistical appropriateness, numerical plausibility, leakage risk, and
    interpretation soundness. Returns a dict with at least a 'verdict' key.
    Never raises — falls back to a neutral PASS-with-note on any failure,
    since a broken reviewer should never block an otherwise-passing pipeline."""
    review_prompt = (
        f"REQUEST: {user_request}\n\n"
        f"DATA SCHEMA:\n{schema_block}\n\n"
        f"SCRIPT:\n{code}\n\n"
        f"PRINTED OUTPUT:\n{stdout.strip()}\n\n"
        f"Review this as instructed. Return ONLY the JSON object."
    )
    try:
        raw = call_ollama(
            [{"role": "user", "content": review_prompt}],
            SYSTEM_REVIEWER,
            model=REVIEWER_MODEL,
            think=REVIEWER_SUPPORTS_THINK,
        )
        cleaned = strip_special_tokens(strip_fences(raw))
        # Defensive: some models inline reasoning as <think>...</think> in content
        # even when a separate thinking field is expected — strip it if present.
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        # If there's leftover prose before/after the JSON object, isolate the braces.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
        result = json.loads(cleaned)
        result.setdefault("verdict", "FLAG")
        result.setdefault("concerns", [])
        return result
    except Exception as e:
        return {
            "verdict": "PASS",
            "concerns": [],
            "note": f"Reviewer step failed to run/parse ({e}) — not blocking on this basis.",
        }


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def pipeline(data_path, user_request, max_fixes=MAX_FIXES, max_review_fixes=MAX_REVIEW_FIXES):
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
    do_clustering = wants_clustering(user_request)
    do_plot = wants_plot(user_request) or do_fitting  # fitting implies a plot of the fit
    active_rules = BASE_RULES
    if do_plot:
        active_rules += "\n\n" + PLOT_RULES
    else:
        active_rules += "\n\n" + NO_PLOT_RULES
    if do_fitting:
        active_rules += "\n\n" + FITTING_RULES
    if do_clustering:
        active_rules += "\n\n" + CLUSTERING_RULES
    print(f"  Plot rules: {'ON' if do_plot else 'OFF'}   Fitting rules: {'ON' if do_fitting else 'OFF'}"
          f"   Clustering rules: {'ON' if do_clustering else 'OFF'}")

    rule_note = (
        f"plotting {'required — include matplotlib output as before' if do_plot else 'FORBIDDEN — do not use matplotlib at all, text/printed output only'}, "
        f"fitting/AIC/bootstrap {'required' if do_fitting else 'not required'}, "
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
    attempt = 0
    while not ok and attempt < max_fixes:
        attempt += 1
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

    # Reviewer pass — only meaningful once the script actually runs cleanly.
    # A broken/failing script has bigger problems than a statistical review.
    review = None
    if ok:
        print("\n=== REVIEW ===")
        review = run_reviewer(user_request, schema_block, code, stdout)
        verdict = review.get("verdict", "FLAG")
        concerns = review.get("concerns", [])
        print(f"  Verdict: {verdict}")
        for c in concerns:
            print(f"  ⚠ {c}")
        if review.get("note"):
            print(f"  ({review['note']})")

        review_attempt = 0
        while verdict == "FLAG" and concerns and review_attempt < max_review_fixes:
            review_attempt += 1
            print(f"\n=== REVIEW FIX ATTEMPT {review_attempt} ===")
            concern_text = "\n".join(f"- {c}" for c in concerns)
            review_fix_prompt = (
                f"A statistical reviewer examined this script's output and raised these "
                f"concerns:\n{concern_text}\n\n"
                f"Return the complete corrected script that addresses these concerns, "
                f"keeping all original functionality and following the same rules as "
                f"before ({rule_note})."
            )
            review_fix_history = [
                {"role": "user", "content": gen_prompt},
                {"role": "assistant", "content": code},
                {"role": "user", "content": review_fix_prompt},
            ]
            raw_fix = call_ollama(review_fix_history, SYSTEM_FIXER)
            fixed = strip_special_tokens(strip_fences(raw_fix))
            if not fixed:
                print("  ⚠ Empty response — keeping prior version.")
                break
            code = fixed
            new_ok, new_stdout, new_stderr = run_script(code)
            if not new_ok:
                print(f"  ✗ Review-fix broke the script — reverting:\n{new_stderr.strip()}")
                # revert to the last known-good code/stdout, keep original review
                break
            stdout = new_stdout
            print(f"  Re-ran successfully. Re-reviewing...")
            review = run_reviewer(user_request, schema_block, code, stdout)
            verdict = review.get("verdict", "FLAG")
            concerns = review.get("concerns", [])
            print(f"  Verdict: {verdict}")
            for c in concerns:
                print(f"  ⚠ {c}")

        # Persist the review regardless of outcome
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(REVIEW_PATH, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2)

    # Save
    print("\n=== SAVE ===")
    with open(SCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"  Script → {SCRIPT_PATH}")

    if ok:
        review_verdict = review.get("verdict", "?") if review is not None else None
        if review_verdict == "FLAG":
            print("\n⚠ DONE WITH UNRESOLVED CONCERNS — script runs, but the reviewer "
                  "still has open concerns after the review-fix budget was used up.")
            print(f"  See {REVIEW_PATH} before trusting these results.")
        else:
            print("\n✓ DONE")
        if os.path.exists(PLOT_PATH):
            print(f"  Plot   → {PLOT_PATH}")
        if review is not None:
            print(f"  Review → {REVIEW_PATH} (verdict: {review_verdict})")
    else:
        print("\n⚠ Pipeline finished without a passing script — saved best attempt.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             required=True,  help="Path to data file (JSON or CSV)")
    parser.add_argument("--request",          required=True,  help="What to do with the data")
    parser.add_argument("--max-fixes",        type=int, default=MAX_FIXES)
    parser.add_argument("--max-review-fixes", type=int, default=MAX_REVIEW_FIXES,
                         help="How many times to send the reviewer's concerns back for correction")
    args = parser.parse_args()

    pipeline(args.data, args.request, max_fixes=args.max_fixes, max_review_fixes=args.max_review_fixes)
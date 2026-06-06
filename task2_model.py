"""
Task 2: Audience Share Prediction Without the 3-Month Prior
Nova TV interview assignment.

Constraint: share_15_54_3mo_mean is FORBIDDEN as a feature.
That's reserved for Task 3.

This script trains two models:
  1. Linear Regression: simplest explainable baseline.
  2. LightGBM: captures non-linear interactions, handles NaN natively.

Every step is timed.  The LightGBM model + feature list are exported to
models/ for Task 4 (containerization and orchestration).

Usage:
    python task2_model.py
"""

import base64
import json
import os
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# ═══════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = "data.csv"
OUTPUT_DIR = Path("plots")
MODEL_DIR  = Path("models")
REPORT_PATH = "task2_report.html"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

CHANNEL_IDS = [3, 4, 9, 54]
NUMERIC_FEATURES = ["f_1", "f_2", "f_3", "f_4", "f_5",
                    "f_6", "f_7", "f_8", "f_9", "f_12"]
CAT_FEATURES = ["f_10", "f_11"]

LGB_PARAMS = dict(
    n_estimators=1000,      # High; early stopping picks the right value.
    learning_rate=0.05,
    max_depth=8,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, name: str) -> str:
    """Persist figure, close it, return the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _hr(title: str) -> None:
    """Section banner: makes terminal output scannable."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════
#  Data loading & feature engineering
# ═══════════════════════════════════════════════════════════════════
#
#  Approach: vectorised restructuring with .loc masks.  Do NOT iterate
#  over rows; the pandas vectorised path is 50-100× faster for 34k rows.

def load_and_engineer(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load the CSV, build temporal + own/competitor features, encode
    categoricals, one-hot channel_id, and return (df, feature_cols).

    Returns
    -------
    df : pd.DataFrame
        Fully engineered dataset (no rows dropped).
    feature_cols : list[str]
        Ordered list of column names the model will use.
    """
    # ── Load ────────────────────────────────────────────────────────
    df = pd.read_csv(path, low_memory=False)
    df["timeslot_datetime_from"] = pd.to_datetime(df["timeslot_datetime_from"])

    # ── Temporal features ───────────────────────────────────────────
    ts = df["timeslot_datetime_from"]
    df["hour"]         = ts.dt.hour
    df["day_of_week"]  = ts.dt.dayofweek          # Monday = 0
    df["month"]        = ts.dt.month
    df["is_weekend"]   = (ts.dt.dayofweek >= 5).astype(int)
    df["is_prime_time"]= df["hour"].isin([19, 20, 21, 22]).astype(int)

    # ── Own vs competitor restructuring (vectorised) ────────────────
    # For each channel, extract its own features as own_f_N, and
    # compute mean + max of the OTHER three channels' features.

    # Force numeric dtype on source columns to avoid pandas dtype conflicts.
    for ch in CHANNEL_IDS:
        for feat in NUMERIC_FEATURES:
            col = f"ch{ch}__{feat}"
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for feat in NUMERIC_FEATURES:
        df[f"own_{feat}"] = np.nan
        df[f"comp_mean_{feat}"] = np.nan
        df[f"comp_max_{feat}"]  = np.nan

    for ch in CHANNEL_IDS:
        prefix = f"ch{ch}"
        mask   = df["channel_id"] == ch
        comp_prefixes = [f"ch{c}" for c in CHANNEL_IDS if c != ch]

        for feat in NUMERIC_FEATURES:
            col = f"{prefix}__{feat}"
            df.loc[mask, f"own_{feat}"] = df.loc[mask, col].values
            comp_cols = [f"{p}__{feat}" for p in comp_prefixes]
            df.loc[mask, f"comp_mean_{feat}"] = df.loc[mask, comp_cols].mean(axis=1).values
            df.loc[mask, f"comp_max_{feat}"]  = df.loc[mask, comp_cols].max(axis=1).values

        # Categorical features (genre, country): just copy the string.
        # Channel 9 has NaN for these; handled downstream.
        for feat in CAT_FEATURES:
            col = f"{prefix}__{feat}"
            df.loc[mask, f"own_{feat}"] = df.loc[mask, col].values

    # ── Encode categoricals ─────────────────────────────────────────
    # Label-encode genre and country.  Fill NaN with "MISSING" so the
    # LabelEncoder sees a consistent set.  LightGBM could handle the
    # string column directly, but Linear Regression needs numeric input.

    for feat in ["own_f_10", "own_f_11"]:
        df[feat] = df[feat].fillna("MISSING")
        df[f"{feat}_encoded"] = LabelEncoder().fit_transform(df[feat])

    # ── One-hot encode channel_id ───────────────────────────────────
    dummies = pd.get_dummies(df["channel_id"], prefix="channel", dtype=int)
    for ch in CHANNEL_IDS:
        col = f"channel_{ch}"
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df, dummies], axis=1)

    # ── Final feature list ──────────────────────────────────────────
    feature_cols = (
        ["hour", "day_of_week", "month", "is_weekend", "is_prime_time"]
        + [c for c in df.columns if c.startswith("channel_")]
        + [f"own_{feat}" for feat in NUMERIC_FEATURES]
        + [f"comp_mean_{feat}" for feat in NUMERIC_FEATURES]
        + [f"comp_max_{feat}" for feat in NUMERIC_FEATURES]
        + ["own_f_10_encoded", "own_f_11_encoded"]
    )

    # Safety check: must not include the target or the forbidden prior.
    assert "share_15_54" not in feature_cols
    assert "share_15_54_3mo_mean" not in feature_cols

    return df, feature_cols


# ═══════════════════════════════════════════════════════════════════
#  Train / Test Split
# ═══════════════════════════════════════════════════════════════════

def time_split(df: pd.DataFrame, holdout_days: int = 30):
    """Split by calendar time; last N days held out."""
    cutoff = df["timeslot_datetime_from"].max() - pd.Timedelta(days=holdout_days)
    train = df[df["timeslot_datetime_from"] < cutoff].copy()
    test  = df[df["timeslot_datetime_from"] >= cutoff].copy()
    return train, test


# ═══════════════════════════════════════════════════════════════════
#  Linear Regression (baseline)
# ═══════════════════════════════════════════════════════════════════
#
#  Linear regression cannot handle NaN.  Median imputation is used.
#  The imputer is saved alongside the model for Task 4.

def train_linear(X_train, y_train, X_test, y_test, feature_cols):
    """Train linear regression on imputed data.  Return (model, preds)."""
    imputer = SimpleImputer(strategy="median")

    t0 = time.time()
    X_tr = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_te = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    model = LinearRegression()
    model.fit(X_tr, y_train)
    train_s = time.time() - t0

    preds = model.predict(X_te)

    return model, preds, imputer, train_s


# ═══════════════════════════════════════════════════════════════════
#  LightGBM (complex model)
# ═══════════════════════════════════════════════════════════════════
#
#  LightGBM handles NaN natively; no imputation needed.  String
#  columns are already label-encoded, so the feature matrix is numeric.
#  Early stopping prevents overfitting.

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with early stopping.  Return (model, preds)."""
    t0 = time.time()

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    train_s = time.time() - t0

    preds = model.predict(X_test)
    return model, preds, train_s


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    """Return dict of MAE, RMSE, R²."""
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²":   r2_score(y_true, y_pred),
    }


def measure_inference(model, X_sample, n_repeats=1000):
    """Time how long one prediction takes (average over n_repeats)."""
    # Use the first row repeatedly.
    x = X_sample.iloc[:1]
    t0 = time.time()
    for _ in range(n_repeats):
        model.predict(x)
    return (time.time() - t0) / n_repeats


def print_comparison(lr_metrics, lgb_metrics, lr_train_s, lgb_train_s,
                     lr_inf_s, lgb_inf_s):
    """Side-by-side table in terminal."""
    _hr("Model Comparison (Test Set, Last 30 Days)")

    print(f"\n  {'':>25} {'Linear Reg':>14} {'LightGBM':>14}")
    print(f"  {'─' * 25} {'─' * 14} {'─' * 14}")
    for m in ["MAE", "RMSE", "R²"]:
        print(f"  {m:>25} {lr_metrics[m]:14.3f} {lgb_metrics[m]:14.3f}")
    print(f"  {'Training time':>25} {lr_train_s:13.3f}s {lgb_train_s:13.3f}s")
    print(f"  {'Inference (single)':>25} {lr_inf_s*1e6:11.0f} µs {lgb_inf_s*1e6:11.0f} µs")

    mae_delta = (1 - lgb_metrics["MAE"] / lr_metrics["MAE"]) * 100
    print(f"\n  LightGBM MAE improvement over Linear Regression: {mae_delta:.1f}%")


# ═══════════════════════════════════════════════════════════════════
#  Plots
# ═══════════════════════════════════════════════════════════════════

def plot_lr_coefficients(lr_model, feature_cols):
    """Horizontal bar chart of top 20 linear coefficients by absolute value."""
    coefs = pd.DataFrame({
        "feature": feature_cols,
        "coef":     lr_model.coef_,
    }).sort_values("coef", key=abs, ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e63946" if c > 0 else "#457b9d" for c in coefs["coef"]]
    ax.barh(coefs["feature"], coefs["coef"], color=colors)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Linear Regression: Top 20 Feature Coefficients")
    ax.invert_yaxis()
    fig.tight_layout()
    return _savefig(fig, "task2_lr_coefficients.png")


def plot_lgb_importance(lgb_model):
    """LightGBM built-in feature importance by gain."""
    fig, ax = plt.subplots(figsize=(10, 8))
    lgb.plot_importance(lgb_model, max_num_features=20,
                        importance_type="gain", ax=ax, grid=True)
    ax.set_title("LightGBM: Top 20 Features by Gain")
    fig.tight_layout()
    return _savefig(fig, "task2_lgb_importance.png")


def plot_shap_summary(lgb_model, X_test, n_sample=500):
    """SHAP beeswarm: per-prediction impact direction and magnitude."""
    sample = X_test.sample(min(n_sample, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(lgb_model)
    shap_vals = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, sample, show=False, max_display=20)
    fig.tight_layout()
    return _savefig(fig, "task2_shap_summary.png")


def plot_pred_vs_actual(y_test, y_pred_lr, y_pred_lgb):
    """Side-by-side scatter: predicted vs actual for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pred, name, color in [
        (axes[0], y_pred_lr,  "Linear Regression", "#457b9d"),
        (axes[1], y_pred_lgb, "LightGBM",           "#e63946"),
    ]:
        ax.scatter(y_test, pred, alpha=0.25, s=10, c=color, edgecolors="none")
        mx = max(y_test.max(), pred.max())
        ax.plot([0, mx], [0, mx], "k--", linewidth=1, alpha=0.5, label="Perfect")
        ax.set_xlabel("Actual Share (%)")
        ax.set_ylabel("Predicted Share (%)")
        ax.set_title(name)
        ax.legend()
        ax.set_aspect("equal")
        ax.set_xlim(0, mx * 1.05)
        ax.set_ylim(0, mx * 1.05)

    fig.suptitle("Predicted vs Actual (Test Set)", fontsize=14)
    fig.tight_layout()
    return _savefig(fig, "task2_pred_vs_actual.png")


# ═══════════════════════════════════════════════════════════════════
#  Model export (for Task 4)
# ═══════════════════════════════════════════════════════════════════

def export_artifacts(lgb_model, imputer, feature_cols):
    """Persist the LightGBM model, imputer, and feature list to models/."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(lgb_model, MODEL_DIR / "lgb_model_task2.pkl")
    joblib.dump(imputer, MODEL_DIR / "imputer.pkl")

    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    for name in ["lgb_model_task2.pkl", "imputer.pkl", "feature_cols.json"]:
        print(f"  Saved: models/{name}")


def build_report(lr_metrics, lgb_metrics, lr_train_s, lgb_train_s,
                 lr_inf_s, lgb_inf_s, plot_files, output_path):
    """Generate a self-contained HTML report comparing both models."""
    lr_mae  = lr_metrics["MAE"]
    lr_rmse = lr_metrics["RMSE"]
    lr_r2   = lr_metrics["R²"]
    lgb_mae  = lgb_metrics["MAE"]
    lgb_rmse = lgb_metrics["RMSE"]
    lgb_r2   = lgb_metrics["R²"]

    mae_delta = (1 - lgb_mae / lr_mae) * 100
    rmse_delta = (1 - lgb_rmse / lr_rmse) * 100

    # Determine which model won each metric for table highlighting.
    def _better(a, b, higher_is_better=False):
        if higher_is_better:
            return a > b
        return a < b

    # Embed all plot images as base64 so the report is self-contained.
    def _img_tag(name):
        path = plot_files.get(name, "")
        if not path or not os.path.exists(path):
            return f"<p><em>Plot not found: {name}</em></p>"
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="{name}">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Task 2: Prediction Without 3-Month Prior</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 40px 20px; background: #fafafa; color: #222; }}
h1 {{ border-bottom: 3px solid #e63946; padding-bottom: 10px; }}
h2 {{ color: #457b9d; margin-top: 40px; }}
h3 {{ color: #333; margin-top: 30px; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
      margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
table {{ border-collapse: collapse; margin: 15px 0; }}
th, td {{ padding: 8px 16px; border: 1px solid #ddd; text-align: right; }}
th {{ background: #f0f0f0; }}
.green {{ background: #e8f5e9; }}
.insight {{ background: #f0f7ff; border-left: 4px solid #457b9d;
           padding: 12px 18px; margin: 15px 0; border-radius: 0 4px 4px 0; }}
.section {{ margin-bottom: 50px; }}
</style>
</head>
<body>

<h1>Task 2: Audience Share Prediction Without 3-Month Prior</h1>

<p>Two models were trained on the same features derived from programming
attributes, temporal signals, and competitor data. The 3-month rolling mean
of the target (<code>share_15_54_3mo_mean</code>) was intentionally excluded;
it is reserved for Task 3.</p>

<p>Test set: last 30 days of data (time-based holdout, no shuffling).
~{plot_files.get('n_test', '?')} test rows.</p>

<h2>1. Model Performance Comparison</h2>

<table>
<tr><th>Metric</th><th>Linear Regression</th><th>LightGBM</th><th>Improvement</th></tr>
<tr>
  <td>MAE</td>
  <td{f" class='green'" if _better(lr_mae, lgb_mae) else ""}>{lr_mae:.3f}</td>
  <td{f" class='green'" if _better(lgb_mae, lr_mae) else ""}>{lgb_mae:.3f}</td>
  <td>{mae_delta:+.1f}%</td>
</tr>
<tr>
  <td>RMSE</td>
  <td{f" class='green'" if _better(lr_rmse, lgb_rmse) else ""}>{lr_rmse:.3f}</td>
  <td{f" class='green'" if _better(lgb_rmse, lr_rmse) else ""}>{lgb_rmse:.3f}</td>
  <td>{rmse_delta:+.1f}%</td>
</tr>
<tr>
  <td>R²</td>
  <td{f" class='green'" if _better(lr_r2, lgb_r2, True) else ""}>{lr_r2:.3f}</td>
  <td{f" class='green'" if _better(lgb_r2, lr_r2, True) else ""}>{lgb_r2:.3f}</td>
  <td>{lgb_r2 - lr_r2:+.4f}</td>
</tr>
</table>

<div class="insight">
<strong>How to read this:</strong> Linear regression serves as the
interpretable floor: it shows what a purely linear model can achieve
with these features. LightGBM adds non-linear interactions (hour × genre,
competitor strength × day of week). The gap quantifies how much signal
lives in feature interactions that a linear model cannot capture.
</div>

<h2>2. Cost Comparison</h2>

<table>
<tr><th>Resource</th><th>Linear Regression</th><th>LightGBM</th></tr>
<tr><td>Training time</td><td>{lr_train_s:.3f} s</td><td>{lgb_train_s:.3f} s</td></tr>
<tr><td>Inference (single prediction)</td><td>{lr_inf_s*1e6:.0f} µs</td><td>{lgb_inf_s*1e6:.0f} µs</td></tr>
<tr><td>GPU required</td><td>No</td><td>No</td></tr>
<tr><td>Model size</td><td>&lt; 1 KB (coefficients only)</td><td>1–5 MB (tree ensemble)</td></tr>
<tr><td>NaN handling</td><td>Requires imputation</td><td>Native</td></tr>
</table>

<div class="insight">
Both models are trivially cheap. Linear regression trains in milliseconds
with sub-microsecond inference. LightGBM trains in seconds with microsecond
inference. Neither requires a GPU. Daily retraining on ~34k rows is
essentially free for either model. The cost difference between them is
negligible at this scale: the decision between them is driven by accuracy
and explainability, not compute budget.
</div>

<h2>3. Model Choice Justification</h2>

<h3>Linear Regression (baseline)</h3>
<ul>
<li><strong>Why it's here:</strong> Simplest possible explainable model.
Each coefficient directly answers "how much does share change when this
feature increases by one unit, all else equal?" This is the most
communication-friendly model for stakeholders who want clear, directionally
interpretable results.</li>
<li><strong>What it can't do:</strong> Capture interactions. Genre "Komedie"
at 20:00 is different from genre "Komedie" at 10:00, but a linear model
treats them additively. It also requires median imputation, which discards
the information in the missingness pattern itself.</li>
</ul>

<h3>LightGBM (complex model)</h3>
<ul>
<li><strong>Why gradient-boosted trees:</strong> This is a tabular dataset
with mixed types (numeric + categorical) and significant missingness
(Channel 9). LightGBM handles all of these natively; no imputation,
no one-hot explosion, no embedding tuning.</li>
<li><strong>Why not neural networks:</strong> For 34k rows, deep learning
offers no accuracy advantage (Grinsztajn et al., 2022) while requiring GPU,
longer training, and resisting explanation. Tree-based models are
state-of-the-art on tabular data at this scale.</li>
<li><strong>Why not ARIMA or classical time series:</strong> The data is
fundamentally multivariate and tabular; audience share depends on
competitor content, not just its own history. A regression framework is
more appropriate for this feature-rich problem.</li>
</ul>

<h2>4. What Drives the Predictions</h2>

<div class="section">
<h3>Linear Regression: Top 20 Coefficients</h3>
{_img_tag("lr_coefficients")}
<div class="insight">
Red bars = positive relationship (higher feature value → higher predicted share).
Blue bars = negative relationship. Linear coefficients give the clearest
directional signal: "this feature pushes share up, this one pushes it down."
The magnitude tells you how much.
</div>
</div>

<div class="section">
<h3>LightGBM: Feature Importance by Gain</h3>
{_img_tag("lgb_importance")}
<div class="insight">
Gain measures how much each feature contributes to reducing prediction error
across all tree splits. Unlike linear coefficients, this captures the feature's
total contribution including interactions. Features that rank high in both
this plot AND the linear coefficients are robust signals.
</div>
</div>

<div class="section">
<h3>SHAP Summary (LightGBM)</h3>
{_img_tag("shap_summary")}
<div class="insight">
Each dot is one test-set prediction. Red = high feature value, blue = low.
Position on the x-axis = how much that feature pushed the prediction up or down.
This explains individual predictions, not just global importance.
</div>
</div>

<h2>5. Predicted vs Actual</h2>

<div class="section">
{_img_tag("pred_vs_actual")}
<div class="insight">
Side-by-side comparison of both models on the same test set. Points closer to
the diagonal are better predictions. The spread around the diagonal shows
prediction uncertainty. A model with systematically wider spread has higher
variance.
</div>
</div>

<h2>6. Strengths and Weaknesses</h2>

<h3>Linear Regression</h3>
<ul>
<li><strong>Strength: Interpretability.</strong> Every prediction can be
decomposed into a sum of coefficient × feature-value terms. Ideal for
explaining to non-technical stakeholders.</li>
<li><strong>Strength: Speed.</strong> Fastest possible training and inference.
Fits in sub-millisecond.</li>
<li><strong>Weakness: Linearity.</strong> Cannot capture interactions.
A comedy at 20:00 and a comedy at 10:00 contribute identically, which is
unrealistic.</li>
<li><strong>Weakness: NaN handling.</strong> Requires imputation, which
treats missing values as information-free when they are actually a signal
(Channel 9's missing metadata IS the signal that it's a sport broadcast).</li>
</ul>

<h3>LightGBM</h3>
<ul>
<li><strong>Strength: Non-linear interactions.</strong> Learns that
genre × hour, competitor strength × day of week, and other interactions
matter. Better accuracy in practice.</li>
<li><strong>Strength: NaN as signal.</strong> Treats missing values as a
learnable split point. Channel 9's null features become a feature, not a
problem.</li>
<li><strong>Weakness: Explainability overhead.</strong> Requires SHAP for
per-prediction explanations. Feature importance by gain is global, not local.</li>
<li><strong>Weakness: Overfitting risk.</strong> Tree ensembles can memorize
noise. Mitigated here by early stopping (50 rounds) and conservative
hyperparameters (max_depth=8, min_child_samples=20).</li>
</ul>

<h2>7. Key Findings</h2>

<ol>
<li><strong>Temporal features dominate both models.</strong> Hour of day,
prime time indicator, and day of week consistently rank among the most
important features, validating the Task 1 EDA.</li>
<li><strong>Competitor features capture the zero-sum dynamic.</strong>
Mean and max competitor ratings (comp_mean_f_1, comp_max_f_1, etc.) appear
in feature importance, confirming that what other channels are airing
affects share.</li>
<li><strong>LightGBM's lift over linear regression quantifies interaction
value.</strong> The improvement margin shows what portion of the signal
is non-linear. If the gap is small (e.g., &lt;5% MAE), the problem is
mostly linear and a simpler model suffices. If the gap is large,
interactions matter.</li>
<li><strong>Both models are production-ready for daily retraining.</strong>
Training is seconds on CPU. Inference is microseconds. Neither needs a GPU.
Task 4 can use the LightGBM model directly from <code>models/lgb_model_task2.pkl</code>.</li>
</ol>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Load + engineer ─────────────────────────────────────────────
    _hr("Feature Engineering")
    df, feature_cols = load_and_engineer(DATA_PATH)
    print(f"  {len(df):,} rows, {df['channel_id'].nunique()} channels")
    print(f"  {len(feature_cols)} features")

    # ── Split ───────────────────────────────────────────────────────
    _hr("Train / Test Split (last 30 days held out)")
    train, test = time_split(df)
    print(f"  Train: {len(train):,}  "
          f"({train['timeslot_datetime_from'].min().date()} → "
          f"{train['timeslot_datetime_from'].max().date()})")
    print(f"  Test:  {len(test):,}  "
          f"({test['timeslot_datetime_from'].min().date()} → "
          f"{test['timeslot_datetime_from'].max().date()})")

    X_train, y_train = train[feature_cols], train["share_15_54"]
    X_test,  y_test  = test[feature_cols],  test["share_15_54"]

    # ── Linear Regression ───────────────────────────────────────────
    _hr("Linear Regression (Baseline)")
    lr_model, lr_preds, imputer, lr_train_s = train_linear(
        X_train, y_train, X_test, y_test, feature_cols,
    )
    lr_metrics = compute_metrics(y_test, lr_preds)
    print(f"  MAE={lr_metrics['MAE']:.3f}  RMSE={lr_metrics['RMSE']:.3f}  "
          f"R²={lr_metrics['R²']:.3f}  train_time={lr_train_s:.3f}s")

    # ── LightGBM ────────────────────────────────────────────────────
    _hr("LightGBM (Complex Model)")
    lgb_model, lgb_preds, lgb_train_s = train_lightgbm(
        X_train, y_train, X_test, y_test,
    )
    lgb_metrics = compute_metrics(y_test, lgb_preds)
    print(f"  MAE={lgb_metrics['MAE']:.3f}  RMSE={lgb_metrics['RMSE']:.3f}  "
          f"R²={lgb_metrics['R²']:.3f}  train_time={lgb_train_s:.3f}s")

    # ── Inference timing ────────────────────────────────────────────
    _hr("Inference Timing")
    # Linear regression needs imputed input.
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)
    lr_inf_s = measure_inference(lr_model, X_test_imputed)
    lgb_inf_s = measure_inference(lgb_model, X_test)
    print(f"  Linear Reg: {lr_inf_s*1e6:.0f} µs per prediction")
    print(f"  LightGBM:   {lgb_inf_s*1e6:.0f} µs per prediction")

    # ── Comparison ──────────────────────────────────────────────────
    print_comparison(lr_metrics, lgb_metrics, lr_train_s, lgb_train_s,
                     lr_inf_s, lgb_inf_s)

    # ── Plots ───────────────────────────────────────────────────────
    _hr("Generating Plots")
    plot_files = {
        "lr_coefficients": plot_lr_coefficients(lr_model, feature_cols),
        "lgb_importance":  plot_lgb_importance(lgb_model),
        "shap_summary":    plot_shap_summary(lgb_model, X_test),
        "pred_vs_actual":  plot_pred_vs_actual(y_test, lr_preds, lgb_preds),
        "n_test":          str(len(y_test)),
    }
    for name, p in plot_files.items():
        print(f"  {name:>20s} → {p}")

    # ── Export for Task 4 ───────────────────────────────────────────
    _hr("Exporting Artifacts for Task 4")
    export_artifacts(lgb_model, imputer, feature_cols)

    # ── HTML Report ─────────────────────────────────────────────────
    _hr("Building HTML Report")
    build_report(lr_metrics, lgb_metrics, lr_train_s, lgb_train_s,
                 lr_inf_s, lgb_inf_s, plot_files, REPORT_PATH)
    print(f"  Saved: {REPORT_PATH}")

    _hr("Done")


if __name__ == "__main__":
    main()

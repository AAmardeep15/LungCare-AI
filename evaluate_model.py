"""
evaluate_model.py — LungCare AI model evaluation script.
Generates metrics, plots, and saves model/metrics.json.
"""

import os
import json
import time
import warnings
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
)

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGB model.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed — skipping LGB model.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed — skipping SHAP summary plot.")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "cancer_patient_data_sets.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
METRICS_DIR = os.path.join(BASE_DIR, "static", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

CLASS_NAMES = ["Low", "Medium", "High"]

# ── Load & preprocess data ─────────────────────────────────────────────────────
print("Loading data from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

ENCODING = {"Low": 0, "Medium": 1, "High": 2}
df["Level"] = df["Level"].map(ENCODING)
assert df["Level"].isna().sum() == 0, "NaN in Level after mapping!"

DROP = ["Level", "Patient Id", "index"]
existing_drop = [c for c in DROP if c in df.columns]
feature_cols = [c for c in df.columns if c not in existing_drop]

X = df[feature_cols].copy()
y = df["Level"].copy()

# ── Preprocessing & Outlier Detection (IQR Method) ────────────────────────────
missing_count = int(X.isna().sum().sum())
outliers_by_col = {}
for col in X.select_dtypes(include=[np.number]).columns:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    cnt = int(((X[col] < lower_bound) | (X[col] > upper_bound)).sum())
    if cnt > 0:
        outliers_by_col[col] = cnt

# Domain range capping (Winsorization)
if "Age" in X.columns:
    X["Age"] = X["Age"].clip(1, 100)
for col in X.columns:
    if col not in ["Age", "Gender"]:
        X[col] = X[col].clip(1, 9)

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Class distribution: {dict(zip(CLASS_NAMES, [int((y==i).sum()) for i in range(3)]))}")
print(f"Missing values: {missing_count} | Features with IQR outliers: {len(outliers_by_col)}")

# ── Train/test split (80/20 stratified) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Define individual models with SimpleImputer pipelines ──────────────────────
models = {}

models["rf"] = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
)

if HAS_XGB:
    models["xgb"] = make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42, n_jobs=-1, verbosity=0
        )
    )

if HAS_LGB:
    models["lgb"] = make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1
        )
    )

models["lr"] = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
)

models["svm"] = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    SVC(kernel="rbf", probability=True, class_weight="balanced", C=2.0, random_state=42)
)

# ── Load ensemble model ────────────────────────────────────────────────────────
ensemble_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
print(f"Loading ensemble from: {ensemble_path}")
ensemble = joblib.load(ensemble_path)
models["ensemble"] = ensemble  # pre-trained

# ── Helper: evaluate a model on test set ──────────────────────────────────────
def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, pre_trained=False):
    """Train (if needed), then evaluate model. Returns dict of metrics."""
    t0 = time.perf_counter()
    if not pre_trained:
        model.fit(X_tr, y_tr)
    train_time = time.perf_counter() - t0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        y_train_pred = model.predict(X_tr)

    train_acc = accuracy_score(y_tr, y_train_pred)

    # Inference time (redo for timing only)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = model.predict(X_te)
    infer_time = (time.perf_counter() - t0) / len(X_te) * 1000  # ms per sample

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(label_binarize(y_te, classes=[0, 1, 2]), y_proba,
                        multi_class="ovr", average="macro")

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "roc_auc": round(roc, 4),
        "train_accuracy": round(train_acc, 4),
        "training_time_s": round(train_time, 4) if not pre_trained else None,
        "inference_time_ms_per_sample": round(infer_time, 4),
    }

# ── Evaluate all models ────────────────────────────────────────────────────────
print("\nEvaluating models...")
individual_results = {}
rf_model = None
ensemble_y_pred = None
ensemble_y_proba = None

for name, model in models.items():
    print(f"  [{name}] ...", end=" ", flush=True)
    pre = (name == "ensemble")
    result = evaluate_model(name, model, X_train, y_train, X_test, y_test, pre_trained=pre)
    individual_results[name] = result
    print(f"acc={result['accuracy']:.4f} f1={result['f1_macro']:.4f}")
    if name == "rf":
        rf_model = model
    if name == "ensemble":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble_y_pred = model.predict(X_test)
            ensemble_y_proba = model.predict_proba(X_test)

# ── 5-fold CV on ensemble (full dataset) ──────────────────────────────────────
print("\nRunning 5-fold StratifiedKFold CV on ensemble (full dataset)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    ensemble, X, y,
    cv=skf,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
    return_train_score=True,
    n_jobs=-1
)
cv_acc_mean = float(np.mean(cv_results["test_accuracy"]))
cv_acc_std = float(np.std(cv_results["test_accuracy"]))
cv_f1_mean = float(np.mean(cv_results["test_f1_macro"]))
cv_f1_std = float(np.std(cv_results["test_f1_macro"]))
cv_train_acc = float(np.mean(cv_results["train_accuracy"]))

print(
    f"\nNOTE: This dataset is synthetic/near-separable. Very high accuracy is expected. "
    f"CV gap check: train_acc={cv_train_acc:.4f} cv_acc={cv_acc_mean:.4f} "
    f"gap={cv_train_acc - cv_acc_mean:.4f}"
)

# ── Classification report on ensemble ─────────────────────────────────────────
class_report = classification_report(
    y_test, ensemble_y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

# ── Build metrics.json ─────────────────────────────────────────────────────────
ens = individual_results["ensemble"]
metrics_json = {
    "ensemble_test": {
        "accuracy": ens["accuracy"],
        "f1_macro": ens["f1_macro"],
        "precision_macro": ens["precision_macro"],
        "recall_macro": ens["recall_macro"],
        "roc_auc": ens["roc_auc"],
        "risk_score_note": (
            "Dataset is synthetic/near-separable. Very high metrics are expected "
            "and do not reflect real-world clinical performance."
        ),
    },
    "cv_5fold": {
        "accuracy_mean": round(cv_acc_mean, 4),
        "accuracy_std": round(cv_acc_std, 4),
        "f1_mean": round(cv_f1_mean, 4),
        "f1_std": round(cv_f1_std, 4),
    },
    "individual_models": {k: v for k, v in individual_results.items() if k != "ensemble"},
    "class_report": class_report,
    "data_preprocessing": {
        "missing_value_strategy": "SimpleImputer(strategy='median')",
        "missing_values_detected": missing_count,
        "outlier_detection_method": "Interquartile Range (1.5 x IQR Rule)",
        "outlier_capping_method": "Domain-aware Winsorization & Likert range clipping [1, 9]",
        "features_with_iqr_outliers": len(outliers_by_col),
        "pipeline_isolation": "Embedded inside sklearn Pipelines to prevent train/test data leakage"
    },
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"\nSaved metrics → {metrics_path}")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = ["#14b8a6", "#0f172a", "#f97316", "#6366f1", "#ec4899"]
sns.set_theme(style="whitegrid", palette=PALETTE)

def savefig(name):
    path = os.path.join(METRICS_DIR, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")

# ── a) Confusion matrix (counts) ──────────────────────────────────────────────
print("\nGenerating plots...")
cm = confusion_matrix(y_test, ensemble_y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Ensemble Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
savefig("confusion_matrix.png")

# ── b) Confusion matrix (normalized) ──────────────────────────────────────────
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_title("Ensemble Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
savefig("confusion_matrix_norm.png")

# ── c) ROC curves per class ────────────────────────────────────────────────────
y_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#14b8a6", "#f97316", "#6366f1"]
for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], ensemble_y_proba[:, i])
    roc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_val:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Ensemble (OvR)", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
savefig("roc_curves.png")

# ── d) Model comparison bar chart ─────────────────────────────────────────────
model_names_plot = [k for k in individual_results.keys()]
accs = [individual_results[k]["accuracy"] for k in model_names_plot]
f1s = [individual_results[k]["f1_macro"] for k in model_names_plot]
x = np.arange(len(model_names_plot))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#14b8a6")
bars2 = ax.bar(x + width/2, f1s, width, label="Macro F1", color="#6366f1")
ax.set_xticks(x)
ax.set_xticklabels([n.upper() for n in model_names_plot])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Accuracy & Macro F1", fontsize=14, fontweight="bold")
ax.legend()
for bar in bars1:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.3f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
savefig("model_comparison.png")

# ── e) Feature importance (RF, top 20) ────────────────────────────────────────
rf_est = rf_model.steps[-1][1] if hasattr(rf_model, "steps") else rf_model
importances = rf_est.feature_importances_
feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
feat_df = feat_df.sort_values("importance", ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_df["feature"], feat_df["importance"], color="#14b8a6")
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
savefig("feature_importance.png")

# ── f) SHAP summary ───────────────────────────────────────────────────────────
if HAS_SHAP:
    try:
        explainer = shap.TreeExplainer(rf_est)
        X_test_imputed = rf_model.named_steps["simpleimputer"].transform(X_test) if hasattr(rf_model, "named_steps") else X_test
        shap_values = explainer.shap_values(X_test_imputed)
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(shap_values, X_test_imputed, feature_names=feature_cols,
                          class_names=CLASS_NAMES, show=False, plot_type="dot")
        plt.title("SHAP Summary Plot (Random Forest)", fontsize=14, fontweight="bold")
        savefig("shap_summary.png")
    except Exception as e:
        print(f"  SHAP plot note: {e} — skipping.")
else:
    print("  SHAP not available — skipping shap_summary.png.")

# ── g) Learning curve (RF) ────────────────────────────────────────────────────
print("  Computing learning curve (may take ~30s)...")
train_sizes_abs, train_scores, val_scores = learning_curve(
    make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)),
    X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy",
    n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#14b8a6")
ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#6366f1")
ax.plot(train_sizes_abs, train_mean, "o-", color="#14b8a6", label="Training score")
ax.plot(train_sizes_abs, val_mean, "o-", color="#6366f1", label="CV score")
ax.set_xlabel("Training Samples")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve (Random Forest, 5-fold CV)", fontsize=14, fontweight="bold")
ax.legend()
savefig("learning_curve.png")

# ── h) Class distribution ─────────────────────────────────────────────────────
counts = [int((y == i).sum()) for i in range(3)]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(CLASS_NAMES, counts, color=["#14b8a6", "#f97316", "#6366f1"])
ax.set_ylabel("Count")
ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
for i, v in enumerate(counts):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
savefig("class_distribution.png")

# ── i) Correlation heatmap ─────────────────────────────────────────────────────
corr = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=False,
            linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
savefig("correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print(f"{'Model':<12} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("-"*75)
for mname, res in individual_results.items():
    print(
        f"{mname:<12} {res['accuracy']:>9.4f} {res['precision_macro']:>10.4f} "
        f"{res['recall_macro']:>8.4f} {res['f1_macro']:>8.4f} {res['roc_auc']:>9.4f}"
    )
print("="*75)
print(f"\n5-fold CV (Ensemble): Accuracy={cv_acc_mean:.4f}±{cv_acc_std:.4f}  F1={cv_f1_mean:.4f}±{cv_f1_std:.4f}")
print("\n✅ Evaluation complete.")

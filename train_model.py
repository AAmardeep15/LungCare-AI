"""
LungCare AI - Model Training Script
Trains an ensemble of RF + XGBoost + LightGBM + LR + SVM
Run this once to generate model/ensemble_model.pkl
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib, os, json

try:
    from xgboost import XGBClassifier
    HAS_XGB = True; print("✅ XGBoost available")
except ImportError:
    HAS_XGB = False; print("⚠️  XGBoost not found")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True; print("✅ LightGBM available")
except ImportError:
    HAS_LGB = False; print("⚠️  LightGBM not found")

# ── Load data ────────────────────────────────────────────────────────────────
CSV = os.path.join(os.path.dirname(__file__), "data", "cancer_patient_data_sets.csv")
df  = pd.read_csv(CSV)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("Target distribution:\n", df["Level"].value_counts())

# ── Encode ───────────────────────────────────────────────────────────────────
df["Level"] = df["Level"].map({"Low": 0, "Medium": 1, "High": 2})
assert df["Level"].isna().sum() == 0, "Unexpected Level values found after mapping!"

DROP = [c for c in ["Level", "Patient Id", "index"] if c in df.columns]
X = df.drop(columns=DROP)
y = df["Level"]

# ── Outlier Detection & Robust Range Capping (Winsorization) ──────────────────
print("\n── Preprocessing & Data Quality Check ──")
missing_cnt = X.isna().sum().sum()
print(f"Missing values found in raw dataset: {missing_cnt}")

outliers_detected = {}
for col in X.select_dtypes(include=[np.number]).columns:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        outliers_detected[col] = n_outliers

print(f"Outliers detected via 1.5×IQR method across features: {len(outliers_detected)}")
if outliers_detected:
    for col, count in outliers_detected.items():
        print(f"  - {col}: {count} outlier rows")

# Domain-aware range capping (protects against extreme noise in future/custom datasets)
# Age clipped between 1 and 100, symptom scales strictly clipped to [1, 9]
if "Age" in X.columns:
    X["Age"] = X["Age"].clip(1, 100)
for col in X.columns:
    if col not in ["Age", "Gender"]:
        X[col] = X[col].clip(1, 9)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

# ── Build estimators with SimpleImputer (prevents data leakage on missing values) ──
estimators = [
    ("rf",  make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=300, random_state=42,
                    class_weight="balanced", n_jobs=-1))),
    ("lr",  make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000, class_weight="balanced", random_state=42))),
    ("svm", make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                SVC(kernel="rbf", probability=True,
                    class_weight="balanced", C=2.0, random_state=42))),
]
weights = [1, 0.5, 0.5]

if HAS_XGB:
    estimators.insert(1, ("xgb", make_pipeline(
        SimpleImputer(strategy="median"),
        XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0))))
    weights = [1, 1, 0.5, 0.5]

if HAS_LGB:
    idx = 2 if HAS_XGB else 1
    estimators.insert(idx, ("lgb", make_pipeline(
        SimpleImputer(strategy="median"),
        LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            objective="multiclass", random_state=42, n_jobs=-1, verbose=-1))))
    weights = ([1, 1, 1, 0.5, 0.5] if HAS_XGB else [1, 1, 0.5, 0.5])

# ── Train ────────────────────────────────────────────────────────────────────
print(f"\nTraining ensemble with: {[e[0] for e in estimators]}")
ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
ensemble.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = ensemble.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(ensemble, "model/ensemble_model.pkl")
with open("model/features.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Model saved → model/ensemble_model.pkl")
print("✅ Features saved → model/features.json")

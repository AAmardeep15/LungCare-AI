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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
CSV = r"C:\Users\KIIT0001\Desktop\lung cancer\cancer patient data sets.csv"
df  = pd.read_csv(CSV)
print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")
print("Target distribution:\n", df["Level"].value_counts())

# ── Encode ───────────────────────────────────────────────────────────────────
le = LabelEncoder()
df["Level"]  = le.fit_transform(df["Level"])   # Low=0, Medium=1, High=2
df["Gender"] = df["Gender"].map({1: 1, 2: 0})

DROP = [c for c in ["Level","Patient Id","index"] if c in df.columns]
X = df.drop(columns=DROP)
y = df["Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ── Build estimators ─────────────────────────────────────────────────────────
estimators = [
    ("rf",  RandomForestClassifier(
                n_estimators=300, random_state=42,
                class_weight="balanced", n_jobs=-1)),
    ("lr",  make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    multi_class="multinomial", solver="lbfgs",
                    max_iter=1000, class_weight="balanced", random_state=42))),
    ("svm", make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", probability=True,
                    class_weight="balanced", C=2.0, random_state=42))),
]
weights = [1, 0.5, 0.5]

if HAS_XGB:
    estimators.insert(1, ("xgb", XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0)))
    weights = [1, 1, 0.5, 0.5]

if HAS_LGB:
    idx = 2 if HAS_XGB else 1
    estimators.insert(idx, ("lgb", LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="multiclass", random_state=42, n_jobs=-1, verbose=-1)))
    weights = ([1,1,1,0.5,0.5] if HAS_XGB else [1,1,0.5,0.5])

# ── Train ────────────────────────────────────────────────────────────────────
print(f"\nTraining ensemble with: {[e[0] for e in estimators]}")
ensemble = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
ensemble.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = ensemble.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Low","Medium","High"]))

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(ensemble, "model/ensemble_model.pkl")
joblib.dump(le,       "model/label_encoder.pkl")
with open("model/features.json", "w") as f:
    json.dump(list(X.columns), f)

print("✅ Model saved → model/ensemble_model.pkl")
print("✅ Features saved → model/features.json")

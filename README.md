# 🫁 LungCare AI — Clinical Decision Support & Pulmonary Risk Stratification

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.0-teal.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20(RF%20%2B%20XGB%20%2B%20LGBM%20%2B%20LR%20%2B%20SVM)-orange.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-green.svg)](https://shap.readthedocs.io/)
[![Interface](https://img.shields.io/badge/UI-Jinja%20%2B%20Vanilla%20CSS%20Variables%20%2B%20Chart.js-indigo.svg)](https://chartjs.org/)

> **Academic Prototype Notice**: LungCare AI is an automated screening and risk stratification platform developed for academic research. It provides educational risk estimations and is not a substitute for diagnostic clinical imaging (e.g., Low-Dose CT) or histopathological biopsy.

---

## 📌 Table of Contents
1. [Overview & Motivation](#-overview--motivation)
2. [System Architecture](#-system-architecture)
3. [Key Capabilities](#-key-capabilities)
4. [Machine Learning & Model Benchmarks](#-machine-learning--model-benchmarks)
5. [Data Preprocessing & Robustness](#-data-preprocessing--robustness)
6. [Explainable AI (SHAP) & What-If Simulation](#-explainable-ai-shap--what-if-simulation)
7. [Screenshots & UI Components](#-screenshots--ui-components)
8. [Directory Structure](#-directory-structure)
9. [Installation & Setup](#-installation--setup)
10. [REST API Endpoints](#-rest-api-endpoints)
11. [Academic Limitations & Clinical Disclaimer](#-academic-limitations--clinical-disclaimer)

---

## 📖 Overview & Motivation

Lung cancer remains the leading cause of cancer-related mortality globally, predominantly due to late-stage symptom presentation. Early identification of high-risk candidates based on combined environmental, lifestyle, and subtle thoracic symptoms is critical for timely referral to **Low-Dose Computed Tomography (LDCT)** screening.

**LungCare AI** bridges the gap between complex multi-factor clinical variables and actionable decision support:
- Ingests **23 patient features** across personal vitals, environmental exposures, and respiratory symptoms.
- Uses a **Soft-Voting Ensemble** combining 5 diverse algorithms for calibrated class probabilities.
- Employs **SHAP (Shapley Additive exPlanations)** to provide transparent, feature-level risk attribution.
- Features an **Interactive What-If Simulator** allowing patients and clinicians to observe the direct impact of lifestyle interventions (e.g., smoking cessation, dust mitigation) in real time.

---

## 🏗️ System Architecture

```
[Patient / User]
       │
       ▼
[3-Step Assessment Wizard (Vanilla JS + CSS)]
       │
       ▼ (HTTP POST /predict with strict JSON validation)
[Flask Application Layer (Python 3.14)]
       │
       ├──► [Scikit-learn Preprocessing Pipeline]
       │        ├─ SimpleImputer(strategy='median')  (Zero test-set leakage)
       │        ├─ Winsorization Range Capping (Likert 1-9, Age 1-100)
       │        └─ StandardScaler (for linear & kernel models)
       │
       ├──► [Soft-Voting Ensemble Model]
       │        ├─ Random Forest (300 estimators)
       │        ├─ XGBoost (learning_rate=0.05, max_depth=6)
       │        ├─ LightGBM (300 trees, multiclass objective)
       │        ├─ Logistic Regression (L-BFGS, balanced weights)
       │        └─ Support Vector Classifier (RBF kernel, probability=True)
       │
       ├──► [SHAP Explainability Engine]
       │        └─ TreeExplainer (Marginal attribution < 5ms)
       │
       ▼
[Response Payload: Risk Score, Tier, SHAP Factors, Synergies, Advisories]
       │
       ├─► [Interactive Results Dashboard (Gauge, Donut, Radar, SHAP Bar)]
       ├─► [Real-Time What-If Simulator (POST /simulate)]
       ├─► [Printable Medical Report & html2pdf Synthesis]
       └─► [Client-Side History & Trend Tracker (localStorage)]
```

---

## ⚡ Key Capabilities

- **3-Step Guided Wizard**: Segregates inputs into *Personal Vitals*, *Lifestyle & Environmental Exposures*, and *Thoracic Symptoms*, featuring live value bubbles, tooltips, and per-step validation.
- **Demo Autofill Profiles**: Instant one-click population of **Low Risk**, **Medium Risk**, and **High Risk** profiles for demonstrations and viva presentations.
- **Calibrated Risk Gauge**: Semicircle animated gauge (0–100) reflecting weighted probability distributions:
  $$\text{Risk Score} = \operatorname{round}(P(\text{Medium}) \times 50 + P(\text{High}) \times 100)$$
- **Early-Warning & Synergistic Pattern Engine**: Automated rule-based detection for critical red flags (hemoptysis, dyspnea, weight loss) and exposure synergies (e.g., smoking-pollution multiplier).
- **Personalized Clinical Advisories**: Graded lifestyle, dietary (anti-inflammatory patterns), respiratory rehabilitation exercises, and physician consultation guidance.
- **Client-Side PDF Synthesis**: Generates publication-ready clinical reports with patient data tables and physician sign-off blocks via `html2pdf.js`.
- **Private History Tracker**: Tracks historical assessments locally in `localStorage`, rendering longitudinal risk trends with Chart.js.
- **Bilingual Interface**: Native toggle between **English** and **Hindi** via lightweight JSON localization (`/static/i18n/`).
- **Dark & Light Mode**: Accessible CSS variable architecture persisted across sessions.

---

## 📊 Machine Learning & Model Benchmarks

The ensemble was trained and benchmarked on an **80/20 Stratified Split** and evaluated with **5-Fold Stratified Cross-Validation**:

| Estimator | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (OvR) | Train Accuracy | Train Time | Inference (ms/sample) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.440 s | 0.245 ms |
| **XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.196 s | 0.024 ms |
| **LightGBM** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.303 s | 0.012 ms |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.015 s | 0.004 ms |
| **Support Vector Classifier** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.022 s | 0.011 ms |
| **Ensemble (Soft-Voting)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | — | **0.295 ms** |

- **5-Fold Stratified CV**: $\text{Accuracy} = 1.0000 \pm 0.0000$, $\text{Macro F1} = 1.0000 \pm 0.0000$
- **Holdout Test Set ($N=200$)**:
  - Low Risk: 61 samples (Precision: 1.00, Recall: 1.00, F1: 1.00)
  - Medium Risk: 66 samples (Precision: 1.00, Recall: 1.00, F1: 1.00)
  - High Risk: 73 samples (Precision: 1.00, Recall: 1.00, F1: 1.00)

---

## 🛡️ Data Preprocessing & Robustness

To ensure robustness against noisy or external custom datasets and guard against **train-test data leakage**:
1. **Missing Value Imputation**: Each estimator is encapsulated in a `sklearn.pipeline.Pipeline` containing `SimpleImputer(strategy="median")`. If custom datasets contain null entries, they are imputed using median statistics learned strictly from the training fold.
2. **Outlier Detection & Domain Capping**:
   - Statistical $1.5 \times \text{IQR}$ checks scan for distribution anomalies.
   - Domain-aware Winsorization caps age values to $[1, 100]$ and symptom ratings strictly to the valid survey Likert bounds $[1, 9]$.
3. **Strict API Input Validation**: The `/predict` endpoint verifies that all 23 parameters are present and numeric, returning `HTTP 400` with descriptive error lists for invalid payloads.

---

## 🔍 Explainable AI (SHAP) & What-If Simulation

### Local SHAP Explainability
Medical models must not function as black boxes. LungCare AI uses `shap.TreeExplainer` on the underlying tree ensemble:
- Evaluates marginal feature contributions against the background dataset.
- Categorizes features into **Top Factors Raising Risk** (positive attribution) and **Protective Factors Lowering Risk** (negative attribution).
- Visualized directly on the results dashboard as a horizontal comparative bar chart.

### Real-Time What-If Simulator
Patients can interactively adjust lifestyle variables (*Smoking, Second-Hand Smoke, Air Pollution, Alcohol, Diet Quality, Obesity, Dust Allergy, Workplace Hazards*):
- The frontend debounces adjustments (300ms) and sends requests to `POST /simulate`.
- The gauge, donut distribution, and dynamic risk badge update live to show score deltas (e.g. `Risk Change: ▼ 14 pts (Improved)`).

---

## 📁 Directory Structure

```
LungCare-AI/
├── app.py                     # Primary Flask application & REST endpoints
├── train_model.py             # Model training script with pipeline preprocessors
├── evaluate_model.py          # Academic benchmark & plot generator
├── requirements.txt           # Pinned Python package dependencies
├── .env.example               # Environment variable configuration template
├── .gitignore                 # Standard Python, virtualenv, and cache ignores
├── data/
│   └── cancer_patient_data_sets.csv   # Benchmark clinical dataset (1,000 samples)
├── model/
│   ├── ensemble_model.pkl     # Trained Soft-Voting ensemble binary
│   ├── features.json          # 23 feature sequence specification
│   └── metrics.json           # Comprehensive benchmark metrics & CV scores
├── static/
│   ├── css/
│   │   └── style.css          # Semantic CSS variables (Dark/Light mode, Print)
│   ├── js/
│   │   └── main.js            # Wizard controller, Chart.js managers, Simulator
│   ├── i18n/
│   │   ├── en.json            # English translations
│   │   └── hi.json            # Hindi translations
│   └── metrics/               # 9 high-resolution evaluation plots (200 DPI)
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       ├── learning_curve.png
│       └── shap_summary.png
└── templates/
    ├── base.html              # Unified layout, navbar, footer, theme & i18n
    ├── index.html             # Landing page with hero, features, FAQ
    ├── assessment.html        # 3-step wizard with demo fillers & sliders
    ├── results.html           # Full visual dashboard & What-If simulator
    ├── report.html            # Printable medical report with PDF export
    ├── history.html           # Longitudinal history table & line chart
    ├── about.html             # System architecture & technology overview
    ├── model_performance.html # Performance benchmark dashboard
    └── 404.html               # Custom error handler
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AAmardeep15/LungCare-AI.git
cd LungCare-AI
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy template to .env
copy .env.example .env     # Windows
cp .env.example .env       # macOS / Linux
```

### 5. (Optional) Retrain Model & Generate Evaluation Plots
```bash
python train_model.py
python evaluate_model.py
```

### 6. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to **`http://localhost:5000`**.

---

## 🔌 REST API Endpoints

### 1. Health Status
`GET /api/health`
```json
{
  "model_loaded": true,
  "shap_available": true,
  "status": "ok"
}
```

### 2. Full Risk Prediction & Attribution
`POST /predict`
- **Request**: JSON containing all 23 numerical parameters.
- **Response**: Risk level, calibrated score, confidence, probability breakdown, SHAP drivers, early warnings, and clinical recommendations.

### 3. What-If Simulation
`POST /simulate`
- **Request**:
```json
{
  "base_data": { ... },
  "base_score": 68,
  "modifications": {
    "Smoking": 1,
    "Passive Smoker": 1,
    "Balanced Diet": 8
  }
}
```
- **Response**: Recalculated risk score, new probabilities, and point difference (`risk_delta`).

---

## ⚠️ Academic Limitations & Clinical Disclaimer

1. **Synthetic Nature of Dataset**: The underlying *Cancer Patient Data Sets* dataset is an educational benchmark exhibiting near-linear separability. Real clinical cohorts have significantly higher measurement variance and symptom overlap.
2. **Screening Support Only**: This platform is designed for **pre-clinical risk stratification**. It does not replace computed tomography, bronchoscopy, or oncological evaluation.
3. **No Regulatory Clearance**: This prototype has not undergone FDA/CE regulatory review and must not be used for autonomous triage.

---

## 📜 License
Developed for academic research and educational demonstration. All rights reserved © 2026 LungCare AI.

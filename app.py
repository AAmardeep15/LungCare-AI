import os
import json
import logging
import warnings
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

# ── Environment & logging ────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-fallback-key")

BASE       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model", "ensemble_model.pkl")
FEAT_PATH  = os.path.join(BASE, "model", "features.json")

# ── Load model & features ────────────────────────────────────────────────────
model, FEATURES = None, []
explainer = None

READABLE_FEATURE_NAMES = {
    "Age": "Patient Age",
    "Gender": "Biological Gender",
    "Air Pollution": "Air Pollution Exposure",
    "Alcohol use": "Alcohol Consumption",
    "Dust Allergy": "Dust Allergy Sensitivity",
    "OccuPational Hazards": "Occupational Workplace Hazards",
    "Genetic Risk": "Genetic / Family History",
    "chronic Lung Disease": "Chronic Lung Disease History",
    "Balanced Diet": "Diet Quality (Balanced Diet)",
    "Obesity": "Obesity / High BMI",
    "Smoking": "Active Smoking Habit",
    "Passive Smoker": "Second-hand (Passive) Smoking",
    "Chest Pain": "Chest Pain Severity",
    "Coughing of Blood": "Hemoptysis (Coughing Blood)",
    "Fatigue": "Chronic Fatigue",
    "Weight Loss": "Unexplained Weight Loss",
    "Shortness of Breath": "Dyspnea (Shortness of Breath)",
    "Wheezing": "Respiratory Wheezing",
    "Swallowing Difficulty": "Dysphagia (Swallowing Difficulty)",
    "Clubbing of Finger Nails": "Digital Clubbing",
    "Frequent Cold": "Frequent Respiratory Infections",
    "Dry Cough": "Persistent Dry Cough",
    "Snoring": "Chronic Snoring",
}

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded")
        # Initialize SHAP TreeExplainer on Random Forest estimator for fast local inference (<5ms)
        try:
            import shap
            rf_estimator = dict(model.named_estimators_).get("rf")
            if hasattr(rf_estimator, "steps"):
                rf_clf = rf_estimator.steps[-1][1]
            else:
                rf_clf = rf_estimator
            explainer = shap.TreeExplainer(rf_clf)
            logger.info("SHAP TreeExplainer initialized successfully")
        except Exception as se:
            logger.warning(f"SHAP explainer init note: {se}")
    except Exception as e:
        logger.warning(f"Model load error: {e}")

if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH) as f:
        FEATURES = json.load(f)
else:
    FEATURES = [
        "Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy",
        "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease",
        "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Chest Pain",
        "Coughing of Blood", "Fatigue", "Weight Loss", "Shortness of Breath",
        "Wheezing", "Swallowing Difficulty", "Clubbing of Finger Nails",
        "Frequent Cold", "Dry Cough", "Snoring",
    ]

# ── Recommendations dict ─────────────────────────────────────────────────────
RECS = {
    "Low": {
        "risk_color": "#10b981", "risk_label": "Low Risk", "risk_range": "10–35",
        "summary": "Your lung cancer risk is currently low. Maintain your healthy lifestyle and schedule regular preventive check-ups.",
        "clinical": [
            "Schedule an annual chest X-ray as a preventive measure",
            "Pulmonary function test (PFT / Spirometry) every 2 years",
            "Monitor any new or persisting respiratory symptoms closely",
            "Discuss your family medical history with your physician",
            "Annual complete blood count (CBC) and metabolic panel recommended",
        ],
        "avoid": [
            "Avoid starting smoking or any tobacco products",
            "Minimize exposure to second-hand smoke environments",
            "Limit alcohol to 1–2 standard units per day maximum",
            "Avoid unprotected work with asbestos, silica, or chemical fumes",
            "Avoid high-pollution areas during outdoor exercise",
        ],
        "exercise": [
            "30 minutes of brisk walking, 5 days a week",
            "Pranayama / deep breathing exercises — 10 minutes daily",
            "Light cardio (cycling, swimming) twice a week",
            "Yoga for stress management and improved lung capacity",
            "Stretching and flexibility exercises 3× per week",
        ],
        "food": [
            "Cruciferous vegetables daily: broccoli, cabbage, cauliflower",
            "2–3 servings of citrus fruits for Vitamin C (antioxidant)",
            "Turmeric with black pepper and green tea (anti-inflammatory)",
            "Omega-3 rich foods: flaxseeds, walnuts, fatty fish (salmon)",
            "Avoid ultra-processed foods; limit red meat to twice a week",
        ],
        "doctor": (
            "General practitioner consultation recommended annually. "
            "No urgent specialist referral needed at this stage."
            " General wellness guidance only — always consult a licensed physician."
        ),
        "next_steps": ["Annual health check-up", "Maintain healthy lifestyle", "Re-screen in 12 months"],
    },
    "Medium": {
        "risk_color": "#f59e0b", "risk_label": "Medium Risk", "risk_range": "35–65",
        "summary": "You have a moderate lung cancer risk. Several risk factors have been identified. Prompt medical consultation is strongly advised.",
        "clinical": [
            "Consult a Pulmonologist within the next 4–6 weeks",
            "Request a Low-Dose CT (LDCT) scan of the chest for cancer screening",
            "Complete pulmonary function testing (Spirometry + DLCO)",
            "Monitor symptoms weekly; seek emergency care if breathing worsens suddenly",
            "Consider genetic counseling if first-degree relatives had lung cancer",
            "Sputum cytology if productive cough has persisted for >3 weeks",
        ],
        "avoid": [
            "Quit smoking immediately — enroll in a nicotine replacement program",
            "Avoid ALL second-hand smoke (home, workplace, social settings)",
            "Stop or significantly reduce alcohol intake (max 1 unit/day)",
            "Never work without N95+ respirator in dusty or chemical environments",
            "Avoid indoor air pollutants: cooking fumes, incense, dusty carpets",
            "Avoid processed meats (bacon, sausages, ham) and limit red meat",
            "Avoid high-fat, high-sugar foods that promote systemic inflammation",
        ],
        "exercise": [
            "Pursed-lip breathing exercises: 15 minutes, twice daily",
            "Diaphragmatic breathing training (consider physiotherapy guidance)",
            "Low-intensity walking: 20–25 minutes/day on clean-air days only",
            "Gentle yoga (avoid strenuous poses that restrict chest expansion)",
            "Avoid exercise in polluted areas or during high pollution index days",
            "Swimming is ideal — controlled environment, low respiratory strain",
        ],
        "food": [
            "High antioxidant diet: berries, spinach, kale, bell peppers daily",
            "Eliminate processed and ultra-processed foods completely",
            "Garlic and onions daily — sulphur compounds may support general wellness",
            "Stay hydrated: 8–10 glasses of filtered water daily",
            "Beta-carotene foods: carrots, sweet potatoes (do NOT supplement if smoker)",
            "Avoid alcohol, carbonated drinks, and refined sugar entirely",
            "Increase dietary fibre: oats, lentils, beans — supports immunity",
        ],
        "doctor": (
            "Urgent referral to Pulmonologist and Oncology Screening Clinic. "
            "LDCT scan recommended within 30 days. Do not delay this appointment."
            " General wellness guidance only — always consult a licensed physician."
        ),
        "next_steps": [
            "Pulmonologist within 4–6 weeks", "LDCT scan",
            "Smoking cessation program", "Re-screen in 6 months",
        ],
    },
    "High": {
        "risk_color": "#ef4444", "risk_label": "High Risk", "risk_range": "65–95",
        "summary": (
            "⚠️ HIGH ALERT: You have a high risk of lung cancer. "
            "Multiple severe risk factors detected. Seek immediate medical attention — do not delay."
        ),
        "clinical": [
            "⚠️ Seek IMMEDIATE consultation with an Oncologist or Senior Pulmonologist",
            "Request emergency Low-Dose CT (LDCT) scan of the chest within 7 days",
            "Sputum Cytology, Bronchoscopy, and Endobronchial Ultrasound (EBUS) evaluation",
            "Complete blood count + tumor markers: CEA, CYFRA 21-1, NSE, ProGRP",
            "PET-CT scan for comprehensive metabolic staging assessment",
            "Genetic mutation testing: EGFR, ALK, ROS1, KRAS, PD-L1 expression",
            "Multidisciplinary cancer team (MDT) evaluation strongly recommended",
        ],
        "avoid": [
            "🚫 STOP SMOKING IMMEDIATELY — this is a medical emergency-level priority",
            "🚫 Avoid ALL tobacco: cigarettes, e-cigarettes, cigars, smokeless tobacco",
            "🚫 Zero alcohol consumption — it may be associated with increased health risks",
            "🚫 Avoid all outdoor air pollutants — use HEPA air purifier at home",
            "🚫 Avoid chemical fumes, dust, asbestos, radon gas — zero tolerance",
            "🚫 Avoid NSAIDs or self-medication without explicit doctor guidance",
            "🚫 Avoid stress, sleep deprivation, and extreme temperature environments",
        ],
        "exercise": [
            "Only perform light activity under physician/physiotherapist supervision",
            "Breathing rehabilitation therapy with certified chest physiotherapist",
            "Supervised diaphragmatic breathing: 10 min, 3× daily",
            "Short slow walks (5–10 minutes only) if breathing comfortably allows",
            "No strenuous exercise until pulmonologist gives explicit clearance",
            "Progressive muscle relaxation and guided meditation daily",
            "Pursed-lip breathing to relieve acute breathlessness episodes",
        ],
        "food": [
            "🥗 Strict Mediterranean diet pattern — anti-inflammatory and potentially health-supportive",
            "Lycopene-rich foods daily: tomatoes, watermelon, pink grapefruit",
            "Cruciferous vegetables every day: broccoli, Brussels sprouts, kale",
            "High-protein diet to prevent muscle wasting: eggs, legumes, lean fish",
            "Vitamin D — test blood levels first; supplement only if deficient",
            "Green tea 2–3 cups daily (EGCG may support general wellness — consult your doctor)",
            "Small, frequent meals (5–6/day) if experiencing appetite loss or nausea",
            "❌ Eliminate: processed meats, red meat, fried foods, refined sugar, alcohol",
            "Consult a clinical oncology dietitian for a personalized nutrition plan",
        ],
        "doctor": (
            "🚨 IMMEDIATE referral to Senior Oncologist + Pulmonologist. "
            "This is a high-priority medical situation. Bring this report to your appointment."
            " General wellness guidance only — always consult a licensed physician."
        ),
        "next_steps": [
            "Emergency oncology consultation NOW", "LDCT scan within 7 days",
            "Full cancer staging workup", "Genetic mutation panel",
            "Multidisciplinary team review",
        ],
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def rule_based(data):
    vals = [float(data.get(f, 1)) for f in FEATURES]
    s = sum(vals[2:])
    r = s / (21 * 9)
    if r < 0.38:   return 0, [0.78, 0.17, 0.05]
    elif r < 0.62: return 1, [0.12, 0.73, 0.15]
    else:          return 2, [0.04, 0.14, 0.82]


def early_warnings(d):
    checks = [
        ("Smoking",               7, "🚬", "Heavy smoking — primary lung cancer risk factor",                "high"),
        ("Coughing of Blood",     5, "🩸", "Hemoptysis detected — requires urgent evaluation",              "high"),
        ("Weight Loss",           6, "⚖️", "Unexplained weight loss — systemic cancer warning sign",        "high"),
        ("Shortness of Breath",   6, "😮‍💨","Significant dyspnea — potential airway compromise",           "high"),
        ("Genetic Risk",          5, "🧬", "High genetic risk — family history increases susceptibility",   "medium"),
        ("Air Pollution",         6, "🏭", "High pollution exposure — chronic lung irritant",               "medium"),
        ("Clubbing of Finger Nails", 5, "💅", "Digital clubbing — classic paraneoplastic indicator",       "medium"),
        ("chronic Lung Disease",  5, "🫁", "Chronic lung disease history — elevated malignancy risk",      "medium"),
        ("OccuPational Hazards",  6, "⚠️", "Significant occupational hazard exposure detected",            "medium"),
        ("Passive Smoker",        6, "💨", "High second-hand smoke exposure — equivalent to light smoking", "medium"),
    ]
    W = []
    for key, thresh, icon, text, sev in checks:
        if float(d.get(key, 0)) >= thresh:
            W.append({"icon": icon, "text": text, "severity": sev})
    return W[:6]


def pattern_recognition(d):
    P = []
    smoke = float(d.get("Smoking", 0))
    air   = float(d.get("Air Pollution", 0))
    cob   = float(d.get("Coughing of Blood", 0))
    wl    = float(d.get("Weight Loss", 0))
    fat   = float(d.get("Fatigue", 0))
    gen   = float(d.get("Genetic Risk", 0))
    chron = float(d.get("chronic Lung Disease", 0))
    sob   = float(d.get("Shortness of Breath", 0))
    whe   = float(d.get("Wheezing", 0))
    cp    = float(d.get("Chest Pain", 0))
    club  = float(d.get("Clubbing of Finger Nails", 0))

    if smoke >= 4 and air >= 4:
        P.append("Synergistic risk: Smoking + Air Pollution dramatically amplifies lung cancer probability")
    if cob >= 4 and wl >= 4 and fat >= 4:
        P.append("Classic triad: Hemoptysis + Weight Loss + Fatigue → high clinical suspicion for malignancy")
    if gen >= 4 and chron >= 4:
        P.append("Genetic predisposition + Chronic lung disease → elevated adenocarcinoma risk")
    if sob >= 5 and whe >= 4 and cp >= 4:
        P.append("Obstructive pattern: Dyspnea + Wheezing + Chest Pain → possible central airway involvement")
    if club >= 5:
        P.append("Digital clubbing detected → associated with pulmonary malignancy and chronic hypoxia")
    if not P:
        P.append("No critical feature-interaction patterns detected at current input levels")
    return P

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/assessment")
def assessment():
    return render_template("assessment.html")


@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/report")
def report():
    return render_template("report.html")


@app.route("/history")
def history():
    return render_template("history.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/model-performance")
def model_performance():
    metrics_path = os.path.join(BASE, "model", "metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading metrics.json: {e}")
    return render_template("model_performance.html", metrics=metrics_data)


def compute_explainability(X, parsed):
    top_raising = []
    top_lowering = []

    if explainer is not None:
        try:
            rf_est = dict(model.named_estimators_).get("rf") if model else None
            if rf_est and hasattr(rf_est, "named_steps") and "simpleimputer" in rf_est.named_steps:
                X_trans = rf_est.named_steps["simpleimputer"].transform(X)
            else:
                X_trans = X
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                high_shap = sv[2][0]
            elif len(np.shape(sv)) == 3:
                high_shap = sv[0, :, 2]
            else:
                high_shap = sv[0]

            indexed_shap = []
            for idx, f in enumerate(FEATURES):
                val = float(high_shap[idx])
                indexed_shap.append((f, val))

            raising = sorted([x for x in indexed_shap if x[1] > 0.0001], key=lambda x: x[1], reverse=True)[:5]
            lowering = sorted([x for x in indexed_shap if x[1] < -0.0001], key=lambda x: x[1])[:3]

            top_raising = [
                {
                    "feature": f,
                    "name": READABLE_FEATURE_NAMES.get(f, f),
                    "value": parsed.get(f, 0),
                    "impact": round(float(v) * 100, 2)
                }
                for f, v in raising
            ]
            top_lowering = [
                {
                    "feature": f,
                    "name": READABLE_FEATURE_NAMES.get(f, f),
                    "value": parsed.get(f, 0),
                    "impact": round(abs(float(v)) * 100, 2)
                }
                for f, v in lowering
            ]
        except Exception as e:
            logger.warning(f"SHAP compute error: {e}")

    # Fallback if SHAP had no distinct splits
    if not top_raising:
        symptoms = [f for f in FEATURES if f not in ["Age", "Gender"] and parsed.get(f, 0) >= 3]
        symptoms.sort(key=lambda f: parsed.get(f, 0), reverse=True)
        top_raising = [
            {"feature": f, "name": READABLE_FEATURE_NAMES.get(f, f), "value": parsed[f], "impact": round(parsed[f] * 4.5, 1)}
            for f in symptoms[:5]
        ]
    if not top_lowering:
        protective = [f for f in FEATURES if f not in ["Age", "Gender"] and ((f == "Balanced Diet" and parsed.get(f, 0) >= 4) or parsed.get(f, 0) <= 2)]
        top_lowering = [
            {"feature": f, "name": READABLE_FEATURE_NAMES.get(f, f), "value": parsed[f], "impact": round(3.5, 1)}
            for f in protective[:3]
        ]

    return {"raising": top_raising, "lowering": top_lowering}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        errors = []

        # ── Validate all 23 features ─────────────────────────────────────────
        parsed = {}
        for f in FEATURES:
            if f not in data:
                errors.append(f"Missing field: '{f}'")
                continue
            try:
                val = float(data[f])
            except (TypeError, ValueError):
                errors.append(f"Field '{f}' must be a number, got: {data[f]!r}")
                continue

            if f == "Age":
                if not (1 <= val <= 120):
                    errors.append(f"Age must be between 1 and 120, got {val}")
            elif f == "Gender":
                if val not in {1.0, 2.0}:
                    errors.append(f"Gender must be 1 or 2, got {val}")
            else:
                if not (1 <= val <= 9):
                    errors.append(f"Field '{f}' must be between 1 and 9, got {val}")

            parsed[f] = val

        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        # ── Predict ──────────────────────────────────────────────────────────
        X = pd.DataFrame([{f: parsed[f] for f in FEATURES}])

        if model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pred  = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0].tolist()
        else:
            pred, proba = rule_based(data)

        # proba order: [Low=0, Medium=1, High=2]
        risk_score = round(proba[1] * 50 + proba[2] * 100)
        confidence = round(max(proba) * 100, 1)

        level_map = {0: "Low", 1: "Medium", 2: "High"}
        risk = level_map[pred]

        explainability = compute_explainability(X, parsed)

        result = {
            "risk_level":  risk,
            "risk_score":  risk_score,
            "confidence":  confidence,
            "probabilities": {
                "Low":    round(proba[0] * 100, 1),
                "Medium": round(proba[1] * 100, 1),
                "High":   round(proba[2] * 100, 1),
            },
            "top_factors":     explainability,
            "early_warnings":  early_warnings(data),
            "patterns":        pattern_recognition(data),
            "recommendations": RECS[risk],
            "timestamp":       datetime.now().strftime("%B %d, %Y at %H:%M"),
            "patient_data":    data,
        }

        logger.info(f"Prediction: {risk} (score={risk_score}, confidence={confidence}%)")
        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.warning(f"Predict error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"success": False, "error": "No JSON payload provided"}), 400

        base_data = payload.get("base_data", {})
        modifications = payload.get("modifications", {})
        base_score = payload.get("base_score")

        sim_data = dict(base_data)
        sim_data.update(modifications)

        parsed = {}
        for f in FEATURES:
            val = float(sim_data.get(f, 1))
            if f == "Age":
                val = max(1.0, min(120.0, val))
            elif f == "Gender":
                val = 1.0 if val not in [1.0, 2.0] else val
            else:
                val = max(1.0, min(9.0, val))
            parsed[f] = val

        X = pd.DataFrame([{f: parsed[f] for f in FEATURES}])

        if model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pred  = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0].tolist()
        else:
            pred, proba = rule_based(parsed)

        new_risk_score = round(proba[1] * 50 + proba[2] * 100)
        level_map = {0: "Low", 1: "Medium", 2: "High"}
        new_risk_level = level_map[pred]

        risk_delta = 0
        if base_score is not None:
            risk_delta = new_risk_score - int(base_score)

        return jsonify({
            "success": True,
            "new_risk_level": new_risk_level,
            "new_risk_score": new_risk_score,
            "new_probabilities": {
                "Low":    round(proba[0] * 100, 1),
                "Medium": round(proba[1] * 100, 1),
                "High":   round(proba[2] * 100, 1),
            },
            "risk_delta": risk_delta,
            "early_warnings": early_warnings(parsed),
            "patterns": pattern_recognition(parsed)
        })
    except Exception as e:
        logger.warning(f"Simulate error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "shap_available": explainer is not None})


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info("LungCare AI starting on http://localhost:5000")
    app.run(debug=debug, port=5000)

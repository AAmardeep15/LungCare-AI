from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib, numpy as np, os, json
from datetime import datetime

app = Flask(__name__)
app.secret_key = "lungcare_ai_secret_2024_xK9mP"

BASE       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "model", "ensemble_model.pkl")
FEAT_PATH  = os.path.join(BASE, "model", "features.json")

model, FEATURES = None, []
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded")
    except Exception as e:
        print(f"⚠️ Model load error: {e}")

if os.path.exists(FEAT_PATH):
    with open(FEAT_PATH) as f:
        FEATURES = json.load(f)
else:
    FEATURES = ["Age","Gender","Air Pollution","Alcohol use","Dust Allergy",
                "OccuPational Hazards","Genetic Risk","chronic Lung Disease",
                "Balanced Diet","Obesity","Smoking","Passive Smoker","Chest Pain",
                "Coughing of Blood","Fatigue","Weight Loss","Shortness of Breath",
                "Wheezing","Swallowing Difficulty","Clubbing of Finger Nails",
                "Frequent Cold","Dry Cough","Snoring"]

RECS = {
    "Low": {
        "risk_color":"#10b981","risk_label":"Low Risk","risk_range":"10–35",
        "summary":"Your lung cancer risk is currently low. Maintain your healthy lifestyle and schedule regular preventive check-ups.",
        "clinical":[
            "Schedule an annual chest X-ray as a preventive measure",
            "Pulmonary function test (PFT / Spirometry) every 2 years",
            "Monitor any new or persisting respiratory symptoms closely",
            "Discuss your family medical history with your physician",
            "Annual complete blood count (CBC) and metabolic panel recommended"
        ],
        "avoid":[
            "Avoid starting smoking or any tobacco products",
            "Minimize exposure to second-hand smoke environments",
            "Limit alcohol to 1–2 standard units per day maximum",
            "Avoid unprotected work with asbestos, silica, or chemical fumes",
            "Avoid high-pollution areas during outdoor exercise"
        ],
        "exercise":[
            "30 minutes of brisk walking, 5 days a week",
            "Pranayama / deep breathing exercises — 10 minutes daily",
            "Light cardio (cycling, swimming) twice a week",
            "Yoga for stress management and improved lung capacity",
            "Stretching and flexibility exercises 3× per week"
        ],
        "food":[
            "Cruciferous vegetables daily: broccoli, cabbage, cauliflower",
            "2–3 servings of citrus fruits for Vitamin C (antioxidant)",
            "Turmeric with black pepper and green tea (anti-inflammatory)",
            "Omega-3 rich foods: flaxseeds, walnuts, fatty fish (salmon)",
            "Avoid ultra-processed foods; limit red meat to twice a week"
        ],
        "doctor":"General practitioner consultation recommended annually. No urgent specialist referral needed at this stage.",
        "next_steps":["Annual health check-up","Maintain healthy lifestyle","Re-screen in 12 months"]
    },
    "Medium": {
        "risk_color":"#f59e0b","risk_label":"Medium Risk","risk_range":"35–65",
        "summary":"You have a moderate lung cancer risk. Several risk factors have been identified. Prompt medical consultation is strongly advised.",
        "clinical":[
            "Consult a Pulmonologist within the next 4–6 weeks",
            "Request a Low-Dose CT (LDCT) scan of the chest for cancer screening",
            "Complete pulmonary function testing (Spirometry + DLCO)",
            "Monitor symptoms weekly; seek emergency care if breathing worsens suddenly",
            "Consider genetic counseling if first-degree relatives had lung cancer",
            "Sputum cytology if productive cough has persisted for >3 weeks"
        ],
        "avoid":[
            "Quit smoking immediately — enroll in a nicotine replacement program",
            "Avoid ALL second-hand smoke (home, workplace, social settings)",
            "Stop or significantly reduce alcohol intake (max 1 unit/day)",
            "Never work without N95+ respirator in dusty or chemical environments",
            "Avoid indoor air pollutants: cooking fumes, incense, dusty carpets",
            "Avoid processed meats (bacon, sausages, ham) and limit red meat",
            "Avoid high-fat, high-sugar foods that promote systemic inflammation"
        ],
        "exercise":[
            "Pursed-lip breathing exercises: 15 minutes, twice daily",
            "Diaphragmatic breathing training (consider physiotherapy guidance)",
            "Low-intensity walking: 20–25 minutes/day on clean-air days only",
            "Gentle yoga (avoid strenuous poses that restrict chest expansion)",
            "Avoid exercise in polluted areas or during high pollution index days",
            "Swimming is ideal — controlled environment, low respiratory strain"
        ],
        "food":[
            "High antioxidant diet: berries, spinach, kale, bell peppers daily",
            "Eliminate processed and ultra-processed foods completely",
            "Garlic and onions daily — sulphur compounds are chemoprotective",
            "Stay hydrated: 8–10 glasses of filtered water daily",
            "Beta-carotene foods: carrots, sweet potatoes (do NOT supplement if smoker)",
            "Avoid alcohol, carbonated drinks, and refined sugar entirely",
            "Increase dietary fibre: oats, lentils, beans — supports immunity"
        ],
        "doctor":"Urgent referral to Pulmonologist and Oncology Screening Clinic. LDCT scan recommended within 30 days. Do not delay this appointment.",
        "next_steps":["Pulmonologist within 4–6 weeks","LDCT scan","Smoking cessation program","Re-screen in 6 months"]
    },
    "High": {
        "risk_color":"#ef4444","risk_label":"High Risk","risk_range":"65–95",
        "summary":"⚠️ HIGH ALERT: You have a high risk of lung cancer. Multiple severe risk factors detected. Seek immediate medical attention — do not delay.",
        "clinical":[
            "⚠️ Seek IMMEDIATE consultation with an Oncologist or Senior Pulmonologist",
            "Request emergency Low-Dose CT (LDCT) scan of the chest within 7 days",
            "Sputum Cytology, Bronchoscopy, and Endobronchial Ultrasound (EBUS) evaluation",
            "Complete blood count + tumor markers: CEA, CYFRA 21-1, NSE, ProGRP",
            "PET-CT scan for comprehensive metabolic staging assessment",
            "Genetic mutation testing: EGFR, ALK, ROS1, KRAS, PD-L1 expression",
            "Multidisciplinary cancer team (MDT) evaluation strongly recommended"
        ],
        "avoid":[
            "🚫 STOP SMOKING IMMEDIATELY — this is a medical emergency-level priority",
            "🚫 Avoid ALL tobacco: cigarettes, e-cigarettes, cigars, smokeless tobacco",
            "🚫 Zero alcohol consumption — it promotes tumour angiogenesis",
            "🚫 Avoid all outdoor air pollutants — use HEPA air purifier at home",
            "🚫 Avoid chemical fumes, dust, asbestos, radon gas — zero tolerance",
            "🚫 Avoid NSAIDs or self-medication without explicit doctor guidance",
            "🚫 Avoid stress, sleep deprivation, and extreme temperature environments"
        ],
        "exercise":[
            "Only perform light activity under physician/physiotherapist supervision",
            "Breathing rehabilitation therapy with certified chest physiotherapist",
            "Supervised diaphragmatic breathing: 10 min, 3× daily",
            "Short slow walks (5–10 minutes only) if breathing comfortably allows",
            "No strenuous exercise until pulmonologist gives explicit clearance",
            "Progressive muscle relaxation and guided meditation daily",
            "Pursed-lip breathing to relieve acute breathlessness episodes"
        ],
        "food":[
            "🥗 Strict Mediterranean diet pattern — anti-inflammatory and anti-cancer",
            "Lycopene-rich foods daily: tomatoes, watermelon, pink grapefruit",
            "Cruciferous vegetables every day: broccoli, Brussels sprouts, kale",
            "High-protein diet to prevent muscle wasting: eggs, legumes, lean fish",
            "Vitamin D — test blood levels first; supplement only if deficient",
            "Green tea 2–3 cups daily (EGCG has documented anti-tumour properties)",
            "Small, frequent meals (5–6/day) if experiencing appetite loss or nausea",
            "❌ Eliminate: processed meats, red meat, fried foods, refined sugar, alcohol",
            "Consult a clinical oncology dietitian for a personalized nutrition plan"
        ],
        "doctor":"🚨 IMMEDIATE referral to Senior Oncologist + Pulmonologist. This is a high-priority medical situation. Bring this report to your appointment.",
        "next_steps":["Emergency oncology consultation NOW","LDCT scan within 7 days","Full cancer staging workup","Genetic mutation panel","Multidisciplinary team review"]
    }
}

def rule_based(data):
    vals = [float(data.get(f, 1)) for f in FEATURES]
    s = sum(vals[2:])
    r = s / (21 * 9)
    if r < 0.38: return 0, [0.78, 0.17, 0.05]
    elif r < 0.62: return 1, [0.12, 0.73, 0.15]
    else: return 2, [0.04, 0.14, 0.82]

def early_warnings(d):
    checks = [
        ("Smoking",7,"🚬","Heavy smoking — primary lung cancer risk factor","high"),
        ("Coughing of Blood",5,"🩸","Hemoptysis detected — requires urgent evaluation","high"),
        ("Weight Loss",6,"⚖️","Unexplained weight loss — systemic cancer warning sign","high"),
        ("Shortness of Breath",6,"😮‍💨","Significant dyspnea — potential airway compromise","high"),
        ("Genetic Risk",5,"🧬","High genetic risk — family history increases susceptibility","medium"),
        ("Air Pollution",6,"🏭","High pollution exposure — chronic lung irritant","medium"),
        ("Clubbing of Finger Nails",5,"💅","Digital clubbing — classic paraneoplastic indicator","medium"),
        ("chronic Lung Disease",5,"🫁","Chronic lung disease history — elevated malignancy risk","medium"),
        ("OccuPational Hazards",6,"⚠️","Significant occupational hazard exposure detected","medium"),
        ("Passive Smoker",6,"💨","High second-hand smoke exposure — equivalent to light smoking","medium"),
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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feat_vals = [float(data.get(f, 1)) for f in FEATURES]
        X = np.array(feat_vals).reshape(1, -1)

        if model:
            pred  = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0].tolist()
        else:
            pred, proba = rule_based(data)

        level_map = {0: "Low", 1: "Medium", 2: "High"}
        risk  = level_map[pred]
        score = int(max(proba) * 100)

        result = {
            "risk_level":      risk,
            "risk_score":      score,
            "probabilities": {
                "Low":    round(proba[0] * 100, 1),
                "Medium": round(proba[1] * 100, 1),
                "High":   round(proba[2] * 100, 1)
            },
            "early_warnings":  early_warnings(data),
            "patterns":        pattern_recognition(data),
            "recommendations": RECS[risk],
            "timestamp":       datetime.now().strftime("%B %d, %Y at %H:%M"),
            "patient_data":    data
        }
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    print("🫁 LungCare AI starting on http://localhost:5000")
    app.run(debug=True, port=5000)

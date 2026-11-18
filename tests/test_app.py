import sys
import os
import json
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test that /api/health responds with 200 and model_loaded: True."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_predict_valid_sample(client):
    """Test /predict with a complete valid 23-feature payload."""
    payload = {
        "Age": 45, "Gender": 1, "Air Pollution": 3, "Alcohol use": 2, "Dust Allergy": 2,
        "OccuPational Hazards": 3, "Genetic Risk": 2, "chronic Lung Disease": 1,
        "Balanced Diet": 6, "Obesity": 2, "Smoking": 2, "Passive Smoker": 2, "Chest Pain": 2,
        "Coughing of Blood": 1, "Fatigue": 2, "Weight Loss": 1, "Shortness of Breath": 2,
        "Wheezing": 1, "Swallowing Difficulty": 1, "Clubbing of Finger Nails": 1,
        "Frequent Cold": 2, "Dry Cough": 2, "Snoring": 2
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    result = data["result"]
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert 0 <= result["risk_score"] <= 100
    assert 0 <= result["confidence"] <= 100
    assert "top_factors" in result
    assert "raising" in result["top_factors"]
    assert "lowering" in result["top_factors"]

def test_predict_missing_field(client):
    """Test /predict returns 400 when required fields are missing."""
    incomplete_payload = {
        "Age": 45,
        "Gender": 1
        # missing 21 other features
    }
    res = client.post("/predict", json=incomplete_payload)
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "errors" in data
    assert len(data["errors"]) == 21

def test_predict_out_of_range_field(client):
    """Test /predict returns 400 when values violate boundary constraints."""
    invalid_payload = {
        "Age": 150,  # Invalid: max 120
        "Gender": 3,  # Invalid: must be 1 or 2
        "Air Pollution": 15,  # Invalid: must be 1-9
        "Alcohol use": 2, "Dust Allergy": 2,
        "OccuPational Hazards": 3, "Genetic Risk": 2, "chronic Lung Disease": 1,
        "Balanced Diet": 6, "Obesity": 2, "Smoking": 2, "Passive Smoker": 2, "Chest Pain": 2,
        "Coughing of Blood": 1, "Fatigue": 2, "Weight Loss": 1, "Shortness of Breath": 2,
        "Wheezing": 1, "Swallowing Difficulty": 1, "Clubbing of Finger Nails": 1,
        "Frequent Cold": 2, "Dry Cough": 2, "Snoring": 2
    }
    res = client.post("/predict", json=invalid_payload)
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert any("Age" in err for err in data["errors"])
    assert any("Gender" in err for err in data["errors"])
    assert any("Air Pollution" in err for err in data["errors"])

def test_simulate_endpoint(client):
    """Test /simulate endpoint recalculates risk scores and delta."""
    base_data = {
        "Age": 55, "Gender": 1, "Air Pollution": 6, "Alcohol use": 5, "Dust Allergy": 5,
        "OccuPational Hazards": 6, "Genetic Risk": 6, "chronic Lung Disease": 5,
        "Balanced Diet": 3, "Obesity": 5, "Smoking": 7, "Passive Smoker": 6, "Chest Pain": 6,
        "Coughing of Blood": 5, "Fatigue": 6, "Weight Loss": 5, "Shortness of Breath": 6,
        "Wheezing": 5, "Swallowing Difficulty": 4, "Clubbing of Finger Nails": 4,
        "Frequent Cold": 5, "Dry Cough": 6, "Snoring": 4
    }
    sim_payload = {
        "base_data": base_data,
        "base_score": 75,
        "modifications": {
            "Smoking": 1,
            "Passive Smoker": 1,
            "Balanced Diet": 9,
            "Alcohol use": 1
        }
    }
    res = client.post("/simulate", json=sim_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "new_risk_score" in data
    assert "new_risk_level" in data
    assert "risk_delta" in data
    # Improving lifestyle should reduce risk score relative to 75
    assert data["new_risk_score"] < 75
    assert data["risk_delta"] < 0

def test_chat_fallback_without_api_key(client, monkeypatch):
    """Test /chat gracefully returns keyword FAQ fallback when GROQ_API_KEY is not set."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    chat_payload = {
        "message": "What is an LDCT scan and how does it detect lung cancer?",
        "history": [],
        "patient_context": {"risk_level": "Low", "risk_score": 15}
    }
    res = client.post("/chat", json=chat_payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert data["source"] == "rule_based_fallback"
    assert "Low-Dose Computed Tomography" in data["reply"]

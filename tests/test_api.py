from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# datos de ejemplo para los tests
SAMPLE_INPUT = {
    "gender": "Male",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 90.5,
    "bmi": 28.0,
    "smoking_status": "never smoked",
}


def test_root(): # verifica que la api responde
    response = client.get("/")
    assert response.status_code == 200


def test_health(): # verifica el endpoint de salud
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_no_stroke(): # simula prediccion sin stroke
    mock_result = {"prediction": 0, "probabilities": [0.85, 0.15]}
    with patch("app.main.predict", return_value=mock_result):
        response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    assert response.json()["risk"] == "bajo"


def test_predict_stroke(): # simula prediccion con stroke
    mock_result = {"prediction": 1, "probabilities": [0.2, 0.8]}
    with patch("app.main.predict", return_value=mock_result):
        response = client.post("/predict", json=SAMPLE_INPUT)
    assert response.status_code == 200
    assert response.json()["risk"] == "alto"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.utils import predict, reload_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="stroke prediction api", version="1.0.0")

#mapeos para convertir texto a numeros (mismo orden que el labelencoder del train.py)
GENDER_MAP = {"Female": 0, "Male": 1, "Other": 2}
MARRIED_MAP = {"No": 0, "Yes": 1}
WORK_MAP = {"Govt_job": 0, "Never_worked": 1, "Private": 2, "Self-employed": 3, "children": 4}
RESIDENCE_MAP = {"Rural": 0, "Urban": 1}
SMOKING_MAP = {"Unknown": 0, "formerly smoked": 1, "never smoked": 2, "smokes": 3}

class StrokeRequest(BaseModel): #modelo de entrada
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class PredictResponse(BaseModel): #modelo de salida
    prediction: int
    probabilities: list[float]
    risk: str

def encode_input(req: StrokeRequest) -> list: #convierte el request a lista de numeros
    return [
        GENDER_MAP.get(req.gender, 0),
        req.age,
        req.hypertension,
        req.heart_disease,
        MARRIED_MAP.get(req.ever_married, 0),
        WORK_MAP.get(req.work_type, 2),
        RESIDENCE_MAP.get(req.Residence_type, 1),
        req.avg_glucose_level,
        req.bmi,
        SMOKING_MAP.get(req.smoking_status, 0),
    ]

@app.get("/")
def root():
    return {"status": "ok", "message": "stroke prediction api corriendo"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResponse)
def make_prediction(request: StrokeRequest):
    try:
        features = encode_input(request) #codifica el input
        result = predict(features) #llama a utils.py
        result["risk"] = "alto" if result["prediction"] == 1 else "bajo" #agrega etiqueta de riesgo
        return result
    except Exception as e:
        logger.error(f"error en prediccion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
def reload():
    try:
        reload_model() #descarga el modelo mas reciente desde s3
        return {"status": "ok", "message": "modelo recargado desde s3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
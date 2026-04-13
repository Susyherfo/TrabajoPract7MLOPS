"""
app/utils.py

Funciones auxiliares para:
- Descargar el modelo desde S3
- Cargar el modelo en memoria
- Realizar predicciones
"""

import os
import logging

import boto3
import joblib
import numpy as np


logger = logging.getLogger(_name_)

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "mi-bucket-mlops")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY", "models/latest/model.pkl")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(_file_), "..", "model", "model.pkl")

_model = None


def download_model_from_s3() -> bool:
    try:
        logger.info(f"Descargando modelo desde s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, LOCAL_MODEL_PATH)
        logger.info("Modelo descargado exitosamente.")
        return True
    except Exception as e:
        logger.error(f"Error descargando modelo desde S3: {e}")
        return False


def load_model():
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Modelo no encontrado localmente, descargando desde S3...")
        success = download_model_from_s3()
        if not success:
            raise RuntimeError(
                "No se pudo cargar el modelo. "
                "Verifica que existe en S3 o en model/model.pkl."
            )

    _model = joblib.load(LOCAL_MODEL_PATH)
    logger.info("Modelo cargado en memoria correctamente.")
    return _model


def predict(features: list) -> dict:
    model = load_model()
    X = np.array(features).reshape(1, -1)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0].tolist()
    probabilities = [round(p, 4) for p in probabilities]

    return {
        "prediction": prediction,
        "probabilities": probabilities,
    }


def reload_model():
    global _model
    _model = None
    logger.info("Recargando modelo desde S3...")
    download_model_from_s3()
    return load_model()
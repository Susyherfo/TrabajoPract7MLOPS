"""
scripts/train.py

Script de entrenamiento del modelo de ML.
- Descarga el dataset desde S3
- Entrena un modelo de clasificación (RandomForest)
- Guarda el modelo localmente y lo sube a S3 con versionado

Agregar esto al README:
Para correrlo apuntando a S3 (en producción o en el pipeline de retrain):

bash

export S3_BUCKET_NAME=nombre-de-tu-bucket
export TARGET_COLUMN=target   # nombre de tu columna objetivo
python scripts/train.py
"""

import os
import io
import json
import logging
import argparse
from datetime import datetime

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# Configuración de logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "mi-bucket-mlops")
S3_DATASET_KEY = os.getenv("S3_DATASET_KEY", "data/dataset.csv")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
LOCAL_METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# ─────────────────────────────────────────────
# Funciones auxiliares S3
# ─────────────────────────────────────────────


def get_s3_client():

    """Crea y retorna un cliente de S3."""
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def download_dataset_from_s3(bucket: str, key: str) -> pd.DataFrame:

    """Descarga un CSV desde S3 y lo retorna como DataFrame."""
    logger.info(f"Descargando dataset desde s3://{bucket}/{key}")
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    logger.info(f"Dataset descargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def upload_model_to_s3(local_path: str, bucket: str, prefix: str, version: str):

    """Sube el modelo local a S3 en dos versiones: versionada y 'latest'."""
    s3 = get_s3_client()

    versioned_key = f"{prefix}{version}/model.pkl"
    latest_key = f"{prefix}latest/model.pkl"

    for key in [versioned_key, latest_key]:
        logger.info(f"Subiendo modelo a s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)

    logger.info("Modelo subido exitosamente a S3.")
    return versioned_key


def upload_metrics_to_s3(metrics: dict, bucket: str, prefix: str, version: str):
    
    """Sube las métricas del entrenamiento a S3 en formato JSON."""
    s3 = get_s3_client()

    versioned_key = f"{prefix}{version}/metrics.json"
    latest_key = f"{prefix}latest/metrics.json"
    payload = json.dumps(metrics, indent=2).encode("utf-8")

    for key in [versioned_key, latest_key]:
        logger.info(f"Subiendo métricas a s3://{bucket}/{key}")
        s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")

    logger.info("Métricas subidas exitosamente a S3.")


# ─────────────────────────────────────────────
# Preparación de datos
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, target_col: str):
    """
    Prepara features y target para el entrenamiento.
    - Elimina filas con NaN
    - Codifica columnas categóricas automáticamente
    - Separa X e y
    """
    logger.info("Preparando datos...")

    if target_col not in df.columns:
        raise ValueError(f"La columna target '{target_col}' no existe en el dataset.")

    df = df.dropna()
    logger.info(f"Filas después de eliminar NaN: {len(df)}")

    # Codificar columnas categóricas (excepto el target)
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Columna '{col}' codificada.")

    # Codificar target si es categórico
    if df[target_col].dtype == object:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        label_encoders[target_col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Distribución del target:\n{y.value_counts().to_string()}")

    return X, y, label_encoders


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestClassifier:
    """Entrena un RandomForestClassifier."""
    logger.info("Entrenando modelo RandomForest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado.")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Evalúa el modelo y retorna un diccionario con las métricas."""
    logger.info("Evaluando modelo...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "n_test_samples": len(y_test),
        "feature_importances": dict(zip(
            [f"feature_{i}" for i in range(X_test.shape[1])],
            [round(float(v), 4) for v in model.feature_importances_],
        )),
    }

    logger.info(f"Accuracy en test: {accuracy:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return metrics


# ─────────────────────────────────────────────
# Generación de datos de ejemplo (si no hay S3)
# ─────────────────────────────────────────────

def generate_sample_dataset() -> pd.DataFrame:
    """
    Genera un dataset de ejemplo (Iris-like) para pruebas locales.
    Se usa cuando no hay conexión a S3 (flag --local).
    """
    logger.info("Generando dataset de ejemplo local...")
    np.random.seed(RANDOM_STATE)
    n = 300

    df = pd.DataFrame({
        "feature_1": np.random.normal(0, 1, n),
        "feature_2": np.random.normal(1, 1.5, n),
        "feature_3": np.random.uniform(0, 5, n),
        "feature_4": np.random.normal(2, 0.5, n),
        "target": np.random.choice([0, 1, 2], size=n),
    })

    logger.info(f"Dataset de ejemplo generado: {df.shape}")
    return df


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────


def run_training_pipeline(use_local: bool = False):

    """
    Ejecuta el pipeline completo de entrenamiento:
    1. Cargar datos (S3 o local)
    2. Preparar datos
    3. Entrenar modelo
    4. Evaluar modelo
    5. Guardar modelo localmente
    6. Subir modelo y métricas a S3
    """
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Iniciando pipeline de entrenamiento — versión: {version}")

    # ── 1. Cargar datos ──────────────────────
    if use_local:
        df = generate_sample_dataset()
    else:
        df = download_dataset_from_s3(S3_BUCKET, S3_DATASET_KEY)

    # ── 2. Preparar datos ────────────────────
    X, y, _ = prepare_data(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

    # ── 3. Entrenar ──────────────────────────
    model = train_model(X_train, y_train)

    # ── 4. Evaluar ───────────────────────────
    metrics = evaluate_model(model, X_test, y_test)
    metrics["version"] = version
    metrics["trained_at"] = datetime.utcnow().isoformat()

    # ── 5. Guardar localmente ────────────────
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    joblib.dump(model, LOCAL_MODEL_PATH)
    logger.info(f"Modelo guardado localmente en: {LOCAL_MODEL_PATH}")

    with open(LOCAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métricas guardadas localmente en: {LOCAL_METRICS_PATH}")

    # ── 6. Subir a S3 ────────────────────────
    if not use_local:
        upload_model_to_s3(LOCAL_MODEL_PATH, S3_BUCKET, S3_MODEL_PREFIX, version)
        upload_metrics_to_s3(metrics, S3_BUCKET, S3_MODEL_PREFIX, version)
    else:
        logger.info("Modo local: omitiendo subida a S3.")

    logger.info(f"Pipeline completado exitosamente. Accuracy: {metrics['accuracy']}")
    return metrics


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y versiona el modelo de ML.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Usar dataset de ejemplo local en lugar de S3 (para desarrollo/testing).",
    )
    args = parser.parse_args()

    run_training_pipeline(use_local=args.local)


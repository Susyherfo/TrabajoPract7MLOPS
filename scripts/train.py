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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "mi-bucket-mlops")
S3_DATASET_KEY = os.getenv("S3_DATASET_KEY", "data/dataset.csv")
S3_MODEL_PREFIX = os.getenv("S3_MODEL_PREFIX", "models/")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
LOCAL_METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")

TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def download_dataset_from_s3(bucket: str, key: str) -> pd.DataFrame:
    logger.info(f"Descargando dataset desde s3://{bucket}/{key}")
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    logger.info(f"Dataset descargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def upload_model_to_s3(local_path: str, bucket: str, prefix: str, version: str):
    s3 = get_s3_client()
    versioned_key = f"{prefix}{version}/model.pkl"
    latest_key = f"{prefix}latest/model.pkl"

    for key in [versioned_key, latest_key]:
        logger.info(f"Subiendo modelo a s3://{bucket}/{key}")
        s3.upload_file(local_path, bucket, key)

    logger.info("Modelo subido exitosamente a S3.")
    return versioned_key


def upload_metrics_to_s3(metrics: dict, bucket: str, prefix: str, version: str):
    s3 = get_s3_client()
    versioned_key = f"{prefix}{version}/metrics.json"
    latest_key = f"{prefix}latest/metrics.json"
    payload = json.dumps(metrics, indent=2).encode("utf-8")

    for key in [versioned_key, latest_key]:
        logger.info(f"Subiendo métricas a s3://{bucket}/{key}")
        s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")

    logger.info("Métricas subidas exitosamente a S3.")


def prepare_data(df: pd.DataFrame, target_col: str):
    logger.info("Preparando datos...")

    if target_col not in df.columns:
        raise ValueError(f"La columna target '{target_col}' no existe en el dataset.")

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df.dropna()
    logger.info(f"Filas después de eliminar NaN: {len(df)}")

    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    if df[target_col].dtype == object:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        label_encoders[target_col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y, label_encoders


def train_model(X_train, y_train) -> RandomForestClassifier:
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
    return metrics


def generate_sample_dataset() -> pd.DataFrame:
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

    return df


def run_training_pipeline(use_local: bool = False):
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Iniciando pipeline de entrenamiento — versión: {version}")

    if use_local:
        df = generate_sample_dataset()
    else:
        df = download_dataset_from_s3(S3_BUCKET, S3_DATASET_KEY)

    X, y, _ = prepare_data(df, TARGET_COLUMN)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    metrics["version"] = version
    metrics["trained_at"] = datetime.utcnow().isoformat()

    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    joblib.dump(model, LOCAL_MODEL_PATH)

    with open(LOCAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if not use_local:
        upload_model_to_s3(LOCAL_MODEL_PATH, S3_BUCKET, S3_MODEL_PREFIX, version)
        upload_metrics_to_s3(metrics, S3_BUCKET, S3_MODEL_PREFIX, version)

    logger.info(f"Pipeline completado. Accuracy: {metrics['accuracy']}")
    return metrics


if __name__ == "_main_":
    parser = argparse.ArgumentParser(description="Entrena y versiona el modelo de ML.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Usar dataset de ejemplo local en lugar de S3.",
    )
    args = parser.parse_args()

    run_training_pipeline(use_local=args.local)
# fin
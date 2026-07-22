"""
API de scoring. Charge model.joblib au démarrage plutôt que MLflow à
chaque appel (latence).

Limite connue : sans feature store, les features d'historique utilisateur
doivent être passées par l'appelant. En vrai il faudrait un Redis ou un
Feast derrière - hors scope ici, c'est juste pour montrer le endpoint.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.common.config import settings
from src.features.build_features import (
    add_amount_features, add_time_features, add_velocity_features, encode_categoricals,
)
from src.serving.schemas import PredictionResponse, TransactionRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model_bundle: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model_bundle()
    except FileNotFoundError as exc:
        # On ne bloque pas le démarrage (utile pour les tests / CI qui ne
        # veulent pas forcément entraîner un modèle), mais /predict échouera
        # explicitement tant qu'aucun modèle n'est disponible.
        logger.warning(str(exc))
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Sert les prédictions du modèle de détection de fraude entraîné via src/training/train.py",
    version="1.0.0",
    lifespan=lifespan,
)


def _model_path():
    return settings.model_dir / "model.joblib"


def load_model_bundle(force: bool = False) -> dict:
    global _model_bundle
    if _model_bundle is None or force:
        path = _model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable à {path}. Lance d'abord `python -m src.training.train`."
            )
        _model_bundle = joblib.load(path)
        logger.info("Modèle chargé depuis %s (features=%d)",
                     path, len(_model_bundle["feature_columns"]))
    return _model_bundle


def _apply_online_features(payload: TransactionRequest) -> pd.DataFrame:
    """Recalcule les mêmes features que le batch, mais pour une transaction
    seule. Si l'appelant ne fournit pas l'historique du user (compteurs,
    moyenne), on met des valeurs neutres au lieu de planter la requête -
    pas idéal, mais pas de feature store pour l'instant."""
    row = {
        "transaction_id": payload.transaction_id,
        "user_id": payload.user_id,
        "card_id": payload.card_id,
        "timestamp": payload.timestamp,
        "amount": payload.amount,
        "merchant_category": payload.merchant_category,
        "merchant_country": payload.merchant_country,
        "device_type": payload.device_type,
        "transaction_type": payload.transaction_type,
        "user_txn_count_before": payload.user_txn_count_before or 0,
        "user_avg_amount_before": payload.user_avg_amount_before or payload.amount,
        "card_distinct_countries_so_far": payload.card_distinct_countries_so_far or 1,
        "seconds_since_last_user_txn": payload.seconds_since_last_user_txn or 3600.0,
    }
    df = pd.DataFrame([row])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = encode_categoricals(df)
    return df


@app.get("/health")
def health() -> dict:
    model_available = _model_path().exists()
    return {"status": "ok", "model_available": model_available}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionRequest) -> PredictionResponse:
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    features_df = _apply_online_features(payload)

    for col in bundle["feature_columns"]:
        if col not in features_df.columns:
            features_df[col] = 0
    X = features_df[bundle["feature_columns"]].astype(float)

    proba = float(bundle["model"].predict_proba(X)[:, 1][0])
    threshold = bundle["threshold"]

    return PredictionResponse(
        transaction_id=payload.transaction_id,
        fraud_probability=round(proba, 6),
        is_fraud_predicted=proba >= threshold,
        threshold_used=threshold,
        model_version=_model_path().stat().st_mtime_ns.__str__(),
    )

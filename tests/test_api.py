from __future__ import annotations

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from src.common.config import settings
from src.features.build_features import FEATURE_COLUMNS


@pytest.fixture
def client_with_dummy_model(monkeypatch):
    """Entraîne un mini modèle bidon (données aléatoires) juste pour tester
    le contrat de l'API (schéma de requête/réponse), sans dépendre d'un vrai
    entraînement complet dans les tests unitaires."""
    import numpy as np

    onehot_cols = [
        "merchant_category_grocery", "merchant_category_travel",
        "merchant_country_FR", "merchant_country_RU",
        "device_type_mobile_app", "device_type_web",
        "transaction_type_purchase", "transaction_type_online_payment",
    ]
    feature_columns = FEATURE_COLUMNS + onehot_cols

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(feature_columns)))
    y = rng.integers(0, 2, size=200)

    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X, y)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.model_dir / "model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_columns, "threshold": 0.5}, model_path)

    import src.serving.api as api_module
    api_module._model_bundle = None

    with TestClient(api_module.app) as c:
        yield c

    try:
        if model_path.exists():
            model_path.unlink()
    except OSError:
        pass  # nettoyage best-effort, ne doit pas faire échouer le test


def test_health_endpoint(client_with_dummy_model):
    resp = client_with_dummy_model.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_valid_response(client_with_dummy_model):
    payload = {
        "transaction_id": 1,
        "user_id": 1,
        "card_id": 11,
        "timestamp": "2026-07-13T02:14:00",
        "amount": 1287.50,
        "merchant_category": "electronics",
        "merchant_country": "RU",
        "device_type": "web",
        "transaction_type": "online_payment",
    }
    resp = client_with_dummy_model.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["transaction_id"] == 1
    assert isinstance(body["is_fraud_predicted"], bool)


def test_predict_rejects_negative_amount(client_with_dummy_model):
    payload = {
        "transaction_id": 2,
        "user_id": 1,
        "card_id": 11,
        "timestamp": "2026-07-13T02:14:00",
        "amount": -10,
        "merchant_category": "electronics",
        "merchant_country": "RU",
        "device_type": "web",
        "transaction_type": "online_payment",
    }
    resp = client_with_dummy_model.post("/predict", json=payload)
    assert resp.status_code == 422

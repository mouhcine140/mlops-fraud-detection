from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Payload attendu par /predict : une transaction brute (pas encore
    transformée). L'API applique elle-même le feature engineering nécessaire
    pour rester cohérente avec le pipeline batch."""

    transaction_id: int
    user_id: int
    card_id: int
    timestamp: datetime
    amount: float = Field(gt=0)
    merchant_category: str
    merchant_country: str
    device_type: str
    transaction_type: str

    # Historique utilisateur optionnel : si absent, l'API applique des valeurs
    # neutres (voir _apply_online_features).
    user_txn_count_before: float | None = None
    user_avg_amount_before: float | None = None
    card_distinct_countries_so_far: float | None = None
    seconds_since_last_user_txn: float | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": 999001,
                "user_id": 42,
                "card_id": 421,
                "timestamp": "2026-07-13T02:14:00Z",
                "amount": 1287.50,
                "merchant_category": "electronics",
                "merchant_country": "RU",
                "device_type": "web",
                "transaction_type": "online_payment",
            }
        }
    )


class PredictionResponse(BaseModel):
    transaction_id: int
    fraud_probability: float
    is_fraud_predicted: bool
    threshold_used: float
    model_version: str

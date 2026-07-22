"""
Charge le CSV brut (data/raw/) dans `raw_transactions`. Batch full-load pour
l'instant ; à remplacer par du CDC / consumer Kafka en prod si besoin.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.common.config import settings
from src.common.db import write_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_TABLE = "raw_transactions"

EXPECTED_COLUMNS = [
    "transaction_id", "user_id", "card_id", "timestamp", "amount",
    "merchant_category", "merchant_country", "device_type",
    "transaction_type", "is_fraud",
]


def load_raw_csv(path=None) -> pd.DataFrame:
    path = path or settings.raw_data_path
    logger.info("Lecture du CSV brut : %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier source : {missing}")

    return df[EXPECTED_COLUMNS]


def ingest(path=None) -> int:
    df = load_raw_csv(path)
    n_rows = write_dataframe(df, RAW_TABLE, if_exists="replace")
    logger.info("Ingestion terminée : %d lignes chargées dans '%s'", n_rows, RAW_TABLE)
    return n_rows


if __name__ == "__main__":
    ingest()

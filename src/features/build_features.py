"""
Deux étapes : le SQL fait les agrégations lourdes (historique
utilisateur/carte), pandas fait le reste. J'aurais pu tout faire en pandas
mais charger toute la table pour recalculer des window functions en mémoire
aurait été plus lent qu'en laissant le moteur SQL le faire.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import settings
from src.common.db import get_engine, read_sql, write_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent
CATEGORICAL_COLUMNS = ["merchant_category", "merchant_country", "device_type", "transaction_type"]
FEATURES_TABLE = "features"


def _load_sql_for_dialect() -> str:
    dialect = get_engine().dialect.name  # "sqlite" | "postgresql"
    filename = "aggregations_postgres.sql" if dialect.startswith("postgres") else "aggregations_sqlite.sql"
    query_path = SQL_DIR / filename
    logger.info("Dialecte détecté : %s -> requête %s", dialect, filename)
    return query_path.read_text()


def run_sql_aggregations() -> pd.DataFrame:
    query = _load_sql_for_dialect()
    df = read_sql(query)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["txn_hour"] = df["timestamp"].dt.hour
    df["txn_day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night_txn"] = df["txn_hour"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
    df["is_weekend_txn"] = (df["txn_day_of_week"] >= 5).astype(int)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])

    # Ratio du montant vs. moyenne historique de l'utilisateur (>1 = transaction
    # inhabituellement élevée pour ce user). On évite la division par zéro pour
    # les tout premiers achats d'un utilisateur (pas d'historique -> ratio = 1).
    safe_avg = df["user_avg_amount_before"].astype(float).replace(0, np.nan)
    ratio = (df["amount"] / safe_avg).replace([np.inf, -np.inf], np.nan)
    df["amount_to_user_avg_ratio"] = ratio.fillna(1.0).astype(float)

    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Nouvel utilisateur / pas d'historique -> on neutralise plutôt que NaN
    df["user_txn_count_before"] = df["user_txn_count_before"].fillna(0)
    df["user_avg_amount_before"] = df["user_avg_amount_before"].fillna(df["amount"])
    df["seconds_since_last_user_txn"] = df["seconds_since_last_user_txn"].fillna(
        df["seconds_since_last_user_txn"].median() if df["seconds_since_last_user_txn"].notna().any() else 0
    )
    df["is_first_transaction"] = (df["user_txn_count_before"] == 0).astype(int)
    # Transaction "rapide" après la précédente (< 60s) : signal classique de
    # test de carte / script automatisé.
    df["is_rapid_succession"] = (df["seconds_since_last_user_txn"] < 60).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One-hot encoding simple. `reference_df` (optionnel) sert à figer les
    colonnes attendues (utile en inférence pour garder le même schéma qu'à
    l'entraînement, même si une catégorie rare n'apparaît pas dans le batch)."""
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)

    if reference_df is not None:
        expected_cols = [c for c in reference_df.columns if any(c.startswith(p + "_") for p in CATEGORICAL_COLUMNS)]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        extra_cols = [
            c for c in df.columns
            if any(c.startswith(p + "_") for p in CATEGORICAL_COLUMNS) and c not in expected_cols
        ]
        df = df.drop(columns=extra_cols)

    return df


FEATURE_COLUMNS = [
    "amount", "log_amount", "amount_to_user_avg_ratio",
    "user_txn_count_before", "user_avg_amount_before",
    "card_distinct_countries_so_far", "seconds_since_last_user_txn",
    "is_first_transaction", "is_rapid_succession",
    "txn_hour", "txn_day_of_week", "is_night_txn", "is_weekend_txn",
]


def build_features(persist: bool = True) -> pd.DataFrame:
    df = run_sql_aggregations()
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = encode_categoricals(df)

    logger.info("Features construites : %d lignes, %d colonnes", *df.shape)

    if persist:
        write_dataframe(df, FEATURES_TABLE, if_exists="replace")

    return df


if __name__ == "__main__":
    build_features()

"""
Génère des transactions synthétiques (fraude ~1.5%, patterns pas trop
subtils : montants élevés, nuit, pays étranger).

Le vrai dataset Kaggle demande un compte et fait 150+ Mo, pas pratique à
committer. Limite honnête : ces patterns sont plus faciles à repérer que de
la vraie fraude, donc les métriques du modèle sont sûrement optimistes par
rapport à un cas réel.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.common.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel", "fashion",
    "entertainment", "utilities", "health", "online_marketplace", "gas_station",
]
COUNTRIES = ["FR", "DE", "BE", "ES", "IT", "GB", "US", "NG", "RU", "CN"]
# Pays "domestiques" les plus fréquents pour la majorité des utilisateurs.
DOMESTIC_COUNTRIES = ["FR", "BE", "DE"]
DEVICE_TYPES = ["mobile_app", "web", "pos_chip", "pos_nfc", "atm"]
TRANSACTION_TYPES = ["purchase", "withdrawal", "online_payment", "transfer"]


def _sample_categorical(rng: np.random.Generator, values: list[str], size: int,
                         weights: list[float] | None = None) -> np.ndarray:
    return rng.choice(values, size=size, p=weights)


def generate_transactions(n_rows: int = 20_000, fraud_rate: float = 0.015,
                           n_users: int = 1_500, seed: int = 42) -> pd.DataFrame:
    """Génère un DataFrame de transactions avec une colonne cible `is_fraud`."""
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    user_ids = rng.integers(1, n_users + 1, size=n_rows)
    card_ids = user_ids * 10 + rng.integers(0, 3, size=n_rows)  # 1-3 cartes/utilisateur

    # --- Transactions légitimes : comportement "normal" ---
    legit_hours = rng.normal(loc=14, scale=4, size=n_legit).clip(0, 23).astype(int)
    legit_amounts = np.round(rng.lognormal(mean=3.2, sigma=0.9, size=n_legit), 2)
    legit_country = _sample_categorical(
        rng, DOMESTIC_COUNTRIES + COUNTRIES,
        n_legit,
        weights=_normalized_weights(len(DOMESTIC_COUNTRIES), len(COUNTRIES)),
    )
    legit_is_fraud = np.zeros(n_legit, dtype=int)

    # --- Transactions frauduleuses : patterns atypiques ---
    fraud_hours = rng.choice(
        list(range(0, 6)) + list(range(22, 24)), size=n_fraud
    )  # nuit / petit matin
    fraud_amounts = np.round(rng.lognormal(mean=5.0, sigma=1.1, size=n_fraud), 2)
    fraud_country = rng.choice(COUNTRIES, size=n_fraud)  # pas de biais domestique
    fraud_is_fraud = np.ones(n_fraud, dtype=int)

    hours = np.concatenate([legit_hours, fraud_hours])
    amounts = np.concatenate([legit_amounts, fraud_amounts])
    countries = np.concatenate([legit_country, fraud_country])
    is_fraud = np.concatenate([legit_is_fraud, fraud_is_fraud])

    n_total = n_legit + n_fraud
    merchant_category = _sample_categorical(rng, MERCHANT_CATEGORIES, n_total)
    device_type = _sample_categorical(rng, DEVICE_TYPES, n_total)
    transaction_type = _sample_categorical(rng, TRANSACTION_TYPES, n_total)

    days_ago = rng.integers(0, 90, size=n_total)
    minutes = rng.integers(0, 60, size=n_total)
    base_date = pd.Timestamp.utcnow().normalize()
    timestamps = [
        base_date - pd.Timedelta(days=int(d)) + pd.Timedelta(hours=int(h), minutes=int(m))
        for d, h, m in zip(days_ago, hours, minutes)
    ]

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_total + 1),
        "user_id": np.concatenate([user_ids[:n_legit], user_ids[n_legit:]]),
        "card_id": np.concatenate([card_ids[:n_legit], card_ids[n_legit:]]),
        "timestamp": timestamps,
        "amount": amounts,
        "merchant_category": merchant_category,
        "merchant_country": countries,
        "device_type": device_type,
        "transaction_type": transaction_type,
        "is_fraud": is_fraud,
    })

    # Mélange pour ne pas avoir toutes les fraudes à la fin
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["transaction_id"] = np.arange(1, len(df) + 1)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Dataset généré : %d transactions, %d frauduleuses (%.2f%%)",
        len(df), df["is_fraud"].sum(), 100 * df["is_fraud"].mean(),
    )
    return df


def _normalized_weights(n_domestic: int, n_foreign: int, domestic_share: float = 0.85):
    dom_w = domestic_share / n_domestic
    for_w = (1 - domestic_share) / n_foreign
    return [dom_w] * n_domestic + [for_w] * n_foreign


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère le dataset synthétique de fraude.")
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--n-users", type=int, default=1_500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=str(settings.raw_data_path))
    args = parser.parse_args()

    df = generate_transactions(
        n_rows=args.n_rows, fraud_rate=args.fraud_rate,
        n_users=args.n_users, seed=args.seed,
    )
    out_path = settings.project_root / args.output if not args.output.startswith("/") else args.output
    df.to_csv(out_path, index=False)
    logger.info("Fichier écrit : %s", out_path)


if __name__ == "__main__":
    main()

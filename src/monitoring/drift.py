"""
PSI par feature, maison (pas Evidently, overkill pour ce besoin).

À savoir : le split référence/actuel par défaut coupe juste le dataset en
deux dans le temps, donc les features cumulatives ressortent "critiques"
sans que ce soit un vrai drift.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # feature quasi constante, PSI non pertinent

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / max(cur_counts.sum(), 1))

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    categories = set(reference.unique()) | set(current.unique())
    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)

    psi = 0.0
    for cat in categories:
        r = max(ref_pct.get(cat, 1e-4), 1e-4)
        c = max(cur_pct.get(cat, 1e-4), 1e-4)
        psi += (c - r) * np.log(c / r)
    return round(float(psi), 4)


@dataclass
class DriftReport:
    feature_scores: dict[str, float] = field(default_factory=dict)
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def drifted_features(self) -> list[str]:
        return [f for f, psi in self.feature_scores.items() if psi >= PSI_WARNING_THRESHOLD]

    @property
    def critical_features(self) -> list[str]:
        return [f for f, psi in self.feature_scores.items() if psi >= PSI_CRITICAL_THRESHOLD]

    def to_dict(self) -> dict:
        return {
            "run_at": self.run_at,
            "feature_scores": self.feature_scores,
            "drifted_features": self.drifted_features,
            "critical_features": self.critical_features,
            "retrain_recommended": len(self.critical_features) > 0,
        }

    def save(self, path=None) -> None:
        path = path or (settings.reports_dir / "drift_report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Rapport de drift écrit : %s", path)


NUMERIC_FEATURES = [
    "amount", "log_amount", "amount_to_user_avg_ratio",
    "user_txn_count_before", "user_avg_amount_before",
    "card_distinct_countries_so_far", "seconds_since_last_user_txn",
]
CATEGORICAL_RAW_FEATURES = ["merchant_category", "merchant_country", "device_type", "transaction_type"]


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftReport:
    report = DriftReport()

    for col in NUMERIC_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            report.feature_scores[col] = _psi_numeric(reference_df[col], current_df[col])

    for col in CATEGORICAL_RAW_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            report.feature_scores[col] = _psi_categorical(reference_df[col], current_df[col])

    for feature, psi in report.feature_scores.items():
        level = "CRITIQUE" if psi >= PSI_CRITICAL_THRESHOLD else (
            "WARNING" if psi >= PSI_WARNING_THRESHOLD else "OK")
        logger.info("Drift %-32s PSI=%.4f [%s]", feature, psi, level)

    report.save()
    return report


if __name__ == "__main__":
    from src.common.db import read_sql

    features = read_sql("SELECT * FROM features")
    split = len(features) // 2
    reference, current = features.iloc[:split], features.iloc[split:]
    compute_drift(reference, current)

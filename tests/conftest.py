"""
Fixtures pytest partagées.

Isole chaque run de tests dans sa propre base SQLite temporaire, pour ne
jamais toucher à data/processed/fraud.db utilisée pour la démo manuelle.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fraud.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # src.common.config lit les env vars au moment de l'import : on force un
    # rechargement du module pour chaque test afin de prendre en compte la
    # DB temporaire, et on réinitialise le singleton d'engine (src.common.db).
    import importlib

    import src.common.config as config_module
    importlib.reload(config_module)

    import src.common.db as db_module
    db_module._engine = None
    importlib.reload(db_module)

    yield

    db_module._engine = None


@pytest.fixture
def sample_transactions_df():
    from src.ingestion.generate_sample_data import generate_transactions
    return generate_transactions(n_rows=500, fraud_rate=0.05, n_users=80, seed=1)

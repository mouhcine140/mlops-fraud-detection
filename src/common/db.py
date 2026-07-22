"""Accès DB fin, SQLAlchemy Core (pas d'ORM) pour rester proche du SQL."""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.common.config import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Retourne un engine SQLAlchemy unique (singleton) pour tout le process."""
    global _engine
    if _engine is None:
        logger.info("Création de l'engine SQLAlchemy vers %s", settings.database_url)
        _engine = create_engine(settings.database_url, future=True)
    return _engine


def read_sql(query: str, **params) -> pd.DataFrame:
    """Exécute une requête SQL et retourne un DataFrame pandas."""
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params=params or None)


def write_dataframe(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> int:
    """Écrit un DataFrame dans une table. Retourne le nombre de lignes écrites."""
    with get_engine().begin() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    logger.info("Table '%s' écrite : %d lignes", table_name, len(df))
    return len(df)


def table_exists(table_name: str) -> bool:
    from sqlalchemy import inspect

    return inspect(get_engine()).has_table(table_name)

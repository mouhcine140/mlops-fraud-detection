"""
DAG Airflow : ingest -> validate -> build_features -> train -> quality gate
-> drift check.

TaskFlow API (@dag / @task). Chaque tâche appelle une fonction de src/ - la
logique métier reste testable indépendamment d'Airflow (voir tests/).

Tourne dans le conteneur Airflow de docker-compose.yml (Dockerfile.airflow,
qui installe requirements.txt en plus de l'image Airflow officielle).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Permet d'importer les modules du repo (src/) depuis le conteneur Airflow,
# où le repo est monté en volume sous /opt/airflow/project (voir docker-compose.yml).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


@dag(
    dag_id="fraud_detection_pipeline",
    description="Pipeline MLOps de détection de fraude : ingestion -> validation -> features -> training -> monitoring",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "fraud-detection", "portfolio"],
)
def fraud_detection_pipeline():

    @task
    def ingest_task() -> int:
        from src.ingestion.ingest import ingest
        return ingest()

    @task
    def validate_task(n_rows: int) -> bool:
        from src.common.db import read_sql
        from src.validation.data_quality import validate_dataframe

        df = read_sql("SELECT * FROM raw_transactions")
        try:
            report = validate_dataframe(df, raise_on_failure=True)
        except ValueError as exc:
            raise AirflowFailException(f"Data quality gate échoué : {exc}")
        return report.is_valid

    @task
    def build_features_task(is_valid: bool) -> int:
        from src.features.build_features import build_features

        df = build_features(persist=True)
        return len(df)

    @task
    def train_task(n_features_rows: int) -> dict:
        from src.training.train import train
        return train()

    @task
    def check_quality_gate(train_result: dict) -> dict:
        """Empêche la promotion d'un modèle qui régresse par rapport au seuil
        métier minimal (ex: recall trop bas = trop de fraudes non détectées)."""
        metrics = train_result["metrics"]
        min_recall, min_precision = 0.40, 0.30

        if metrics["recall"] < min_recall or metrics["precision"] < min_precision:
            raise AirflowFailException(
                f"Modèle rejeté : recall={metrics['recall']:.3f} (min {min_recall}), "
                f"precision={metrics['precision']:.3f} (min {min_precision})"
            )
        return train_result

    @task
    def drift_check_task(train_result: dict) -> None:
        from src.common.db import read_sql
        from src.monitoring.drift import compute_drift

        features = read_sql("SELECT * FROM features")
        split = max(1, len(features) // 2)
        reference, current = features.iloc[:split], features.iloc[split:]
        report = compute_drift(reference, current)

        if report.critical_features:
            # On log en warning plutôt que de faire échouer le DAG : le drift
            # déclenche un ré-entraînement/une investigation, pas un blocage dur.
            import logging
            logging.getLogger(__name__).warning(
                "Drift critique détecté sur : %s", report.critical_features
            )

    n_rows = ingest_task()
    is_valid = validate_task(n_rows)
    n_features_rows = build_features_task(is_valid)
    train_result = train_task(n_features_rows)
    validated_result = check_quality_gate(train_result)
    drift_check_task(validated_result)


fraud_detection_pipeline()

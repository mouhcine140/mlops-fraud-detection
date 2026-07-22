#!/usr/bin/env bash
# Lance le pipeline complet en local, sans docker/Airflow, avec SQLite.
# Utile pour développer/déboguer rapidement avant de passer par docker-compose.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/5 Génération des données synthétiques =="
python -m src.ingestion.generate_sample_data

echo "== 2/5 Ingestion =="
python -m src.ingestion.ingest

echo "== 3/5 Validation qualité des données =="
python -m src.validation.data_quality

echo "== 4/5 Feature engineering =="
python -m src.features.build_features

echo "== 5/5 Entraînement + tracking MLflow =="
python -m src.training.train

echo "== Drift check (référence vs. batch simulé) =="
python -m src.monitoring.drift

echo ""
echo "Pipeline terminé. Pour lancer l'API : uvicorn src.serving.api:app --reload"
echo "Pour visualiser MLflow : mlflow ui --backend-store-uri sqlite:///mlruns.db"

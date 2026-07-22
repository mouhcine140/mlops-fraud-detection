"""
Config centralisée. Tout est surchargeable via variables d'env / .env
(voir .env.example) pour basculer SQLite (démo) <-> Postgres (docker-compose)
sans toucher au code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Racine du projet = deux niveaux au-dessus de ce fichier (src/common/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    database_url: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'data' / 'processed' / 'fraud.db'}"
    )

    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "fraud_transactions_sample.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    # MLflow 3.x a désactivé le file store par défaut, donc je pointe vers un
    # fichier sqlite pour mlruns au lieu d'un dossier brut.
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlruns.db'}"
    )
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

    model_dir: Path = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
    reports_dir: Path = PROJECT_ROOT / "reports"

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    random_seed: int = 42

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.model_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

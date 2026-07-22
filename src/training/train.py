"""
Entraînement du modèle, tracking MLflow (params, métriques, artifacts) et
enregistrement dans le model registry.

Déséquilibre de classes géré via scale_pos_weight plutôt que du SMOTE - plus
simple et suffisant pour ~1.5% de fraude.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.config import settings
from src.common.db import read_sql
from src.features.build_features import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "is_fraud"

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False
    logger.warning("xgboost non disponible, fallback sur RandomForestClassifier.")


def load_feature_table() -> pd.DataFrame:
    return read_sql("SELECT * FROM features")


def get_full_feature_columns(df: pd.DataFrame) -> list[str]:
    """FEATURE_COLUMNS + toutes les colonnes one-hot générées par
    encode_categoricals (leur nombre dépend des catégories présentes)."""
    onehot_cols = [c for c in df.columns if c not in FEATURE_COLUMNS
                   and c not in ("transaction_id", "user_id", "card_id", "timestamp", TARGET_COLUMN)]
    return FEATURE_COLUMNS + onehot_cols


def build_model(scale_pos_weight: float, seed: int):
    if HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )


def find_best_threshold(y_true, y_proba) -> tuple[float, float]:
    """Choisit le seuil de décision qui maximise le F1-score sur la validation,
    plutôt que le seuil par défaut de 0.5 (peu pertinent avec un fort
    déséquilibre de classes)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1s[:-1])) if len(thresholds) > 0 else 0
    if len(thresholds) == 0:
        return 0.5, 0.0
    return float(thresholds[best_idx]), float(f1s[best_idx])


def train(test_size: float = 0.2, seed: int | None = None) -> dict:
    seed = seed if seed is not None else settings.random_seed
    df = load_feature_table()
    feature_cols = get_full_feature_columns(df)

    X = df[feature_cols].astype(float)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        model = build_model(scale_pos_weight, seed)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        best_threshold, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_at_best_threshold": best_f1,
            "best_threshold": best_threshold,
        }
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        mlflow.log_params({
            "model_type": "XGBClassifier" if HAS_XGBOOST else "RandomForestClassifier",
            "n_train_rows": len(X_train),
            "n_test_rows": len(X_test),
            "n_features": len(feature_cols),
            "scale_pos_weight": round(scale_pos_weight, 2),
            "test_size": test_size,
            "seed": seed,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({"true_positives": tp, "false_positives": fp,
                             "true_negatives": tn, "false_negatives": fn})

        # Flavor MLflow dédié pour XGBoost (sérialisation native), sklearn
        # sinon. mlflow.sklearn.log_model() sur un XGBClassifier échoue en
        # MLflow 3.x : le flavor sklearn passe par skops, qui refuse par
        # défaut les types xgboost.Booster / XGBClassifier ("untrusted types").
        log_model_fn = mlflow.xgboost.log_model if HAS_XGBOOST else mlflow.sklearn.log_model

        try:
            log_model_fn(
                model, artifact_path="model",
                input_example=X_test.head(3),
                registered_model_name="fraud-detection-model",
            )
        except Exception as exc:
            logger.warning("Enregistrement dans le model registry ignoré (%s) ; log simple de l'artifact.", exc)
            log_model_fn(model, artifact_path="model", input_example=X_test.head(3))

        settings.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.model_dir / "model.joblib"
        joblib.dump({"model": model, "feature_columns": feature_cols,
                     "threshold": best_threshold}, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info("Run MLflow : %s", run.info.run_id)
        logger.info("Métriques : %s", metrics)

        return {"run_id": run.info.run_id, "metrics": metrics, "model_path": str(model_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de détection de fraude.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    result = train(test_size=args.test_size, seed=args.seed)
    print(result)


if __name__ == "__main__":
    main()

"""
Data quality checks maison, façon Great Expectations : règles déclaratives,
résultat structuré par check, rapport JSON + décision pass/fail globale.
Pas de dépendance externe pour une dizaine de règles (à réévaluer si le
volume de règles grossit - voir README).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from src.common.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str  # "critical" | "warning"
    details: str


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_valid(self) -> bool:
        """Le dataset est considéré valide s'il n'y a aucun échec CRITIQUE
        (les warnings n'empêchent pas le pipeline de continuer)."""
        return all(c.passed for c in self.checks if c.severity == "critical")

    def to_dict(self) -> dict:
        return {
            "run_at": self.run_at,
            "is_valid": self.is_valid,
            "n_checks": len(self.checks),
            "n_passed": sum(c.passed for c in self.checks),
            "n_failed": sum(not c.passed for c in self.checks),
            "checks": [c.__dict__ for c in self.checks],
        }

    def save(self, path=None) -> None:
        path = path or (settings.reports_dir / "data_quality_report.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("Rapport de validation écrit : %s", path)


class DataQualityChecker:
    """Applique une série de règles de qualité sur un DataFrame de transactions."""

    REQUIRED_COLUMNS = [
        "transaction_id", "user_id", "card_id", "timestamp", "amount",
        "merchant_category", "merchant_country", "device_type",
        "transaction_type", "is_fraud",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = ValidationReport()

    def _check(self, name: str, condition: bool, severity: str, details: str) -> None:
        self.report.checks.append(CheckResult(name, bool(condition), severity, details))
        level = logging.INFO if condition else (
            logging.ERROR if severity == "critical" else logging.WARNING
        )
        logger.log(level, "[%s] %s -> %s (%s)", severity.upper(), name,
                    "OK" if condition else "FAIL", details)

    def check_schema(self) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        self._check(
            "schema_required_columns", len(missing) == 0, "critical",
            f"Colonnes manquantes : {missing}" if missing else "Toutes les colonnes attendues sont présentes",
        )

    def check_not_empty(self) -> None:
        self._check(
            "dataset_not_empty", len(self.df) > 0, "critical",
            f"{len(self.df)} lignes",
        )

    def check_primary_key_unique(self) -> None:
        if "transaction_id" not in self.df.columns:
            return
        n_dup = self.df["transaction_id"].duplicated().sum()
        self._check(
            "transaction_id_unique", n_dup == 0, "critical",
            f"{n_dup} doublons de transaction_id",
        )

    def check_no_nulls_in_critical_columns(self) -> None:
        critical_cols = ["transaction_id", "user_id", "amount", "timestamp", "is_fraud"]
        for col in critical_cols:
            if col not in self.df.columns:
                continue
            n_null = self.df[col].isna().sum()
            self._check(
                f"no_nulls__{col}", n_null == 0, "critical",
                f"{n_null} valeurs nulles",
            )

    def check_null_rate_warning(self, max_rate: float = 0.05) -> None:
        for col in self.df.columns:
            rate = self.df[col].isna().mean()
            if rate > 0:
                self._check(
                    f"null_rate__{col}", rate <= max_rate, "warning",
                    f"{rate:.2%} de valeurs nulles (seuil {max_rate:.0%})",
                )

    def check_amount_positive(self) -> None:
        if "amount" not in self.df.columns:
            return
        n_invalid = (self.df["amount"] <= 0).sum()
        self._check(
            "amount_strictly_positive", n_invalid == 0, "critical",
            f"{n_invalid} transactions avec montant <= 0",
        )

    def check_amount_reasonable_range(self, max_amount: float = 50_000) -> None:
        if "amount" not in self.df.columns:
            return
        n_extreme = (self.df["amount"] > max_amount).sum()
        self._check(
            "amount_below_extreme_threshold", n_extreme / max(len(self.df), 1) < 0.01, "warning",
            f"{n_extreme} transactions au-dessus de {max_amount}",
        )

    def check_target_binary(self) -> None:
        if "is_fraud" not in self.df.columns:
            return
        unique_values = [int(v) for v in self.df["is_fraud"].dropna().unique()]
        valid_values = set(unique_values) <= {0, 1}
        self._check(
            "is_fraud_binary", valid_values, "critical",
            f"Valeurs uniques trouvées : {sorted(unique_values)}",
        )

    def check_class_balance_sane(self, min_rate: float = 0.0005, max_rate: float = 0.3) -> None:
        if "is_fraud" not in self.df.columns or len(self.df) == 0:
            return
        rate = self.df["is_fraud"].mean()
        self._check(
            "fraud_rate_within_sane_bounds", min_rate <= rate <= max_rate, "warning",
            f"Taux de fraude observé : {rate:.4%}",
        )

    def check_categorical_values_known(self) -> None:
        known = {
            "device_type": {"mobile_app", "web", "pos_chip", "pos_nfc", "atm"},
            "transaction_type": {"purchase", "withdrawal", "online_payment", "transfer"},
        }
        for col, allowed in known.items():
            if col not in self.df.columns:
                continue
            unknown_values = set(self.df[col].dropna().unique()) - allowed
            self._check(
                f"known_categories__{col}", len(unknown_values) == 0, "warning",
                f"Valeurs inconnues : {unknown_values}" if unknown_values else "OK",
            )

    def run_all(self) -> ValidationReport:
        checks: list[Callable[[], None]] = [
            self.check_schema,
            self.check_not_empty,
            self.check_primary_key_unique,
            self.check_no_nulls_in_critical_columns,
            self.check_null_rate_warning,
            self.check_amount_positive,
            self.check_amount_reasonable_range,
            self.check_target_binary,
            self.check_class_balance_sane,
            self.check_categorical_values_known,
        ]
        for check_fn in checks:
            check_fn()
        return self.report


def validate_dataframe(df: pd.DataFrame, raise_on_failure: bool = True) -> ValidationReport:
    checker = DataQualityChecker(df)
    report = checker.run_all()
    report.save()

    if not report.is_valid and raise_on_failure:
        failed = [c.name for c in report.checks if not c.passed and c.severity == "critical"]
        raise ValueError(f"Validation des données échouée (checks critiques) : {failed}")

    return report


if __name__ == "__main__":
    from src.common.db import read_sql

    data = read_sql("SELECT * FROM raw_transactions")
    validate_dataframe(data)

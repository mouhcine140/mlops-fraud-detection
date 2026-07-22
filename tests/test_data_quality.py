from __future__ import annotations

import pandas as pd
import pytest

from src.validation.data_quality import DataQualityChecker, validate_dataframe


def test_valid_dataset_passes(sample_transactions_df):
    checker = DataQualityChecker(sample_transactions_df)
    report = checker.run_all()
    assert report.is_valid


def test_missing_required_column_fails_critically(sample_transactions_df):
    df = sample_transactions_df.drop(columns=["amount"])
    checker = DataQualityChecker(df)
    report = checker.run_all()
    assert not report.is_valid
    assert any(c.name == "schema_required_columns" and not c.passed for c in report.checks)


def test_duplicate_transaction_id_fails_critically(sample_transactions_df):
    df = sample_transactions_df.copy()
    df.loc[1, "transaction_id"] = df.loc[0, "transaction_id"]
    checker = DataQualityChecker(df)
    report = checker.run_all()
    assert not report.is_valid


def test_negative_amount_fails_critically(sample_transactions_df):
    df = sample_transactions_df.copy()
    df.loc[0, "amount"] = -50.0
    checker = DataQualityChecker(df)
    report = checker.run_all()
    failed_critical = [c.name for c in report.checks if not c.passed and c.severity == "critical"]
    assert "amount_strictly_positive" in failed_critical


def test_validate_dataframe_raises_on_critical_failure(sample_transactions_df):
    df = sample_transactions_df.drop(columns=["is_fraud"])
    with pytest.raises(ValueError):
        validate_dataframe(df, raise_on_failure=True)


def test_validate_dataframe_does_not_raise_when_disabled(sample_transactions_df):
    df = sample_transactions_df.drop(columns=["is_fraud"])
    report = validate_dataframe(df, raise_on_failure=False)
    assert not report.is_valid

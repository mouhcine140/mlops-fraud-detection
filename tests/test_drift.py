from __future__ import annotations

import numpy as np
import pandas as pd

from src.monitoring.drift import _psi_categorical, _psi_numeric, compute_drift


def test_psi_numeric_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(size=2000))
    current = pd.Series(rng.normal(size=2000))
    psi = _psi_numeric(reference, current)
    assert psi < 0.05


def test_psi_numeric_is_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(loc=0, size=2000))
    current = pd.Series(rng.normal(loc=5, size=2000))  # gros décalage
    psi = _psi_numeric(reference, current)
    assert psi > 0.25


def test_psi_categorical_is_near_zero_for_same_proportions():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 500 + ["b"] * 500)
    assert _psi_categorical(reference, current) < 0.01


def test_psi_categorical_detects_new_dominant_category():
    reference = pd.Series(["a"] * 900 + ["b"] * 100)
    current = pd.Series(["a"] * 100 + ["b"] * 900)
    assert _psi_categorical(reference, current) > 0.25


def test_compute_drift_flags_critical_features():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({
        "amount": rng.normal(loc=50, size=1000),
        "merchant_category": rng.choice(["grocery", "travel"], size=1000),
    })
    current = pd.DataFrame({
        "amount": rng.normal(loc=500, size=1000),  # drift fort
        "merchant_category": rng.choice(["grocery", "travel"], size=1000),
    })
    report = compute_drift(reference, current)
    assert "amount" in report.critical_features

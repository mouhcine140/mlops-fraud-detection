from __future__ import annotations

import pandas as pd

from src.features.build_features import (
    add_amount_features, add_time_features, add_velocity_features, encode_categoricals,
)


def _base_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-13 02:00:00", "2026-07-13 14:30:00"]),
        "amount": [100.0, 250.0],
        "user_avg_amount_before": [50.0, None],
        "user_txn_count_before": [3, None],
        "seconds_since_last_user_txn": [30.0, None],
        "merchant_category": ["grocery", "travel"],
        "merchant_country": ["FR", "RU"],
        "device_type": ["mobile_app", "web"],
        "transaction_type": ["purchase", "online_payment"],
    })


def test_time_features_flag_night_transactions():
    df = add_time_features(_base_df())
    assert df.loc[0, "is_night_txn"] == 1  # 02:00
    assert df.loc[1, "is_night_txn"] == 0  # 14:30


def test_amount_ratio_uses_user_history():
    df = add_amount_features(_base_df())
    assert df.loc[0, "amount_to_user_avg_ratio"] == 2.0  # 100 / 50


def test_amount_ratio_defaults_to_one_without_history():
    df = _base_df()
    df["user_avg_amount_before"] = [None, None]
    df = add_amount_features(df)
    assert (df["amount_to_user_avg_ratio"] == 1.0).all()


def test_velocity_features_flag_rapid_succession_and_first_txn():
    df = add_velocity_features(_base_df())
    assert df.loc[0, "is_rapid_succession"] == 1  # 30s < 60s
    assert df.loc[1, "is_first_transaction"] == 1  # user_txn_count_before était NaN -> 0


def test_encode_categoricals_produces_onehot_columns():
    df = encode_categoricals(_base_df())
    assert "merchant_category_grocery" in df.columns
    assert "device_type_web" in df.columns
    assert "merchant_category" not in df.columns


def test_encode_categoricals_aligns_to_reference_schema():
    reference = encode_categoricals(_base_df())
    new_row = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-14 09:00:00"]),
        "amount": [42.0],
        "user_avg_amount_before": [42.0],
        "user_txn_count_before": [1],
        "seconds_since_last_user_txn": [500.0],
        "merchant_category": ["gas_station"],  # catégorie absente de la référence
        "merchant_country": ["FR"],
        "device_type": ["mobile_app"],
        "transaction_type": ["purchase"],
    })
    aligned = encode_categoricals(new_row, reference_df=reference)
    assert set(c for c in reference.columns if c.startswith("merchant_category_")) <= set(aligned.columns)
    assert "merchant_category_gas_station" not in aligned.columns

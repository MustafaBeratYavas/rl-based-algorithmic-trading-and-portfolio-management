"""Validate processed dataset loading and tensor alignment for environments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.envs.data_provider import EnvironmentDataProvider

FEATURES = [
    "Norm_RSI",
    "Norm_MACD",
    "Norm_MACD_Signal",
    "Norm_SMA_20",
    "Norm_SMA_50",
    "Norm_Log_Return",
]


def make_provider_config(dataset_path, **overrides):
    """Return the shared provider config with per-test overrides applied last."""
    config = {
        "data_path": str(dataset_path),
        "lookback_window": 5,
        "price_column": "Close",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "features": FEATURES,
    }
    config.update(overrides)
    return config


def test_data_provider_loads_feature_and_price_matrices(synthetic_processed_dataset) -> None:
    dataset = EnvironmentDataProvider(make_provider_config(synthetic_processed_dataset)).load()

    assert dataset.data_matrix.shape == (45, 2, 6)
    assert dataset.close_prices.shape == (45, 2)
    assert dataset.tickers == ["AAA", "BBB"]
    assert len(dataset.dates) == 45


def test_data_provider_enforces_configured_ticker_universe(synthetic_processed_dataset) -> None:
    config = make_provider_config(synthetic_processed_dataset, tickers=["AAA", "MISSING"])

    with pytest.raises(ValueError, match="missing configured tickers"):
        EnvironmentDataProvider(config).load()


def test_data_provider_can_filter_to_configured_ticker_subset(synthetic_processed_dataset) -> None:
    config = make_provider_config(synthetic_processed_dataset, tickers=["BBB"])

    dataset = EnvironmentDataProvider(config).load()

    assert dataset.tickers == ["BBB"]
    assert dataset.data_matrix.shape == (45, 1, 6)


def test_data_provider_rejects_missing_feature_columns(synthetic_processed_dataset) -> None:
    config = make_provider_config(synthetic_processed_dataset, features=[*FEATURES, "Missing"])

    with pytest.raises(ValueError, match="missing columns"):
        EnvironmentDataProvider(config).load()


def test_data_provider_rejects_empty_feature_contract(synthetic_processed_dataset) -> None:
    config = make_provider_config(synthetic_processed_dataset, features=[])

    with pytest.raises(ValueError, match="at least one feature"):
        EnvironmentDataProvider(config).load()


def test_data_provider_rejects_price_column_as_feature(synthetic_processed_dataset) -> None:
    config = make_provider_config(synthetic_processed_dataset, features=[*FEATURES, "Close"])

    with pytest.raises(ValueError, match="must not also appear in features"):
        EnvironmentDataProvider(config).load()


def test_data_provider_reports_dataset_range_for_empty_date_filter(
    synthetic_processed_dataset,
) -> None:
    config = make_provider_config(
        synthetic_processed_dataset,
        start_date="2030-01-01",
        end_date="2030-12-31",
    )

    with pytest.raises(ValueError, match="Dataset covers"):
        EnvironmentDataProvider(config).load()


def test_data_provider_rejects_unshared_ticker_dates(tmp_path) -> None:
    # Construct disjoint calendars to prove the provider refuses unaligned panels.
    records = []
    for ticker, dates in [
        ("AAA", pd.date_range("2024-01-01", periods=6)),
        ("BBB", pd.date_range("2024-02-01", periods=6)),
    ]:
        for idx, date in enumerate(dates):
            records.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": 100.0 + idx,
                    **{feature: float(idx) for feature in FEATURES},
                }
            )
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(records).set_index("Date").to_parquet(path)

    with pytest.raises(ValueError, match="No common trading dates"):
        EnvironmentDataProvider(make_provider_config(path)).load()


def test_data_provider_drops_invalid_rows_before_returning_dataset(tmp_path) -> None:
    # Mix invalid features and prices into both tickers to verify row-level filtering.
    records = []
    dates = pd.date_range("2024-01-01", periods=10)
    for ticker in ["AAA", "BBB"]:
        for idx, date in enumerate(dates):
            close = 100.0 + idx
            records.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Close": 0.0 if idx == 5 else close,
                    **{feature: np.nan if idx == 0 else float(idx) for feature in FEATURES},
                }
            )
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(records).set_index("Date").to_parquet(path)

    dataset = EnvironmentDataProvider(make_provider_config(path, lookback_window=2)).load()

    assert len(dataset.dates) == 8
    assert np.isfinite(dataset.data_matrix).all()
    assert (dataset.close_prices > 0).all()


def test_data_provider_ignores_missing_invalid_and_non_mapping_metadata(tmp_path) -> None:
    path = tmp_path / "dataset.parquet"

    assert EnvironmentDataProvider._load_metadata(path) == {}

    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text("{bad json", encoding="utf-8")
    assert EnvironmentDataProvider._load_metadata(path) == {}

    metadata_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert EnvironmentDataProvider._load_metadata(path) == {}


def test_data_provider_loads_valid_metadata(tmp_path) -> None:
    path = tmp_path / "dataset.parquet"
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata_path.write_text('{"start_date": "2024-01-01"}', encoding="utf-8")

    assert EnvironmentDataProvider._load_metadata(path) == {"start_date": "2024-01-01"}

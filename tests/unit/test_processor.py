"""Validate data processor cleaning, indicator, split, and normalization contracts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.processor import DataProcessor


def _write_raw_prices(raw_path, ticker: str, rows: int = 10) -> None:
    # Build compact OHLCV CSV fixtures that still satisfy short indicator warm-up windows.
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.full(rows, 1_000),
        },
        index=dates,
    )
    frame.to_csv(raw_path / f"{ticker}.csv")


def _fast_indicator_config(raw_path, processed_path, tickers: list[str]) -> dict:
    # Short indicator windows keep end-to-end processor tests fast and deterministic.
    return {
        "tickers": tickers,
        "paths": {"raw_data": raw_path, "processed_data": processed_path},
        "indicators": {
            "rsi_period": 2,
            "macd_fast": 2,
            "macd_slow": 3,
            "macd_signal": 2,
            "sma_periods": [2],
        },
        "normalization": {"enabled": True},
        "splits": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }


def test_load_raw_csv_coerces_numeric_columns_and_sorts_dates(tmp_path) -> None:
    csv_path = tmp_path / "raw.csv"
    frame = pd.DataFrame(
        {
            "Open": ["bad", "10.0", None],
            "Ticker": ["AAA", "AAA", "AAA"],
        },
        index=["2024-01-03", "2024-01-01", "2024-01-02"],
    )
    frame.to_csv(csv_path)
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})

    result = processor._load_raw_csv(csv_path)

    assert result.index.tolist() == sorted(result.index.tolist())
    assert np.isnan(result.loc[pd.Timestamp("2024-01-03"), "Open"])
    assert result.loc[pd.Timestamp("2024-01-01"), "Ticker"] == "AAA"


def test_load_raw_csv_rejects_non_datetime_index(tmp_path) -> None:
    csv_path = tmp_path / "raw.csv"
    pd.DataFrame({"Close": [100.0]}, index=["not-a-date"]).to_csv(csv_path)
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})

    with pytest.raises(ValueError, match="DatetimeIndex"):
        processor._load_raw_csv(csv_path)


def test_clean_data_does_not_backfill_future_values(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame(
        {"Close": [np.nan, 10.0, np.nan]}, index=pd.date_range("2024-01-01", periods=3)
    )

    cleaned = processor._clean_data(df)

    assert cleaned.index[0] == pd.Timestamp("2024-01-02")
    assert cleaned.iloc[-1]["Close"] == 10.0


def test_rsi_handles_zero_loss_without_infinite_values(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    prices = pd.Series(np.arange(1.0, 30.0))

    rsi = processor._calculate_rsi(prices, period=14).dropna()

    assert np.isfinite(rsi).all()
    assert (rsi <= 100).all()


def test_validate_raw_data_rejects_missing_columns(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))

    with pytest.raises(ValueError, match="missing required columns"):
        processor._validate_raw_data(df, "TEST")


def test_validate_raw_data_rejects_missing_configured_price_column(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "price_column": "Adj Close",
        }
    )
    df = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5], "Volume": [100]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    with pytest.raises(ValueError, match="price_column"):
        processor._validate_raw_data(df, "TEST")


def test_validate_raw_data_rejects_empty_dataframe(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([]),
    )

    with pytest.raises(ValueError, match="has no rows"):
        processor._validate_raw_data(df, "TEST")


def test_validate_raw_data_rejects_non_positive_prices(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame(
        {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [-5.0], "Volume": [100]},
        index=pd.date_range("2024-01-01", periods=1),
    )

    with pytest.raises(ValueError, match="non-positive prices"):
        processor._validate_raw_data(df, "TEST")


def test_drop_invalid_feature_rows_rejects_empty_result() -> None:
    df = pd.DataFrame({"feature": [np.nan, np.inf]})

    with pytest.raises(ValueError, match="No valid rows"):
        DataProcessor._drop_invalid_feature_rows(df)


def test_add_indicators_uses_configured_windows(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "indicators": {
                "rsi_period": 2,
                "macd_fast": 2,
                "macd_slow": 3,
                "macd_signal": 2,
                "sma_periods": [2],
            },
        }
    )
    prices = np.linspace(100.0, 110.0, 8)
    df = pd.DataFrame({"Close": prices}, index=pd.date_range("2024-01-01", periods=8))

    result = processor._add_indicators(df)

    assert {"RSI", "MACD", "MACD_Signal", "SMA_2", "Log_Return"}.issubset(result.columns)
    assert len(result) > 0


def test_split_ratios_must_sum_to_one(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "splits": {"train": 0.5, "validation": 0.2, "test": 0.1},
        }
    )
    df = pd.DataFrame({"A": range(10)}, index=pd.date_range("2024-01-01", periods=10))

    with pytest.raises(ValueError, match="sum to 1.0"):
        processor._build_chronological_splits(df)


def test_split_config_must_be_explicit_when_overridden(tmp_path) -> None:
    with pytest.raises(ValueError, match="train, validation, and test"):
        DataProcessor(
            {
                "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
                "splits": {"train": 0.8, "validation": 0.2},
            }
        )


def test_chronological_splits_require_at_least_three_dates(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame({"A": [1, 2]}, index=pd.date_range("2024-01-01", periods=2))

    with pytest.raises(ValueError, match="at least three"):
        processor._build_chronological_splits(df)


def test_chronological_splits_preserve_temporal_order(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame({"A": range(20)}, index=pd.date_range("2024-01-01", periods=20))

    splits = processor._build_chronological_splits(df)

    assert splits["train"].max() < splits["validation"].min()
    assert splits["validation"].max() < splits["test"].min()


def test_macd_returns_two_finite_series_with_matching_length(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    prices = pd.Series(np.linspace(90.0, 110.0, num=50))

    macd, signal = processor._calculate_macd(prices, fast=12, slow=26, signal=9)

    assert len(macd) == len(prices)
    assert len(signal) == len(prices)
    assert np.isfinite(macd.dropna()).all()
    assert np.isfinite(signal.dropna()).all()


def test_macd_waits_for_minimum_history_before_emitting_values(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    prices = pd.Series(np.linspace(90.0, 110.0, num=50))

    macd, signal = processor._calculate_macd(prices, fast=12, slow=26, signal=9)

    assert macd.iloc[:25].isna().all()
    assert signal.iloc[:33].isna().all()


def test_inferred_feature_columns_exclude_raw_prices_and_normalized_columns(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
            "Ticker": ["AAA"],
            "RSI": [50.0],
            "Norm_RSI": [0.0],
            "MACD": [0.1],
        }
    )

    assert processor._infer_feature_columns(df) == ["RSI", "MACD"]


def test_inferred_feature_columns_exclude_configured_adjusted_price(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "price_column": "Adj Close",
        }
    )
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.4],
            "Volume": [100],
            "Ticker": ["AAA"],
            "RSI": [50.0],
        }
    )

    assert processor._infer_feature_columns(df) == ["RSI"]


def test_normalization_refuses_raw_price_column(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "normalization": {"enabled": True, "feature_columns": ["Close"]},
        }
    )
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame({"Close": np.arange(5, dtype=float), "Ticker": "AAA"}, index=dates)

    with pytest.raises(ValueError, match="raw price_column"):
        processor._add_normalized_features(df, dates[:3])


def test_normalization_returns_original_frame_when_disabled(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame({"RSI": np.arange(5, dtype=float)}, index=dates)

    assert processor._add_normalized_features(df, dates[:3]) is df


def test_normalization_rejects_missing_feature_columns(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "normalization": {"enabled": True, "feature_columns": ["Missing"]},
        }
    )
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame({"RSI": np.arange(5, dtype=float)}, index=dates)

    with pytest.raises(ValueError, match="missing feature columns"):
        processor._add_normalized_features(df, dates[:3])


def test_normalization_rejects_empty_training_window(tmp_path) -> None:
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "normalization": {"enabled": True, "feature_columns": ["RSI"]},
        }
    )
    dates = pd.date_range("2024-01-01", periods=5)
    df = pd.DataFrame({"RSI": np.arange(5, dtype=float)}, index=dates)

    with pytest.raises(ValueError, match="training split is empty"):
        processor._add_normalized_features(df, pd.DatetimeIndex([]))


def test_normalization_fits_scaler_on_training_window_only(tmp_path) -> None:
    # Add large validation/test values to prove scaling parameters come only from training.
    processor = DataProcessor(
        {
            "paths": {"raw_data": tmp_path, "processed_data": tmp_path},
            "normalization": {"enabled": True, "feature_columns": ["RSI"]},
        }
    )
    dates = pd.date_range("2024-01-01", periods=8)
    df = pd.DataFrame(
        {
            "RSI": [10.0, 20.0, 30.0, 40.0, 1000.0, 1001.0, 1002.0, 1003.0],
            "Ticker": ["AAA"] * 8,
        },
        index=dates,
    )

    result = processor._add_normalized_features(df, dates[:4])

    assert np.isclose(result.loc[dates[:4], "Norm_RSI"].mean(), 0.0)
    assert result.loc[dates[4], "Norm_RSI"] > 80.0


def test_inferred_feature_columns_rejects_frames_without_engineered_features(tmp_path) -> None:
    processor = DataProcessor({"paths": {"raw_data": tmp_path, "processed_data": tmp_path}})
    df = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Volume": [100],
            "Ticker": ["AAA"],
            "Norm_RSI": [0.0],
        }
    )

    with pytest.raises(ValueError, match="No engineered feature columns"):
        processor._infer_feature_columns(df)


def test_process_all_raises_when_no_ticker_data_exists(tmp_path) -> None:
    processor = DataProcessor(
        {
            "tickers": ["MISSING"],
            "paths": {"raw_data": tmp_path / "raw", "processed_data": tmp_path / "processed"},
        }
    )

    with pytest.raises(RuntimeError, match="No data processed"):
        processor.process_all()


def test_process_all_rejects_skipped_required_tickers(tmp_path) -> None:
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    raw_path.mkdir()
    _write_raw_prices(raw_path, "AAA")
    config = _fast_indicator_config(raw_path, processed_path, ["AAA", "MISSING"])
    processor = DataProcessor(config)

    with pytest.raises(RuntimeError, match="skipped required tickers"):
        processor.process_all()


def test_process_all_writes_split_files_and_metadata_for_partial_runs(tmp_path) -> None:
    raw_path = tmp_path / "raw"
    processed_path = tmp_path / "processed"
    raw_path.mkdir()
    _write_raw_prices(raw_path, "AAA")
    config = _fast_indicator_config(raw_path, processed_path, ["AAA", "MISSING"])
    config["processing"] = {"allow_partial": True}

    DataProcessor(config).process_all()

    assert (processed_path / "processed_dataset.parquet").exists()
    assert (processed_path / "train_dataset.parquet").exists()
    assert (processed_path / "validation_dataset.parquet").exists()
    assert (processed_path / "test_dataset.parquet").exists()
    assert (processed_path / "feature_scaler.joblib").exists()

    metadata = json.loads((processed_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["tickers"] == ["AAA"]
    assert metadata["skipped_tickers"] == ["MISSING"]
    assert metadata["sha256"]

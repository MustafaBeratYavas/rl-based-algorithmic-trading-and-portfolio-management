"""Validate Yahoo Finance downloader quality gates and retry behavior."""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src.data.downloader import YFinanceDownloader
from src.utils.exceptions import DownloadError


def make_downloader_config(tmp_path, **download_overrides):
    # Keep downloader tests fast by disabling retry delay unless a test overrides it.
    download_config = {
        "retries": 2,
        "backoff_seconds": 0.0,
        "minimum_rows": 5,
        "max_nan_ratio": 0.20,
        "strict": True,
        **download_overrides,
    }
    return {
        "tickers": ["TEST"],
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "paths": {"raw_data": tmp_path},
        "download": download_config,
    }


def valid_download_frame(rows: int = 6) -> pd.DataFrame:
    # Provide a reusable OHLCV frame that satisfies the downloader contract.
    return pd.DataFrame(
        {
            "Open": [10.0] * rows,
            "High": [11.0] * rows,
            "Low": [9.0] * rows,
            "Close": [10.5] * rows,
            "Volume": [1000] * rows,
        },
        index=pd.date_range("2024-01-01", periods=rows),
    )


def test_fetch_data_raises_when_no_tickers_configured(tmp_path) -> None:
    downloader = YFinanceDownloader({"tickers": [], "paths": {"raw_data": tmp_path}})

    with pytest.raises(ValueError, match="No tickers configured"):
        downloader.fetch_data()


def test_validate_download_rejects_empty_dataframe(tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path))

    with pytest.raises(ValueError, match="No data retrieved"):
        downloader._validate_download(pd.DataFrame(), "TEST")


def test_validate_download_rejects_missing_required_columns(tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path))
    df = pd.DataFrame({"Close": [1.0] * 5})

    with pytest.raises(ValueError, match="missing required columns"):
        downloader._validate_download(df, "TEST")


def test_validate_download_rejects_insufficient_rows(tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path))

    with pytest.raises(ValueError, match="minimum"):
        downloader._validate_download(valid_download_frame(rows=1), "TEST")


def test_validate_download_rejects_excessive_nan_ratio(tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path, max_nan_ratio=0.10))
    df = valid_download_frame(rows=10)
    df.loc[df.index[:3], "Close"] = None

    with pytest.raises(ValueError, match="NaN ratio"):
        downloader._validate_download(df, "TEST")


def test_validate_download_rejects_non_positive_close_prices(tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path))
    df = valid_download_frame()
    df.loc[df.index[0], "Close"] = 0.0

    with pytest.raises(ValueError, match="non-positive close prices"):
        downloader._validate_download(df, "TEST")


def test_download_ticker_writes_valid_csv(monkeypatch, tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path))
    seen_kwargs = {}

    def fake_download(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return valid_download_frame()

    monkeypatch.setattr("src.data.downloader.yf.download", fake_download)

    assert downloader._download_ticker("TEST") is True
    assert (tmp_path / "TEST.csv").exists()
    assert seen_kwargs["auto_adjust"] is True


def test_download_ticker_rejects_duplicate_columns_after_flattening(monkeypatch, tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path, retries=1))
    # Simulate a batched yfinance response that becomes ambiguous after flattening.
    frame = pd.DataFrame(
        [[10.0, 10.1, 11.0, 9.0, 10.5, 1000] for _ in range(6)],
        index=pd.date_range("2024-01-01", periods=6),
    )
    frame.columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAA"),
            ("Open", "BBB"),
            ("High", "AAA"),
            ("Low", "AAA"),
            ("Close", "AAA"),
            ("Volume", "AAA"),
        ]
    )
    monkeypatch.setattr("src.data.downloader.yf.download", lambda *args, **kwargs: frame)

    assert downloader._download_ticker("TEST") is False


def test_download_ticker_retries_request_exceptions(monkeypatch, tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path, retries=1))
    monkeypatch.setattr(
        "src.data.downloader.yf.download",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.ReadTimeout("slow")),
    )

    assert downloader._download_ticker("TEST") is False


def test_fetch_data_raises_download_error_in_strict_mode(monkeypatch, tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path, strict=True))
    monkeypatch.setattr(downloader, "_download_ticker", lambda ticker: False)

    with pytest.raises(DownloadError, match="TEST"):
        downloader.fetch_data()


def test_fetch_data_continues_when_strict_mode_is_disabled(monkeypatch, tmp_path) -> None:
    downloader = YFinanceDownloader(make_downloader_config(tmp_path, strict=False))
    monkeypatch.setattr(downloader, "_download_ticker", lambda ticker: False)

    downloader.fetch_data()

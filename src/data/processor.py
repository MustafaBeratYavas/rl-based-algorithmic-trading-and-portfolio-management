"""Build model-ready datasets from raw OHLCV files.

DataProcessor owns the integrity boundary between vendor snapshots and model
training artifacts. It validates raw inputs, derives technical features, applies
train-only normalization, writes split-specific parquet files, and records
lineage metadata for reproducible experiments.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import ensure_directory, resolve_project_path


class DataProcessor:
    """Convert per-ticker raw CSV files into reproducible modeling artifacts.

    The processor owns all data-integrity decisions after download: numeric
    coercion, indicator generation, chronological splitting, train-only feature
    scaling, split parquet writes, and metadata checksums. It never backfills
    prices or fits transforms outside the training window.
    """

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
    DEFAULT_SPLITS = {"train": 0.7, "validation": 0.15, "test": 0.15}
    DEFAULT_NORMALIZATION = {"enabled": False}
    DEFAULT_PROCESSING = {"allow_partial": False}

    def __init__(self, config: dict[str, Any]):
        """Bind processing paths, feature settings, and partial-run policy from config."""
        self.config = config
        self.logger = get_logger(__name__)
        self.raw_path = resolve_project_path(config.get("paths", {}).get("raw_data", "data/raw"))
        self.processed_path = ensure_directory(
            config.get("paths", {}).get("processed_data", "data/processed")
        )
        self.tickers = config.get("tickers", [])
        self.indicators = config.get("indicators", {})
        self.price_column = config.get("price_column", "Close")
        self.split_config = self._load_split_config(config.get("splits"))
        self.normalization_config = {
            **self.DEFAULT_NORMALIZATION,
            **config.get("normalization", {}),
        }
        self.processing_config = {**self.DEFAULT_PROCESSING, **config.get("processing", {})}

    def process_all(self) -> None:
        """Build the processed dataset, split artifacts, scaler, and lineage metadata.

        Raises:
            ValueError: If required configuration or feature data is invalid.
            RuntimeError: If no ticker can be processed or required tickers are skipped.
        """
        # Fail fast on empty universes; a successful run must always produce real assets.
        if not self.tickers:
            raise ValueError("No tickers configured for data processing.")

        self.logger.info("Starting data processing.")
        processed_dfs: list[pd.DataFrame] = []
        skipped_tickers: list[str] = []

        for ticker in self.tickers:
            file_path = self.raw_path / f"{ticker}.csv"
            if not file_path.exists():
                self.logger.warning("Raw data file not found for %s. Skipping.", ticker)
                skipped_tickers.append(ticker)
                continue

            try:
                # Validate each symbol independently before it can enter the shared dataset.
                df = self._load_raw_csv(file_path)
                self._validate_raw_data(df, ticker)
                df = self._clean_data(df)
                df = self._add_indicators(df)
                df["Ticker"] = ticker
                processed_dfs.append(df)
                self.logger.info("Processed features for %s", ticker)
            except (KeyError, ValueError, pd.errors.ParserError) as exc:
                self.logger.exception("Error processing %s: %s", ticker, exc)
                skipped_tickers.append(ticker)

        if not processed_dfs:
            raise RuntimeError("No data processed. Check raw data files and configuration.")
        if skipped_tickers and not self.processing_config["allow_partial"]:
            skipped = ", ".join(sorted(skipped_tickers))
            raise RuntimeError(
                "Data processing skipped required tickers: "
                f"{skipped}. Set processing.allow_partial=true to permit this explicitly."
            )

        # Build one chronological panel so split boundaries are shared across all symbols.
        final_df = pd.concat(processed_dfs, sort=False).sort_index()
        final_df = self._drop_invalid_feature_rows(final_df)

        split_dates = self._build_chronological_splits(final_df)
        final_df = self._add_normalized_features(final_df, split_dates["train"])
        final_df = self._drop_invalid_feature_rows(final_df)

        output_file = self.processed_path / "processed_dataset.parquet"
        final_df.to_parquet(output_file)
        self._write_split_files(final_df, split_dates)
        self._write_metadata(final_df, output_file, skipped_tickers)
        self.logger.info("Saved processed dataset to %s", output_file)

    def _load_raw_csv(self, file_path: str | Path) -> pd.DataFrame:
        """Load one raw CSV as a date-indexed, numerically coerced time series."""
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Expected a DatetimeIndex in {file_path}")
        # Coerce vendor fields at the ingestion edge so downstream logic receives numerics.
        numeric_columns = [column for column in df.columns if column != "Ticker"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        return df.dropna(how="all").sort_index()

    def _validate_raw_data(self, df: pd.DataFrame, ticker: str) -> None:
        """Enforce the minimum price contract required for feature engineering."""
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(f"{ticker} is missing required columns: {sorted(missing)}")
        if self.price_column not in df.columns:
            raise ValueError(f"{ticker} is missing configured price_column={self.price_column!r}")
        if df.empty:
            raise ValueError(f"{ticker} has no rows after loading.")
        if (df[self.price_column] <= 0).any():
            raise ValueError(f"{ticker} contains non-positive prices in {self.price_column}.")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid values without introducing look-ahead leakage."""
        # Forward-fill only; backward-fill would leak future information into earlier rows.
        cleaned = df.replace([np.inf, -np.inf], np.nan).ffill()
        return cleaned.dropna()

    @classmethod
    def _load_split_config(cls, split_config: dict[str, Any] | None) -> dict[str, float]:
        """Return train/validation/test ratios, requiring all keys when overridden."""
        if split_config is None:
            return {key: float(value) for key, value in cls.DEFAULT_SPLITS.items()}

        required_keys = set(cls.DEFAULT_SPLITS)
        missing_keys = sorted(required_keys.difference(split_config))
        if missing_keys:
            raise ValueError(
                "Data splits must define train, validation, and test together. "
                f"Missing: {missing_keys}"
            )
        return {key: float(split_config[key]) for key in cls.DEFAULT_SPLITS}

    @staticmethod
    def _drop_invalid_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with unusable model features and fail when no signal remains."""
        cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()
        if cleaned.empty:
            raise ValueError("No valid rows remain after feature cleaning.")
        return cleaned

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create deterministic technical indicators from the configured price column."""
        # Derive indicators from one configured price series to avoid mixed price bases.
        price = df[self.price_column]

        df = df.copy()
        df["RSI"] = self._calculate_rsi(price, self.indicators.get("rsi_period", 14))

        macd, macd_signal = self._calculate_macd(
            price,
            self.indicators.get("macd_fast", 12),
            self.indicators.get("macd_slow", 26),
            self.indicators.get("macd_signal", 9),
        )
        df["MACD"] = macd
        df["MACD_Signal"] = macd_signal

        for period in self.indicators.get("sma_periods", [20, 50]):
            df[f"SMA_{period}"] = price.rolling(window=period).mean()

        df["Log_Return"] = np.log(price / price.shift(1))
        return df.dropna()

    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with a bounded denominator for monotonic price windows."""
        delta = series.diff()
        gain = (
            delta.where(delta > 0, 0.0)
            .ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )
        loss = (
            (-delta.where(delta < 0, 0.0))
            .ewm(
                alpha=1 / period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )
        # Monotonic rallies can produce zero average loss; cap the denominator at epsilon.
        safe_loss = loss.replace(0, np.finfo(float).eps)
        rs = gain / safe_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        series: pd.Series,
        fast: int,
        slow: int,
        signal: int,
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal lines with minimum-history windows."""
        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        return macd, macd_signal

    def _build_chronological_splits(self, df: pd.DataFrame) -> dict[str, pd.DatetimeIndex]:
        """Create shared date-based splits with non-empty train, validation, and test windows."""
        # Split by unique dates so every asset observes the same out-of-sample boundaries.
        ratios = {
            "train": float(self.split_config.get("train", 0.7)),
            "validation": float(self.split_config.get("validation", 0.15)),
            "test": float(self.split_config.get("test", 0.15)),
        }
        ratio_sum = sum(ratios.values())
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError(f"Data split ratios must sum to 1.0, got {ratio_sum:.4f}")

        dates = pd.DatetimeIndex(sorted(df.index.unique()))
        if len(dates) < 3:
            raise ValueError(
                "Need at least three unique dates to create train/validation/test splits."
            )

        train_end = max(1, int(len(dates) * ratios["train"]))
        validation_end = max(train_end + 1, train_end + int(len(dates) * ratios["validation"]))
        validation_end = min(validation_end, len(dates) - 1)

        return {
            "train": dates[:train_end],
            "validation": dates[train_end:validation_end],
            "test": dates[validation_end:],
        }

    def _add_normalized_features(
        self,
        df: pd.DataFrame,
        train_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Add normalized feature columns using scalers fitted only on training dates."""
        if not self.normalization_config.get("enabled", False):
            return df

        try:
            from joblib import dump
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError("Feature normalization requires scikit-learn and joblib.") from exc

        feature_columns = self.normalization_config.get("feature_columns")
        if not feature_columns:
            feature_columns = self._infer_feature_columns(df)

        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Cannot normalize missing feature columns: {missing}")
        if self.price_column in feature_columns:
            raise ValueError(
                f"Refusing to normalize raw price_column={self.price_column!r} as a model "
                "feature. Add derived or explicitly scaled price features instead."
            )

        # Fit scaling on the training window only, then reuse those parameters globally.
        scaler = StandardScaler()
        train_mask = df.index.isin(train_dates)
        if not train_mask.any():
            raise ValueError("Cannot fit feature scaler because the training split is empty.")
        scaler.fit(df.loc[train_mask, feature_columns])

        normalized_columns = [f"Norm_{column}" for column in feature_columns]
        transformed = scaler.transform(df[feature_columns])
        result = df.copy()
        result.loc[:, normalized_columns] = transformed

        scaler_path = self.processed_path / "feature_scaler.joblib"
        dump(
            {
                "scaler": scaler,
                "feature_columns": feature_columns,
                "normalized_columns": normalized_columns,
            },
            scaler_path,
        )
        self.logger.info("Saved feature scaler to %s", scaler_path)
        return result

    def _infer_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Infer engineered feature columns while excluding raw prices and labels."""
        excluded_columns = self.REQUIRED_COLUMNS | {"Ticker", self.price_column}
        feature_columns = [
            column
            for column in df.columns
            if column not in excluded_columns and not column.startswith("Norm_")
        ]
        if not feature_columns:
            raise ValueError("No engineered feature columns are available for normalization.")
        return feature_columns

    def _write_split_files(
        self, df: pd.DataFrame, split_dates: dict[str, pd.DatetimeIndex]
    ) -> None:
        """Write split-specific parquet datasets for explicit train/eval consumption."""
        for split_name, dates in split_dates.items():
            split_df = df[df.index.isin(dates)]
            split_df.to_parquet(self.processed_path / f"{split_name}_dataset.parquet")

    def _write_metadata(
        self,
        df: pd.DataFrame,
        output_file: Path,
        skipped_tickers: list[str],
    ) -> None:
        """Write checksum-backed dataset lineage metadata next to the processed artifact."""
        # Capture dataset lineage so experiments can be traced back to an exact artifact.
        checksum = hashlib.sha256(output_file.read_bytes()).hexdigest()
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_file": output_file.name,
            "sha256": checksum,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "tickers": sorted(df["Ticker"].unique().tolist()),
            "skipped_tickers": skipped_tickers,
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "price_column": self.price_column,
        }
        metadata_path = self.processed_path / "dataset_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

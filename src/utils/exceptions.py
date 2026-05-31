"""Define domain exceptions for pipeline failure boundaries."""

from __future__ import annotations


class RLPortfolioError(Exception):
    """Base class for domain failures that callers can handle uniformly."""

    pass


class DownloadError(RLPortfolioError):
    """Raised when a market-data provider cannot satisfy the download contract."""

    pass


class ProcessingError(RLPortfolioError):
    """Raised when raw data cannot be converted into trusted artifacts."""

    pass


class TrainingError(RLPortfolioError):
    """Raised when training fails before a model artifact can be trusted."""

    pass

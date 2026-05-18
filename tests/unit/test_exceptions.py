"""Validate domain exception hierarchy and inheritance contracts."""

from __future__ import annotations

import pytest

from src.utils.exceptions import (
    DownloadError,
    ProcessingError,
    RLPortfolioError,
    TrainingError,
)


# Every domain exception must be a subclass of the shared base class.
@pytest.mark.parametrize(
    "exception_class",
    [DownloadError, ProcessingError, TrainingError],
    ids=["DownloadError", "ProcessingError", "TrainingError"],
)
def test_domain_exceptions_inherit_from_base(exception_class: type) -> None:
    assert issubclass(exception_class, RLPortfolioError)
    assert issubclass(exception_class, Exception)


def test_base_exception_is_a_subclass_of_builtin_exception() -> None:
    assert issubclass(RLPortfolioError, Exception)

    with pytest.raises(RLPortfolioError):
        raise RLPortfolioError("pipeline failure")


def test_download_error_preserves_message() -> None:
    error = DownloadError("Ticker AAPL failed after 3 retries")

    assert "AAPL" in str(error)
    assert "3 retries" in str(error)


def test_processing_error_preserves_message() -> None:
    error = ProcessingError("No valid rows remain")

    assert "No valid rows" in str(error)


def test_training_error_preserves_message() -> None:
    error = TrainingError("Model diverged at step 5000")

    assert "diverged" in str(error)


# A handler for the base class must catch all domain-specific errors.
def test_base_handler_catches_all_domain_exceptions() -> None:
    for exc_class in (DownloadError, ProcessingError, TrainingError):
        with pytest.raises(RLPortfolioError):
            raise exc_class("test")


def test_download_error_is_not_caught_by_sibling_handler() -> None:
    with pytest.raises(DownloadError):
        try:
            raise DownloadError("network timeout")
        except ProcessingError:
            pytest.fail("DownloadError should not be caught by ProcessingError handler")

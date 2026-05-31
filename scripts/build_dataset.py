"""Coordinate the configured market data build pipeline.

The CLI loads the data contract, configures logging, optionally refreshes raw
vendor snapshots, and rebuilds processed artifacts so downstream training uses a
single auditable dataset snapshot.
"""

from __future__ import annotations

import argparse

from src.data.downloader import YFinanceDownloader
from src.data.processor import DataProcessor
from src.utils.config import load_yaml_config
from src.utils.exceptions import DownloadError
from src.utils.logger import configure_logging_from_config, get_logger


def parse_args() -> argparse.Namespace:
    """Parse dataset-build options without touching runtime state."""
    parser = argparse.ArgumentParser(description="Download and process portfolio market data.")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Orchestrate download, processing, and controlled CLI failure reporting."""
    args = parse_args()
    config = load_yaml_config(args.config)
    configure_logging_from_config(config)
    logger = get_logger("build_dataset")

    try:
        # Allow indicator and split regeneration without forcing another vendor download.
        if not args.skip_download:
            downloader = YFinanceDownloader(config)
            downloader.fetch_data()

        # Rebuild processed artifacts on every run so downstream experiments use one snapshot.
        processor = DataProcessor(config)
        processor.process_all()

        logger.info("Dataset build pipeline completed successfully.")
    except (DownloadError, FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

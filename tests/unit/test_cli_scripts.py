"""Validate CLI entry point argument parsing and orchestration contracts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts.build_dataset import main as build_main
from scripts.build_dataset import parse_args as build_parse_args
from scripts.evaluate_agent import main as evaluate_main
from scripts.evaluate_agent import parse_args as evaluate_parse_args
from scripts.train_agent import main as train_main
from scripts.train_agent import parse_args as train_parse_args


# Parser tests pin the default CLI contract exposed to users.
def test_build_dataset_parser_defaults() -> None:
    with patch("sys.argv", ["build_dataset"]):
        args = build_parse_args()

    assert args.config == "configs/data_config.yaml"
    assert args.skip_download is False


def test_build_dataset_parser_accepts_skip_download() -> None:
    with patch("sys.argv", ["build_dataset", "--skip-download"]):
        args = build_parse_args()

    assert args.skip_download is True


def test_evaluate_agent_parser_defaults() -> None:
    with patch("sys.argv", ["evaluate_agent"]):
        args = evaluate_parse_args()

    assert args.env_config == "configs/env_config.yaml"
    assert args.train_config == "configs/train_config.yaml"
    assert args.output_dir == "logs/evaluation"
    assert args.data_split == "test"


def test_train_agent_parser_defaults() -> None:
    with patch("sys.argv", ["train_agent"]):
        args = train_parse_args()

    assert args.env_config == "configs/env_config.yaml"
    assert args.train_config == "configs/train_config.yaml"
    assert args.data_split == "train"


# Main-function tests isolate orchestration by replacing external dependencies.
def test_build_dataset_main_calls_downloader_and_processor(tmp_path) -> None:
    config = {
        "tickers": ["TEST"],
        "paths": {"raw_data": str(tmp_path / "raw"), "processed_data": str(tmp_path / "out")},
    }
    mock_downloader = MagicMock()
    mock_processor = MagicMock()

    with (
        patch("sys.argv", ["build_dataset", "--config", str(tmp_path / "cfg.yaml")]),
        patch("scripts.build_dataset.load_yaml_config", return_value=config),
        patch("scripts.build_dataset.configure_logging_from_config"),
        patch("scripts.build_dataset.YFinanceDownloader", return_value=mock_downloader),
        patch("scripts.build_dataset.DataProcessor", return_value=mock_processor),
    ):
        build_main()

    mock_downloader.fetch_data.assert_called_once()
    mock_processor.process_all.assert_called_once()


def test_build_dataset_main_skips_download_when_flag_set(tmp_path) -> None:
    config = {
        "tickers": ["TEST"],
        "paths": {"raw_data": str(tmp_path / "raw"), "processed_data": str(tmp_path / "out")},
    }
    mock_processor = MagicMock()

    with (
        patch(
            "sys.argv",
            ["build_dataset", "--config", str(tmp_path / "cfg.yaml"), "--skip-download"],
        ),
        patch("scripts.build_dataset.load_yaml_config", return_value=config),
        patch("scripts.build_dataset.configure_logging_from_config"),
        patch("scripts.build_dataset.YFinanceDownloader") as mock_dl_cls,
        patch("scripts.build_dataset.DataProcessor", return_value=mock_processor),
    ):
        build_main()

    mock_dl_cls.assert_not_called()
    mock_processor.process_all.assert_called_once()


def test_build_dataset_main_exits_on_download_error(tmp_path) -> None:
    from src.utils.exceptions import DownloadError

    config = {"tickers": ["TEST"], "paths": {"raw_data": str(tmp_path)}}
    mock_downloader = MagicMock()
    mock_downloader.fetch_data.side_effect = DownloadError("vendor failure")

    try:
        with (
            patch("sys.argv", ["build_dataset", "--config", str(tmp_path / "cfg.yaml")]),
            patch("scripts.build_dataset.load_yaml_config", return_value=config),
            patch("scripts.build_dataset.configure_logging_from_config"),
            patch("scripts.build_dataset.YFinanceDownloader", return_value=mock_downloader),
        ):
            build_main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit(1) on DownloadError")


def test_train_agent_main_calls_learn_and_save(tmp_path) -> None:
    env_config = {
        "data_path": str(tmp_path / "data.parquet"),
        "data_paths": {"train": str(tmp_path / "train.parquet")},
    }
    train_config = {
        "algorithm": "PPO",
        "total_timesteps": 10,
        "model_save_path": str(tmp_path / "model"),
        "seed": 42,
        "n_envs": 1,
        "checkpoint_frequency": 5,
        "checkpoint_dir": str(tmp_path / "ckpt"),
    }
    mock_model = MagicMock()
    mock_env = MagicMock()
    mock_env.close = MagicMock()

    with (
        patch("sys.argv", ["train_agent"]),
        patch("scripts.train_agent.load_yaml_config", side_effect=[env_config, train_config]),
        patch("scripts.train_agent.configure_logging_from_config"),
        patch("scripts.train_agent.set_global_seed"),
        patch("scripts.train_agent.PortfolioEnv", return_value=mock_env),
        patch("scripts.train_agent.AgentFactory.create_agent", return_value=mock_model),
        patch("scripts.train_agent.SaveBestModelCallback"),
    ):
        train_main()

    mock_model.learn.assert_called_once()
    mock_model.save.assert_called_once()
    mock_env.close.assert_called_once()


def test_evaluate_agent_main_calls_backtest_and_saves_report(tmp_path) -> None:
    env_config = {
        "data_path": str(tmp_path / "test.parquet"),
        "data_paths": {"test": str(tmp_path / "test.parquet")},
    }
    train_config = {"algorithm": "PPO", "model_save_path": str(tmp_path / "model")}
    mock_model = MagicMock()
    mock_env = MagicMock()
    mock_env.close = MagicMock()
    mock_backtester = MagicMock()
    mock_backtester.run_backtest.return_value = {"Final Balance": 100_000.0}
    mock_backtester.portfolio_values = [100_000.0]

    with (
        patch("sys.argv", ["evaluate_agent"]),
        patch("scripts.evaluate_agent.load_yaml_config", side_effect=[env_config, train_config]),
        patch("scripts.evaluate_agent.configure_logging_from_config"),
        patch("scripts.evaluate_agent.PortfolioEnv", return_value=mock_env),
        patch("scripts.evaluate_agent.AgentFactory.load_agent", return_value=mock_model),
        patch("scripts.evaluate_agent.Backtester", return_value=mock_backtester),
        patch("scripts.evaluate_agent.BacktestVisualizer"),
        patch("scripts.evaluate_agent.ensure_directory", return_value=tmp_path),
    ):
        evaluate_main()

    mock_backtester.run_backtest.assert_called_once()
    mock_env.close.assert_called_once()

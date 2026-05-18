"""Evaluate a trained portfolio agent and persist deterministic backtest artifacts."""

from __future__ import annotations

import argparse
import json

from src.envs.portfolio_env import PortfolioEnv
from src.evaluation.backtester import Backtester
from src.evaluation.visualizer import BacktestVisualizer
from src.models.agent_factory import AgentFactory
from src.utils.config import config_with_data_split, load_yaml_config, require_keys
from src.utils.logger import configure_logging_from_config, get_logger
from src.utils.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained portfolio management RL agent."
    )
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--output-dir", default="logs/evaluation")
    parser.add_argument("--data-split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = config_with_data_split(load_yaml_config(args.env_config), args.data_split)
    train_config = load_yaml_config(args.train_config)
    configure_logging_from_config(train_config)
    logger = get_logger("evaluate_agent")
    require_keys(train_config, ["algorithm", "model_save_path"], "training")

    logger.info("Initializing Evaluation Environment on data split %s...", args.data_split)
    env = PortfolioEnv(env_config)
    try:
        logger.info("Loading trained model...")
        # Reuse the factory to keep evaluation aligned with the algorithm used for training.
        model = AgentFactory.load_agent(
            train_config["algorithm"], train_config["model_save_path"], env=env
        )

        backtester = Backtester(env=env, model=model)
        report = backtester.run_backtest()

        output_dir = ensure_directory(args.output_dir)
        report_path = output_dir / "backtest_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Save chart diagnostics beside the report so each evaluation run is self-contained.
        visualizer = BacktestVisualizer(output_dir)
        visualizer.save_equity_curve(backtester.portfolio_values)
        visualizer.save_drawdown(backtester.portfolio_values)
        logger.info("Evaluation artifacts saved to %s", output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()

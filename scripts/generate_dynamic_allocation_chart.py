"""Render the 1920x1080 dynamic portfolio allocation chart."""

from __future__ import annotations

import argparse

from src.envs.portfolio_env import PortfolioEnv
from src.evaluation.portfolio_research_charts import (
    run_deterministic_policy_trace,
    save_dynamic_allocation_chart,
)
from src.models.agent_factory import AgentFactory
from src.utils.config import config_with_data_split, load_yaml_config, require_keys
from src.utils.logger import configure_logging_from_config
from src.utils.paths import ensure_directory


def parse_args() -> argparse.Namespace:
    """Parse chart generation options."""
    parser = argparse.ArgumentParser(
        description="Generate a stacked dynamic allocation chart for the trained policy."
    )
    parser.add_argument("--env-config", default="configs/env_config.yaml")
    parser.add_argument("--train-config", default="configs/train_config.yaml")
    parser.add_argument("--data-split", default="test")
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--file-name", default="dynamic_portfolio_allocation_over_time.png")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--end-date",
        help="Optional evaluation end date override, for example 2026-06-23.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the trained policy, run the holdout backtest, and save the chart."""
    args = parse_args()
    base_env_config = load_yaml_config(args.env_config)
    env_config = config_with_data_split(base_env_config, args.data_split)
    if args.end_date:
        env_config["end_date"] = args.end_date
    train_config = load_yaml_config(args.train_config)
    configure_logging_from_config(train_config)
    require_keys(train_config, ["algorithm", "model_save_path"], "training")

    env = PortfolioEnv(env_config)
    try:
        model = AgentFactory.load_agent(
            train_config["algorithm"],
            train_config["model_save_path"],
            env=env,
        )
        trace = run_deterministic_policy_trace(env, model)
        output_path = ensure_directory(args.output_dir) / args.file_name
        chart_path = save_dynamic_allocation_chart(
            trace,
            output_path,
            display_tickers=base_env_config.get("tickers"),
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
        print(chart_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()

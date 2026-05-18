"""Validate PortfolioEnv accounting, observation, and episode-boundary behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.envs.data_provider import EnvironmentDataset
from src.envs.portfolio_env import PortfolioEnv


def make_env_config(dataset_path) -> dict:
    # Centralize the default environment contract used by file-backed tests.
    return {
        "data_path": str(dataset_path),
        "initial_balance": 100_000.0,
        "lookback_window": 5,
        "transaction_fee_pct": 0.001,
        "risk_free_rate": 0.252,
        "price_column": "Close",
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "features": [
            "Norm_RSI",
            "Norm_MACD",
            "Norm_MACD_Signal",
            "Norm_SMA_20",
            "Norm_SMA_50",
            "Norm_Log_Return",
        ],
    }


def test_env_reset_returns_expected_observation_shapes(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))

    obs, info = env.reset(seed=123)

    assert obs["market_history"].shape == (5, 2, 6)
    assert obs["portfolio_weights"].shape == (3,)
    assert info["portfolio_value"] == 100_000.0


def test_zero_action_is_normalized_to_cash_position(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))

    assert np.allclose(info["weights"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["portfolio_value"] > 100_000.0


def test_invalid_action_shape_raises_clear_error(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    try:
        env.step(np.ones(2, dtype=np.float32))
    except ValueError as exc:
        assert "Expected action shape" in str(exc)
    else:
        raise AssertionError("Expected invalid action shape to raise ValueError.")


def test_price_column_cannot_be_used_as_model_feature(synthetic_processed_dataset) -> None:
    config = make_env_config(synthetic_processed_dataset)
    config["features"] = [*config["features"], "Close"]

    try:
        PortfolioEnv(config)
    except ValueError as exc:
        assert "must not also appear in features" in str(exc)
    else:
        raise AssertionError("Expected raw price feature validation to fail.")


def test_transaction_costs_apply_to_risky_asset_turnover_only(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    target_weights = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    assert np.isclose(env._calculate_transaction_costs(target_weights), 100.0)


def test_weights_drift_after_price_movement_before_next_turnover() -> None:
    # Use an injected price panel so the expected post-return drift is exact.
    dates = list(pd.date_range("2024-01-01", periods=4, freq="B"))
    dataset = EnvironmentDataset(
        data_matrix=np.zeros((4, 2, 1), dtype=np.float32),
        close_prices=np.array(
            [
                [100.0, 100.0],
                [200.0, 100.0],
                [200.0, 100.0],
                [200.0, 100.0],
            ],
            dtype=np.float32,
        ),
        tickers=["AAA", "BBB"],
        dates=dates,
    )
    env = PortfolioEnv(
        {
            "initial_balance": 100_000.0,
            "lookback_window": 1,
            "features": ["Norm_RSI"],
            "price_column": "Close",
            "transaction_fee_pct": 0.0,
            "risk_free_rate": 0.0,
        },
        dataset=dataset,
    )
    env.reset()

    _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))

    assert np.allclose(info["weights"], np.array([2 / 3, 1 / 3, 0.0]), atol=1e-6)


def test_transaction_costs_cannot_make_portfolio_value_negative(
    synthetic_processed_dataset,
) -> None:
    config = make_env_config(synthetic_processed_dataset)
    config["transaction_fee_pct"] = 2.0
    config["risk_free_rate"] = 0.0
    env = PortfolioEnv(config)
    env.reset()

    _, _, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert info["portfolio_value"] == 0.0


def test_observation_mutation_does_not_modify_environment_state(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))

    obs, _ = env.reset()
    obs["portfolio_weights"][0] = 1.0
    obs["market_history"][0, 0, 0] = 999.0

    assert env.weights[0] == 0.0
    assert env.data_matrix[0, 0, 0] != 999.0


def test_reset_restores_cash_position_and_initial_balance(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()
    env.step(np.array([0.5, 0.5, 0.0], dtype=np.float32))

    obs, info = env.reset()

    assert info["portfolio_value"] == 100_000.0
    assert np.allclose(obs["portfolio_weights"], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert info["returns_history_len"] == 0


def test_nan_inf_and_negative_action_values_are_sanitized(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    _, _, _, _, info = env.step(np.array([np.nan, np.inf, -1.0], dtype=np.float32))

    assert np.allclose(info["weights"], np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_step_sets_done_when_episode_is_truncated(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    truncated = False
    for _ in range(100):
        _, _, _, truncated, _ = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
        if truncated:
            break

    assert truncated is True
    assert env.done is True


def test_episode_uses_final_price_row_before_truncating() -> None:
    # Ensure truncation happens after the final available return has been applied.
    dates = list(pd.date_range("2024-01-01", periods=4, freq="B"))
    dataset = EnvironmentDataset(
        data_matrix=np.zeros((4, 1, 1), dtype=np.float32),
        close_prices=np.array([[100.0], [110.0], [120.0], [130.0]], dtype=np.float32),
        tickers=["AAA"],
        dates=dates,
    )
    env = PortfolioEnv(
        {
            "initial_balance": 100_000.0,
            "lookback_window": 1,
            "features": ["Norm_RSI"],
            "price_column": "Close",
            "transaction_fee_pct": 0.0,
            "risk_free_rate": 0.0,
            "termination_threshold": 0.0,
        },
        dataset=dataset,
    )
    env.reset()

    truncated = False
    info = {}
    while not truncated:
        _, _, _, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))

    assert np.isclose(info["portfolio_value"], 130_000.0)
    assert env.current_step == len(dataset.close_prices)


def test_step_sets_done_when_termination_threshold_is_breached(
    synthetic_processed_dataset,
) -> None:
    config = make_env_config(synthetic_processed_dataset)
    config["risk_free_rate"] = 0.0
    env = PortfolioEnv(config)
    env.reset()
    env.portfolio_value = 1.0

    _, _, terminated, _, _ = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    assert terminated is True
    assert env.done is True


def test_step_after_done_raises_clear_error(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()
    env.done = True

    try:
        env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    except RuntimeError as exc:
        assert "after the episode" in str(exc)
    else:
        raise AssertionError("Expected step after done to raise RuntimeError.")


def test_render_does_not_raise_for_valid_environment(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()

    env.render()


def test_daily_reward_strategy_can_be_selected(synthetic_processed_dataset) -> None:
    config = make_env_config(synthetic_processed_dataset)
    config["reward_strategy"] = "daily"
    config["risk_free_rate"] = 0.0
    env = PortfolioEnv(config)
    env.reset()

    _, reward, _, _, info = env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    assert reward == 0.0
    assert info["portfolio_value"] == 100_000.0


def test_observation_bounds_are_validated(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.current_step = len(env.data_matrix) + 1

    try:
        env._get_observation()
    except RuntimeError as exc:
        assert "outside the available data matrix" in str(exc)
    else:
        raise AssertionError("Expected invalid observation step to raise RuntimeError.")


def test_step_rejects_current_step_outside_price_matrix(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    env.reset()
    env.current_step = len(env.close_prices)

    try:
        env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    except RuntimeError as exc:
        assert "outside the available price matrix" in str(exc)
    else:
        raise AssertionError("Expected invalid price step to raise RuntimeError.")


def test_drift_weights_falls_back_to_cash_for_invalid_growth(
    synthetic_processed_dataset,
) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))
    weights = env._drift_weights(
        np.array([0.5, 0.5, 0.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        portfolio_growth=0.0,
    )

    assert np.allclose(weights, np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_close_is_idempotent_for_gym_compatibility(synthetic_processed_dataset) -> None:
    env = PortfolioEnv(make_env_config(synthetic_processed_dataset))

    env.close()
    env.close()


def test_missing_data_path_raises_clear_error() -> None:
    config = make_env_config("unused.parquet")
    config.pop("data_path")

    try:
        PortfolioEnv(config)
    except ValueError as exc:
        assert "data_path" in str(exc)
    else:
        raise AssertionError("Expected missing data_path to raise ValueError.")


def test_empty_dataset_file_raises_clear_error(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["Ticker", "Close"]).to_parquet(path)
    config = make_env_config(path)

    try:
        PortfolioEnv(config)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected empty dataset to raise ValueError.")

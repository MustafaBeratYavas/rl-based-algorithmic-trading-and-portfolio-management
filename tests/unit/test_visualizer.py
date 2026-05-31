"""Validate backtest chart rendering, edge cases, and output file integrity."""

from __future__ import annotations

from src.evaluation.visualizer import BacktestVisualizer


# Equity-curve tests verify that portfolio value paths are exported as chart files.
def test_visualizer_writes_equity_curve(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_equity_curve([100.0, 101.0, 99.5])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_equity_curve_accepts_monotonically_increasing_values(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_equity_curve([100.0, 110.0, 120.0, 130.0, 140.0])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_equity_curve_accepts_monotonically_decreasing_values(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_equity_curve([100.0, 90.0, 80.0, 70.0])

    assert output_path.exists()


def test_equity_curve_handles_single_value(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_equity_curve([100.0])

    assert output_path.exists()


def test_equity_curve_respects_custom_filename(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_equity_curve([100.0, 200.0], file_name="custom.png")

    assert output_path.name == "custom.png"
    assert output_path.exists()


# Drawdown tests verify that peak-to-trough diagnostics are exported consistently.
def test_visualizer_writes_drawdown_chart(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_drawdown([100.0, 110.0, 90.0, 120.0])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_drawdown_chart_handles_no_drawdown(tmp_path) -> None:
    # Monotonic increase produces zero drawdown everywhere.
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_drawdown([100.0, 110.0, 120.0, 130.0])

    assert output_path.exists()


def test_drawdown_chart_handles_single_value(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_drawdown([100.0])

    assert output_path.exists()


def test_drawdown_chart_respects_custom_filename(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)

    output_path = visualizer.save_drawdown([100.0, 50.0], file_name="dd.png")

    assert output_path.name == "dd.png"
    assert output_path.exists()


# Output-directory tests keep chart export behavior predictable for nested paths.
def test_visualizer_creates_nested_output_directory(tmp_path) -> None:
    nested_dir = tmp_path / "deep" / "nested" / "charts"
    visualizer = BacktestVisualizer(nested_dir)

    output_path = visualizer.save_equity_curve([100.0, 200.0])

    assert nested_dir.exists()
    assert output_path.exists()


def test_multiple_charts_coexist_in_same_directory(tmp_path) -> None:
    visualizer = BacktestVisualizer(tmp_path)
    values = [100.0, 110.0, 95.0, 120.0]

    equity_path = visualizer.save_equity_curve(values)
    drawdown_path = visualizer.save_drawdown(values)

    assert equity_path.exists()
    assert drawdown_path.exists()
    assert equity_path != drawdown_path

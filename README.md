<div align="center">
  <h1>REINFORCEMENT LEARNING BASED ALGORITHMIC TRADING AND PORTFOLIO MANAGEMENT</h1>
  <p>
    <strong>Project Focus:</strong> Reproducible portfolio research workflows covering market data preparation, Gymnasium-based simulation, reinforcement learning training, and deterministic backtesting.
  </p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white" alt="Python">
    </a>
    <a href="https://gymnasium.farama.org/">
      <img src="https://img.shields.io/badge/Gymnasium-Portfolio%20Env-0081A5" alt="Gymnasium">
    </a>
    <a href="https://stable-baselines3.readthedocs.io/">
      <img src="https://img.shields.io/badge/Stable--Baselines3-PPO%20%7C%20A2C%20%7C%20SAC-6F42C1" alt="Stable-Baselines3">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    </a>
  </p>
</div>

This repository implements a reproducible research pipeline for reinforcement-learning-based portfolio allocation. It covers the full experiment lifecycle: market data preparation, split-aware environment construction, policy training, model evaluation, and deterministic backtest reporting.

The codebase is organized around explicit engineering boundaries. Data processing builds validated artifacts before training begins, the Gymnasium environment focuses only on portfolio simulation and accounting, Stable-Baselines3 integration is centralized behind an algorithm-aware factory, and evaluation is handled by a dedicated backtesting layer. This separation keeps experiments easier to audit, reduces configuration drift, and makes data-leakage safeguards visible in the project structure.

The project is intended for disciplined experimentation rather than live trading. Configuration files define the data universe, cost model, feature contract, training parameters, and evaluation split so runs can be reproduced and reviewed without changing application code.

<details>
<summary><b>Click to expand project structure details</b></summary>

```text
.
├── .github
│   └── workflows
│       └── ci.yml                         # Python matrix quality, type, test, coverage, and artifact workflow
├── configs
│   ├── data_config.yaml                   # Market data, download strictness, indicators, splits, normalization, and logging
│   ├── env_config.yaml                    # Portfolio environment, cost model, reward, features, and date window
│   └── train_config.yaml                  # Algorithm, seed, timesteps, checkpoints, TensorBoard, logging, and hyperparameters
├── data
│   ├── raw                                # Downloaded Yahoo Finance CSV files
│   └── processed                          # Processed parquet datasets, scaler, and dataset metadata
├── logs
│   ├── optuna_studies                     # Persistent Optuna SQLite study files
│   └── tb_logs                            # Stable-Baselines3 TensorBoard event logs
├── models
│   ├── checkpoints                        # Best-model checkpoints created during training
│   └── final                              # Final persisted trained agents
├── scripts
│   ├── build_dataset.py                   # Dataset download and feature engineering CLI
│   ├── evaluate_agent.py                  # Model loading, backtesting, reporting, and chart CLI
│   ├── optimize_hyperparams.py            # Optuna hyperparameter optimization CLI
│   └── train_agent.py                     # RL agent training CLI
├── src
│   ├── data                               # Downloader and DataProcessor pipeline
│   ├── envs                               # Environment data provider, Gymnasium PortfolioEnv, and reward strategies
│   ├── evaluation                         # Backtester, shared financial metrics, benchmark strategies, and visualizations
│   ├── models                             # Stable-Baselines3 AgentFactory and callbacks
│   └── utils                              # Config, exceptions, logging, path, and reproducibility helpers
├── tests
│   ├── conftest.py                        # Pytest configuration and shared fixtures
│   ├── integration                        # Opt-in build/train/evaluate smoke coverage
│   └── unit                               # Pytest regression coverage for core behavior
├── .editorconfig                          # Shared editor formatting defaults
├── .gitattributes                         # Cross-platform line-ending rules
├── .gitignore                             # Git exclusions for runtime artifacts and generated caches
├── .pre-commit-config.yaml                # Local pre-commit hooks for syntax, formatting, linting, and safety checks
├── LICENSE                                # MIT license terms
├── Makefile                               # Local quality, test, and run shortcuts
├── pyproject.toml                         # Project metadata, build config, dependencies, and tool settings
├── README.md                              # Project documentation
├── setup.bat                              # Windows environment bootstrap helper
└── setup.sh                               # Linux/macOS environment bootstrap helper
```

</details>

<details>
<summary><b>Click to expand technology stack details</b></summary>

| Component | Technology | Purpose |
|:---|:---|:---|
| **Market Data** | yfinance | Downloads OHLCV ETF data from Yahoo Finance |
| **Data Processing** | Pandas, NumPy, scikit-learn | Cleaning, technical indicators, chronological splits, feature normalization, and parquet generation |
| **Storage Format** | Parquet, joblib, JSON | Processed datasets, feature scaler, and dataset metadata |
| **RL Environment** | Gymnasium | Long-only portfolio simulation with a cash position and market frictions |
| **RL Algorithms** | Stable-Baselines3 | PPO, A2C, and SAC policy training through a common factory |
| **Optimization** | Optuna | Persistent hyperparameter search with SQLite-backed studies |
| **Evaluation** | NumPy, Matplotlib | Backtest metrics, passive benchmark comparisons, equity curve, and drawdown chart generation |
| **Configuration** | PyYAML | Declarative data, environment, and training settings |
| **Code Quality & CI** | Pytest, pytest-cov, Ruff, mypy, pre-commit, GitHub Actions | Unit and integration smoke tests, coverage gates, linting, formatting, type checking, and continuous validation |

</details>

<details>
<summary><b>Click to expand technical pipeline details</b></summary>

### Environment Pipeline

The environment isolates simulation logic from data loading and exposes a consistent Gymnasium interface for agent training.

| Component | Configuration | Purpose |
|:---|:---|:---|
| Observation Space | Dict of market history and portfolio weights | Provides trailing engineered features plus the current allocation state |
| Action Space | Box of N assets + cash | Represents continuous target portfolio weights |
| Market Frictions | Fees, slippage, market impact | Applies configurable trading costs during rebalancing |
| Reward Design | Configured reward strategy | Supports daily-return and Sharpe-ratio reward schemes |

### Training & Optimization Strategy

Training is driven by YAML configuration and uses Stable-Baselines3 algorithms behind a common factory.

| Stage | Algorithm Focus | Goal |
|:---|:---|:---|
| Policy Training | PPO, A2C, SAC | Learn optimal portfolio allocation weights given the current market observation |
| Hyperparameter Tuning | Optuna | Search selected algorithm parameters using isolated train and validation environments |

### Data & Evaluation Strategy

The data pipeline separates training, validation, and test artifacts before they are consumed by the environment.

| Area | Approach |
|:---|:---|
| Dataset Split | Chronological train, validation, and test parquet files |
| Preprocessing | Feature scaling fitted only on the training window |
| Backtesting | Deterministic evaluation on the selected hold-out split |
| Reporting | JSON backtest report plus equity and drawdown diagnostic charts |

</details>

## Table of Contents

- [Dependencies](#dependencies)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Limitations & Disclaimers](#limitations--disclaimers)
- [License](#license)

## Dependencies

To ensure reproducibility and isolate dependencies, it is recommended to use a virtual environment.

### Step 1 - Create Virtual Environment:

```bash
python -m venv .venv
```

### Step 2 - Activate Virtual Environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 - Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### Step 4 - Install Project Dependencies:

Install only the runtime dependencies when you only need to run the pipeline:

```bash
python -m pip install .
```

Install the development extras when you also need tests, linting, type checking, and pre-commit tooling:

```bash
python -m pip install ".[dev]"
```

*Runtime dependencies: `numpy`, `pandas`, `scikit-learn`, `joblib`, `pyarrow`, `gymnasium`, `stable-baselines3`, `tensorboard`, `PyYAML`, `optuna`, `matplotlib`, `seaborn`, `yfinance`, `lxml`, `requests`.*

*Development dependencies: `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `types-PyYAML`.*

## Quickstart

### 1. Build the Dataset

Creates the reproducible market data snapshot by downloading the configured symbols, engineering features, fitting training-only scalers, and writing split-aware artifacts.

```bash
python -m scripts.build_dataset
```

Use this variant when raw CSV files already exist and only the processed features, scaler, metadata, and split files need to be regenerated.

```bash
python -m scripts.build_dataset --skip-download
```

### 2. Train an Agent

Trains the configured reinforcement learning algorithm on the `train` split using the environment and training contracts defined under `configs/`.

```bash
python -m scripts.train_agent
```

### 3. Evaluate a Trained Agent

Evaluates the saved model on the `test` split and writes deterministic backtest metrics plus diagnostic charts to the selected output directory.

```bash
python -m scripts.evaluate_agent --data-split test --output-dir logs/evaluation
```

## Configuration

Runtime behavior is controlled by three YAML files under `configs/`. Keep these files as the source of truth for experiment setup so data generation, training, and evaluation can be reproduced without editing application code.

| File | Key Parameters | Description |
|:---|:---|:---|
| `data_config.yaml` | `tickers`, date range, `price_column`, paths, download policy, indicators, splits, normalization, logging | Defines how raw market data is downloaded, validated, transformed, split, normalized, and persisted |
| `env_config.yaml` | dataset paths, ticker universe, balance, lookback window, cost model, reward strategy, risk-free rate, features, date range | Defines the portfolio simulation contract consumed by training and evaluation commands |
| `train_config.yaml` | algorithm, timesteps, seed, vectorization, hyperparameters, checkpoint paths, TensorBoard path, logging | Defines the model training contract and default algorithm parameters |

## Limitations & Disclaimers

> **Important:** This section is critical for understanding the real-world applicability of the reported metrics, trained policies, and backtest results.

### Market Data and Execution Scope

- **Historical data dependency:** Results depend on the quality, availability, adjustment policy, and revision history of Yahoo Finance data. Regenerate datasets deliberately and review `dataset_metadata.json` before comparing experiment runs.
- **Daily bar resolution:** The default environment uses daily OHLCV data and does not model intraday price formation, order-book depth, bid-ask spreads, queue priority, or broker-specific execution behavior.
- **Simplified transaction modeling:** Fees, slippage, and market impact are configurable approximations. They should not be treated as a complete representation of real execution costs, liquidity constraints, borrow costs, taxes, or regulatory frictions.
- **Backtest interpretation:** Backtest results are scenario analyses over historical data, not evidence of executable future performance or capital-preserving behavior under live market conditions.

### Reinforcement Learning and Statistical Limitations

- **Training sensitivity:** RL outcomes can change materially with reward design, random seed, date range, feature set, normalization window, cost assumptions, algorithm choice, and hyperparameter settings.
- **Limited validation scope:** The default workflow uses chronological train, validation, and test splits. It does not by itself establish robustness across multiple market regimes, rolling windows, stress periods, or alternative asset universes.
- **Generalization risk:** Performance on a hold-out split does not prove that a learned policy will generalize to unseen market regimes. Any configuration that overlaps train, validation, and test periods should be treated as data leakage.
- **Benchmark dependency:** Policy behavior should be interpreted relative to transparent baselines such as equal-weight, buy-and-hold, or other domain-appropriate strategies rather than in isolation.

### Intended Use

- The project is intended for software engineering, data science, quantitative research, and reinforcement-learning experimentation.
- Generated models, metrics, charts, and reports should be treated as experimental research artifacts, not as trading instructions or investment recommendations.
- Before interpreting any result, review the configuration files, generated datasets, split boundaries, model artifacts, baseline comparisons, and cost assumptions behind the run.
- Do not use this project as the sole basis for portfolio allocation, trading, risk management, or production investment decisions. Consult qualified financial, legal, tax, and compliance professionals for real-world deployment considerations.

## License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for full terms.

Copyright (c) 2026 **Mustafa Berat Yavaş**

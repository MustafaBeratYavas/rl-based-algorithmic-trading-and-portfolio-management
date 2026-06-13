<div align="center">
  <h1>REINFORCEMENT LEARNING BASED ALGORITHMIC TRADING AND PORTFOLIO MANAGEMENT</h1>
  <p>
    <strong>Project Focus:</strong> Reproducible portfolio research workflows for data preparation, Gymnasium simulation, reinforcement learning training, and deterministic backtesting.
  </p>

  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white" alt="Python 3.11-3.13">
    </a>
    <a href="https://github.com/MustafaBeratYavas/rl-based-algorithmic-trading-and-portfolio-management/actions/workflows/ci.yml">
      <img src="https://github.com/MustafaBeratYavas/rl-based-algorithmic-trading-and-portfolio-management/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <a href="./Dockerfile">
      <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker Ready">
    </a>
    <a href="https://docs.astral.sh/ruff/">
      <img src="https://img.shields.io/badge/Ruff-Lint%20%26%20Format-D7FF64?logo=ruff&logoColor=black" alt="Ruff lint and format">
    </a>
    <a href="https://mypy-lang.org/">
      <img src="https://img.shields.io/badge/mypy-Type%20Checked-2A6DB2" alt="mypy type checked">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    </a>
  </p>
</div>

This repository provides a reproducible research pipeline for reinforcement-learning-based portfolio allocation. It supports the full experiment lifecycle: market data preparation, split-aware environment construction, policy training, model evaluation, and deterministic backtest reporting.

The codebase is organized around clear engineering boundaries. Data processing produces validated artifacts before training, the Gymnasium environment is limited to portfolio simulation and accounting, Stable-Baselines3 integration is centralized behind a common agent factory, and evaluation is handled by a dedicated backtesting layer.

The project is designed for disciplined experimentation, not live trading. Configuration files define the data universe, cost model, feature contract, training parameters, and evaluation split so experiments can be reproduced and reviewed without changing application code.

<details>
<summary><b>Click to expand project structure details</b></summary>

```text
.
|-- .github
|   |-- dependabot.yml                     # Scheduled dependency update policy for Python, Actions, Docker, and pre-commit
|   `-- workflows
|       `-- ci.yml                         # Docker quality gate, Python compatibility, coverage, and artifact workflow
|-- configs
|   |-- data_config.yaml                   # Market data, download strictness, indicators, splits, normalization, and logging
|   |-- env_config.yaml                    # Portfolio environment, cost model, reward, features, and date window
|   `-- train_config.yaml                  # Algorithm, seed, timesteps, checkpoints, TensorBoard, logging, and hyperparameters
|-- data
|   |-- raw                                # Downloaded Yahoo Finance CSV files
|   `-- processed                          # Processed parquet datasets, scaler, and dataset metadata
|-- docker
|   `-- entrypoint.sh                      # Container command router for pipeline and quality tasks
|-- logs
|   |-- optuna_studies                     # Persistent Optuna SQLite study files
|   `-- tb_logs                            # Stable-Baselines3 TensorBoard event logs
|-- models
|   |-- checkpoints                        # Best-model checkpoints created during training
|   `-- final                              # Final persisted trained agents
|-- scripts
|   |-- build_dataset.py                   # Dataset download and feature engineering CLI
|   |-- evaluate_agent.py                  # Model loading, backtesting, reporting, and chart CLI
|   |-- optimize_hyperparams.py            # Optuna hyperparameter optimization CLI
|   `-- train_agent.py                     # RL agent training CLI
|-- src
|   |-- data                               # Downloader and DataProcessor pipeline
|   |-- envs                               # Environment data provider, Gymnasium PortfolioEnv, and reward strategies
|   |-- evaluation                         # Backtester, shared financial metrics, benchmark strategies, and visualizations
|   |-- models                             # Stable-Baselines3 AgentFactory and callbacks
|   `-- utils                              # Config, exceptions, logging, path, and reproducibility helpers
|-- tests
|   |-- integration                        # Opt-in build/train/evaluate smoke coverage
|   |-- unit                               # Pytest regression coverage for core behavior
|   `-- conftest.py                        # Pytest configuration and shared fixtures
|-- .dockerignore                          # Docker build context exclusions
|-- .editorconfig                          # Shared editor formatting defaults
|-- .gitattributes                         # Cross-platform line-ending rules
|-- .gitignore                             # Git exclusions for runtime artifacts and generated caches
|-- .pre-commit-config.yaml                # Local pre-commit hooks for syntax, formatting, linting, and safety checks
|-- docker-compose.yml                     # Docker Compose services for pipeline, TensorBoard, and optional GPU runs
|-- Dockerfile                             # Reproducible Python runtime image
|-- LICENSE                                # MIT license terms
|-- Makefile                               # Docker-first quality, test, and run shortcuts
|-- pyproject.toml                         # Project metadata, dependency intent, build config, and tool settings
|-- requirements.lock                      # Locked dependency graph for reproducible CI and Docker development installs
`-- README.md                              # Project documentation
```

</details>

<details>
<summary><b>Click to expand technology stack details</b></summary>

| Component | Technology | Purpose |
|:---|:---|:---|
| **Runtime** | Python 3.11-3.13 | Supported interpreter range for local, Docker, and CI workflows |
| **Market Data** | yfinance | Downloads configured OHLCV market data from Yahoo Finance |
| **Data Processing** | Pandas, NumPy, scikit-learn | Cleans data, engineers features, creates chronological splits, and fits training-only scalers |
| **Storage & Artifacts** | Parquet, joblib, JSON | Persists processed datasets, feature scalers, metadata, model outputs, and reports |
| **RL Environment** | Gymnasium | Exposes a long-only portfolio simulation with cash handling and configurable market frictions |
| **RL Backend** | PyTorch, Stable-Baselines3 | Trains PPO, A2C, and SAC policies through a shared agent factory |
| **Optimization & Tracking** | Optuna, TensorBoard | Runs resumable hyperparameter studies and records training telemetry |
| **Evaluation** | NumPy, Matplotlib | Computes backtest metrics, passive benchmarks, equity curves, and drawdown diagnostics |
| **Configuration** | PyYAML | Keeps data, environment, training, and logging behavior declarative |
| **Dependency Management** | `pyproject.toml`, `requirements.lock`, Dependabot | Declares direct dependencies, locks CI/Docker development installs, and schedules safe update PRs |
| **Containerization** | Docker, Docker Compose | Provides a CPU-first pipeline image, TensorBoard service, and optional GPU profile |
| **Quality & CI** | Pytest, pytest-cov, Ruff, mypy, pre-commit, GitHub Actions | Enforces tests, coverage gates, linting, formatting, type checking, and continuous validation |

</details>

<details>
<summary><b>Click to expand technical pipeline details</b></summary>

### Data Preparation Pipeline

The data pipeline turns declarative market-data configuration into validated artifacts before any training or evaluation command consumes them.

| Stage | Contract | Purpose |
|:---|:---|:---|
| Market Download | Configured ticker universe and date range | Builds the raw OHLCV data snapshot used by the experiment |
| Feature Engineering | Technical indicators and selected feature columns | Produces the model-facing feature contract |
| Split Generation | Chronological train, validation, and test windows | Keeps experiment stages isolated by time |
| Leakage Control | Scalers fitted only on the training window | Prevents validation and test information from influencing preprocessing |
| Artifact Output | Parquet files, scaler artifact, and dataset metadata | Gives training, optimization, and evaluation a reproducible data boundary |

### Environment Pipeline

The environment isolates simulation logic from data loading and exposes a consistent Gymnasium interface for agent training.

| Component | Configuration | Purpose |
|:---|:---|:---|
| Observation Space | Dict of market history and portfolio weights | Provides trailing engineered features plus the current allocation state |
| Action Space | Box over risky assets plus cash | Clips and normalizes model output into valid long-only target allocations |
| Portfolio Accounting | Cash, risky assets, market drift, and termination rules | Revalues holdings, applies cash return, and bounds episode lifecycle |
| Market Frictions | Buy/sell fees, slippage, and market impact | Applies configurable trading costs during rebalancing |
| Reward Design | Configured reward strategy | Supports daily-return and Sharpe-ratio reward schemes |

### Training & Optimization Strategy

Training is driven by YAML configuration and uses Stable-Baselines3 algorithms behind a common factory.

| Stage | Contract | Purpose |
|:---|:---|:---|
| Policy Training | PPO, A2C, SAC through AgentFactory | Learns allocation policies from the configured training split |
| Environment Vectorization | Configurable `n_envs` | Enables SB3 vectorized training while keeping single-env runs easy to debug |
| Checkpointing | Best-model callback and final model path | Persists recoverable training artifacts under the project path contract |
| Hyperparameter Tuning | Optuna with isolated train/eval environments | Searches selected algorithm parameters without sharing validation data with final training |
| Training Telemetry | TensorBoard log directory from config | Records learning curves without hard-coding output paths |

### Data & Evaluation Strategy

Evaluation is deterministic and report-oriented so trained policies can be compared against transparent passive baselines.

| Area | Approach |
|:---|:---|
| Dataset Split | Evaluation runs against a selected chronological hold-out split |
| Policy Execution | Deterministic model prediction through the Gymnasium step contract |
| Metrics | Total return, max drawdown, Sharpe ratio, and Sortino ratio |
| Benchmarks | Equal-weight baseline, plus SPY buy-and-hold and 60/40 stock-bond baselines when available |
| Reporting | JSON backtest report plus equity and drawdown diagnostic charts |

</details>

## Table of Contents

- [Dependencies](#dependencies)
- [Quickstart](#quickstart)
- [Docker Setup and Execution](#docker-setup-and-execution)
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

Install only the runtime dependencies using the locked dependency constraints:

```bash
python -m pip install --constraint requirements.lock .
```

Install the development extras using the same locked dependency constraints:

```bash
python -m pip install --constraint requirements.lock ".[dev]"
```

**Runtime dependencies:** `numpy`, `pandas`, `scikit-learn`, `joblib`, `pyarrow`, `gymnasium`, `stable-baselines3`, `tensorboard`, `PyYAML`, `optuna`, `matplotlib`, `seaborn`, `yfinance`, `lxml`, `requests`.

**Development dependencies:** `mypy`, `pre-commit`, `pytest`, `pytest-cov`, `ruff`, `types-PyYAML`.

## Quickstart

### Prerequisites

* **Python:** Python 3.11 through 3.13 available on your `PATH`.
* **Market data access:** Internet access is required when downloading configured Yahoo Finance data. Use `--skip-download` when raw CSV files already exist.
* **Docker (optional):** Docker and Docker Compose are required only for [containerized execution](#docker-setup-and-execution). The local workflow does not depend on Docker.

### 1. Build the Dataset

Builds a reproducible market-data snapshot from the configured universe, engineers features, fits training-only scalers, and writes split-aware artifacts for downstream training and evaluation.

```bash
python -m scripts.build_dataset
```

### 2. Train an Agent

Trains the configured reinforcement-learning policy on the `train` split using the environment contract and model configuration defined under `configs/`.

```bash
python -m scripts.train_agent
```

### 3. Evaluate a Trained Agent

Runs deterministic policy evaluation on the `test` split and writes structured backtest metrics, benchmark comparisons, and diagnostic charts to the selected output directory.

```bash
python -m scripts.evaluate_agent --data-split test --output-dir logs/evaluation
```

## Docker Setup and Execution

Docker provides a reproducible runtime for data preparation, training, evaluation, and quality checks without installing the full research stack directly on the host machine.

The default Compose image is CPU-first on purpose. It installs PyTorch from the official CPU wheel index and applies the locked dependency constraints before installing the project package, keeping local and CI-equivalent runs predictable without pulling CUDA wheels. GPU execution is available separately through the optional `gpu` Compose profile.

### 1. Build the Image

Builds the default pipeline image:

```bash
docker compose build pipeline
```

### 2. Run the Pipeline

Builds the configured market-data artifacts inside the container:

```bash
docker compose run --rm pipeline build-data
```

Trains the configured reinforcement-learning policy inside the container:

```bash
docker compose run --rm pipeline train
```

Runs deterministic evaluation inside the container and writes backtest outputs to the selected directory:

```bash
docker compose run --rm pipeline evaluate --data-split test --output-dir logs/evaluation
```

### 3. Optional Container Commands

Regenerates processed artifacts from existing raw CSV inputs:

```bash
docker compose run --rm pipeline build-data --skip-download
```

Runs an Optuna hyperparameter search using the configured training and evaluation splits:

```bash
docker compose run --rm pipeline optimize --n-trials 10
```

Serves TensorBoard from the configured log directory:

```bash
docker compose up tensorboard
```

Runs unit tests inside the container:

```bash
docker compose run --rm pipeline test-unit
```

Runs the full containerized quality gate used by CI:

```bash
docker compose run --rm pipeline ci
```

### 4. Optional GPU Image

Builds the optional GPU image for machines with a compatible NVIDIA runtime:

```bash
docker compose --profile gpu build pipeline-gpu
```

Runs training with the GPU-enabled service:

```bash
docker compose --profile gpu run --rm pipeline-gpu train
```

## Configuration

Runtime behavior is controlled by three YAML files under `configs/`. Keep these files as the source of truth for experiment setup so data generation, training, and evaluation can be reproduced without editing application code.

| File | Key Parameters | Description |
|:---|:---|:---|
| `data_config.yaml` | `tickers`, `start_date`, `end_date`, `price_column`, `paths`, `download`, `indicators`, `splits`, `normalization`, `logging` | Defines how raw market data is downloaded, validated, transformed, split, normalized, and persisted |
| `env_config.yaml` | `data_paths`, `tickers`, `initial_balance`, `lookback_window`, `transaction_fee_pct`, `buy_fee_pct`, `sell_fee_pct`, `slippage_pct`, `market_impact_pct`, `reward_strategy`, `risk_free_rate`, `features`, `start_date`, `end_date` | Defines the portfolio simulation contract consumed by training and evaluation commands |
| `train_config.yaml` | `algorithm`, `total_timesteps`, `optimization_timesteps`, `seed`, `n_envs`, `learning_rate`, `batch_size`, `n_steps`, `gamma`, `ent_coef`, `model_save_path`, `checkpoint_dir`, `tensorboard_log`, `logging` | Defines the model training contract and default algorithm parameters |

## Limitations & Disclaimers

> **Important:** This section is critical for understanding the real-world applicability of the reported metrics, trained policies, and backtest results.

### Market Data and Execution Scope

- **Historical data dependency:** Results depend on Yahoo Finance data quality, availability, adjustment policy, and revision history. Regenerate datasets deliberately and review `dataset_metadata.json` before comparing runs.
- **Daily bar resolution:** The default environment uses daily OHLCV data. It does not model intraday price formation, order-book depth, bid-ask spreads, queue priority, or broker-specific execution behavior.
- **Simplified execution costs:** Fees, slippage, and market impact are configurable approximations. They do not represent the full cost of liquidity, borrow constraints, taxes, regulation, or live order execution.
- **Backtest interpretation:** Backtest results are historical scenario analyses, not evidence of executable future performance or capital preservation under live market conditions.

### Reinforcement Learning and Statistical Limitations

- **Training sensitivity:** RL outcomes can change materially with reward design, random seed, date range, feature set, normalization window, cost assumptions, algorithm choice, and hyperparameter settings.
- **Validation scope:** The default workflow uses chronological train, validation, and test splits. It does not establish robustness across market regimes, rolling windows, stress periods, or alternative asset universes by itself.
- **Generalization risk:** Hold-out performance does not prove that a learned policy will generalize to unseen markets. Any overlap between train, validation, and test periods should be treated as data leakage.
- **Benchmark context:** Policy behavior should be interpreted against transparent baselines such as equal-weight, buy-and-hold, or other domain-appropriate strategies rather than in isolation.

### Intended Use

- Use this project for software engineering, data science, quantitative research, and reinforcement-learning experimentation.
- Treat generated models, metrics, charts, and reports as experimental research artifacts, not trading instructions or investment recommendations.
- Review the configuration files, generated datasets, split boundaries, model artifacts, baseline comparisons, and cost assumptions before interpreting any result.
- Do not use this project as the sole basis for portfolio allocation, trading, risk management, or production investment decisions. Consult qualified financial, legal, tax, and compliance professionals before considering real-world deployment.

## License

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for full terms.

Copyright (c) 2026 **Mustafa Berat Yavas**

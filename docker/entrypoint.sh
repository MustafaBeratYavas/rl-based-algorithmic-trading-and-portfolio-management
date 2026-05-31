#!/usr/bin/env sh
set -eu

ensure_runtime_dirs() {
    mkdir -p \
        /app/data/raw \
        /app/data/processed \
        /app/logs/tb_logs \
        /app/logs/optuna_studies \
        /app/models/checkpoints \
        /app/models/final
}

show_help() {
    cat <<'EOF'
RL portfolio container

Commands:
  build-data [args]       Run the market data preparation pipeline
  train [args]            Train the configured RL agent
  evaluate [args]         Evaluate a saved agent and write reports/charts
  optimize [args]         Run Optuna hyperparameter tuning
  tensorboard [args]      Serve TensorBoard on 0.0.0.0:6006
  lint                    Run Ruff lint and format checks
  format                  Apply Ruff fixes and formatting
  typecheck               Run mypy
  test [args]             Run pytest
  test-unit               Run unit tests with coverage
  test-integration        Run opt-in integration tests
  ci                      Run compile, lint, typecheck, and unit tests
  pre-commit [args]       Run pre-commit hooks
  shell                   Open /bin/sh

Examples:
  docker compose run --rm pipeline build-data --skip-download
  docker compose run --rm pipeline train
  docker compose run --rm pipeline evaluate --data-split test
  docker compose run --rm pipeline optimize --n-trials 10
  docker compose run --rm pipeline ci
  docker compose up tensorboard
EOF
}

ensure_runtime_dirs

command_name="${1:-help}"

case "${command_name}" in
    help|-h|--help)
        show_help
        ;;
    shell|sh)
        exec /bin/sh
        ;;
    bash)
        exec bash
        ;;
    build-data|dataset)
        shift
        exec rl-build-dataset "$@"
        ;;
    train)
        shift
        exec rl-train-agent "$@"
        ;;
    evaluate|eval)
        shift
        exec rl-evaluate-agent "$@"
        ;;
    optimize|tune)
        shift
        exec rl-optimize-hyperparams "$@"
        ;;
    tensorboard)
        shift
        exec python -m tensorboard.main \
            --logdir "${TENSORBOARD_LOGDIR:-logs/tb_logs}" \
            --host "${TENSORBOARD_HOST:-0.0.0.0}" \
            --port "${TENSORBOARD_PORT:-6006}" \
            "$@"
        ;;
    lint)
        python -m ruff check src scripts tests
        exec python -m ruff format --check src scripts tests
        ;;
    format)
        python -m ruff check src scripts tests --fix
        exec python -m ruff format src scripts tests
        ;;
    typecheck|type)
        exec python -m mypy src scripts
        ;;
    test)
        shift
        exec python -m pytest "$@"
        ;;
    test-unit)
        exec python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-report=xml
        ;;
    test-integration)
        exec env RUN_INTEGRATION=1 python -m pytest tests/integration -ra --strict-config --strict-markers --tb=short -o addopts=
        ;;
    pre-commit)
        shift
        exec python -m pre_commit run --all-files "$@"
        ;;
    ci)
        python -m compileall -q src scripts tests
        sh -n /usr/local/bin/rl-entrypoint
        python -m ruff check src scripts tests
        python -m ruff format --check src scripts tests
        python -m mypy src scripts
        exec python -m pytest tests/unit --cov=src --cov-report=term-missing --cov-report=xml
        ;;
    *)
        exec "$@"
        ;;
esac

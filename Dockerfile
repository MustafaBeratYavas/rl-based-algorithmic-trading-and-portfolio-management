# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm

ARG APP_UID=1000
ARG APP_GID=1000
ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_EXTRAS=dev
ARG TORCH_VERSION=2.10.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INSTALL_RETRIES=5
ARG PIP_INSTALL_TIMEOUT=120

LABEL org.opencontainers.image.title="Reinforcement Learning-Based Algorithmic Trading and Portfolio Management" \
      org.opencontainers.image.description="Containerized research pipeline for reinforcement-learning-based algorithmic trading and portfolio management." \
      org.opencontainers.image.source="https://github.com/MustafaBeratYavas/rl-based-algorithmic-trading-and-portfolio-management" \
      org.opencontainers.image.licenses="MIT"

ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_PROGRESS_BAR=off \
    PIP_ROOT_USER_ACTION=ignore \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    RUFF_CACHE_DIR=/tmp/ruff_cache \
    MYPY_CACHE_DIR=/tmp/mypy_cache \
    XDG_CACHE_HOME=/tmp/.cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --shell /usr/sbin/nologin app

COPY pyproject.toml README.md LICENSE Makefile .pre-commit-config.yaml ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "${TORCH_INDEX_URL}" ]; then \
        python -m pip install \
            --retries "${PIP_INSTALL_RETRIES}" \
            --timeout "${PIP_INSTALL_TIMEOUT}" \
            --progress-bar off \
            --index-url "${TORCH_INDEX_URL}" \
            "torch==${TORCH_VERSION}"; \
    else \
        python -m pip install \
            --retries "${PIP_INSTALL_RETRIES}" \
            --timeout "${PIP_INSTALL_TIMEOUT}" \
            --progress-bar off \
            "torch==${TORCH_VERSION}"; \
    fi

COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY configs ./configs

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [ -n "${INSTALL_EXTRAS}" ]; then \
        python -m pip install \
            --retries "${PIP_INSTALL_RETRIES}" \
            --timeout "${PIP_INSTALL_TIMEOUT}" \
            --progress-bar off \
            -e ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install \
            --retries "${PIP_INSTALL_RETRIES}" \
            --timeout "${PIP_INSTALL_TIMEOUT}" \
            --progress-bar off \
            .; \
    fi \
    && python -m pip check

COPY docker/entrypoint.sh /usr/local/bin/rl-entrypoint

RUN chmod +x /usr/local/bin/rl-entrypoint \
    && mkdir -p \
        data/raw \
        data/processed \
        logs/tb_logs \
        logs/optuna_studies \
        models/checkpoints \
        models/final \
        /tmp/.cache \
        /tmp/matplotlib \
        /tmp/mypy_cache \
        /tmp/ruff_cache \
    && chown -R app:app /app /tmp/.cache /tmp/matplotlib /tmp/mypy_cache /tmp/ruff_cache

USER app

ENTRYPOINT ["/usr/local/bin/rl-entrypoint"]
CMD ["help"]

#!/usr/bin/env bash

set -Eeuo pipefail

# Bootstrap the local Linux/macOS development environment.

log() {
    printf '[%s] %s\n' "$1" "$2"
}

info() {
    log "INFO" "$1"
}

setup() {
    log "SETUP" "$1"
}

success() {
    log "SUCCESS" "$1"
}

warning() {
    log "WARNING" "$1"
}

error() {
    log "ERROR" "$1" >&2
}

on_error() {
    local line_number="$1"
    error "Unexpected failure at line ${line_number}."
}

find_python() {
    local candidate

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
            command -v "$PYTHON_BIN"
            return 0
        fi

        error "PYTHON_BIN points to a Python interpreter that was not found: ${PYTHON_BIN}"
        return 1
    fi

    for candidate in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done

    return 1
}

ensure_supported_platform() {
    case "$(uname -s)" in
        Linux|Darwin)
            return 0
            ;;
        *)
            warning "This setup helper is intended for Linux/macOS. Continuing anyway."
            ;;
    esac
}

trap 'on_error "$LINENO"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
cd "$SCRIPT_DIR"

info "Starting RL Portfolio Management environment setup..."
ensure_supported_platform

PYTHON="$(find_python)" || {
    error "Python is required. Supported versions are enforced by pyproject.toml."
    exit 1
}

PYTHON_VERSION="$("$PYTHON" -c 'import platform; print(platform.python_version())')"
setup "Using Python ${PYTHON_VERSION}: ${PYTHON}"
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)'; then
    error "Unsupported Python version ${PYTHON_VERSION}. Supported versions are >=3.11 and <3.14."
    exit 1
fi

if [[ ! -d ".venv" || ! -x ".venv/bin/python" ]]; then
    setup "Virtual environment not found or not compatible with Linux/macOS. Creating..."
    "$PYTHON" -m venv .venv
fi

VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    error "Failed to create a usable virtual environment at .venv."
    exit 1
fi

setup "Installing project with development dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
"$VENV_PYTHON" -m pip install -e ".[dev]"

if [[ -d ".git" ]]; then
    setup "Installing pre-commit hooks..."
    if ! "$VENV_PYTHON" -m pre_commit install; then
        warning "pre-commit hook installation failed. Dependencies are installed, but hooks are not active."
    fi
else
    warning "Skipping pre-commit hook installation because .git was not found."
fi

printf '\n'
success "Environment setup completed successfully."
info "Available commands:"
printf '  %s\n' \
    "python -m scripts.build_dataset" \
    "python -m scripts.train_agent" \
    "python -m scripts.evaluate_agent" \
    "python -m scripts.optimize_hyperparams"

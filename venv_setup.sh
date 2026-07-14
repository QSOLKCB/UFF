#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TARGET="."
if [[ "${1:-}" == "--dev" ]]; then
  INSTALL_TARGET=".[dev]"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "Creating .venv with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv

echo "Installing UFF ($INSTALL_TARGET)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e "$INSTALL_TARGET"

echo "UFF environment ready."
echo "Activate it with: source .venv/bin/activate"

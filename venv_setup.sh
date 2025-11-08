#!/usr/bin/env bash
set -e

# Detect OS-specific Python
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Creating virtual environment..."
$PYTHON_BIN -m venv .venv

echo "Activating..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "✅  QSOL_UFF environment ready!"
echo "To activate later: source .venv/bin/activate"


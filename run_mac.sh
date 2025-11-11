#!/usr/bin/env bash
set -euo pipefail
PY_CMD="python3"
if command -v python3.12 >/dev/null 2>&1; then PY_CMD="python3.12"; fi
if [ ! -d ".venv" ]; then "$PY_CMD" -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
exec streamlit run app.py

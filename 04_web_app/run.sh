#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run app.py --server.address "${HOST:-127.0.0.1}" --server.port "${PORT:-8501}" --browser.gatherUsageStats false --theme.base dark "$@"

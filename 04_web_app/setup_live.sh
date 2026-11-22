#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --without-pip .venv
fi
python3 -m pip --python "$PWD/.venv/bin/python" install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
bash setup.sh
python3 prepare_service.py

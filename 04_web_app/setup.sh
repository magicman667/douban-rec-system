#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --without-pip .venv
fi
python3 -m pip --python "$PWD/.venv/bin/python" install -r requirements.txt
mkdir -p runtime
if [ ! -s runtime/NotoSansCJKsc-Regular.otf ]; then
  curl --fail --location --retry 2 --max-time 180 https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf -o runtime/NotoSansCJKsc-Regular.otf
fi

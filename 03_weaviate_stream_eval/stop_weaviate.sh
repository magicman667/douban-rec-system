#!/usr/bin/env bash
set -euo pipefail

BASE="${MEMBER3_OUTPUT_ROOT:-/srv/douban-data/outputs/member3/full-v1-20261120}"
PID_FILE="${BASE}/weaviate.pid"
if [[ ! -f "${PID_FILE}" ]]; then
  echo "No member3 Weaviate PID file"
  exit 0
fi
PID="$(cat "${PID_FILE}")"
if ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "Stale PID file removed"
  exit 0
fi
EXE="$(readlink -f "/proc/${PID}/exe")"
if [[ "${EXE}" != /home/member3/tools/weaviate-v*/weaviate ]]; then
  echo "Refusing to stop unexpected process ${PID}: ${EXE}" >&2
  exit 1
fi
kill -TERM "${PID}"
for _ in $(seq 1 30); do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "Weaviate stopped"
    exit 0
  fi
  sleep 1
done
echo "Weaviate did not stop within 30 seconds" >&2
exit 1

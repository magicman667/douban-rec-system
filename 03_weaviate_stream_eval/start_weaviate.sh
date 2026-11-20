#!/usr/bin/env bash
set -euo pipefail

VERSION="${WEAVIATE_VERSION:-1.40.6}"
BASE="${MEMBER3_OUTPUT_ROOT:-/srv/douban-data/outputs/member3/full-v1-20261120}"
BINARY="${WEAVIATE_BINARY:-/home/member3/tools/weaviate-v${VERSION}/weaviate}"
PID_FILE="${BASE}/weaviate.pid"
LOG_FILE="${BASE}/logs/weaviate.log"

mkdir -p "${BASE}/weaviate_data" "${BASE}/logs"
if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "Weaviate already running with PID $(cat "${PID_FILE}")"
  exit 0
fi

nohup env \
  PERSISTENCE_DATA_PATH="${BASE}/weaviate_data" \
  AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  DEFAULT_VECTORIZER_MODULE=none \
  DISABLE_TELEMETRY=true \
  QUERY_DEFAULTS_LIMIT=25 \
  CLUSTER_HOSTNAME=member3-node \
  GRPC_PORT=50052 \
  RAFT_PORT=18300 \
  CLUSTER_GOSSIP_BIND_PORT=17946 \
  CLUSTER_DATA_BIND_PORT=17947 \
  LOG_LEVEL=info \
  "${BINARY}" --host 127.0.0.1 --port 18080 --scheme http \
  >"${LOG_FILE}" 2>&1 </dev/null &
echo "$!" >"${PID_FILE}"

for _ in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:18080/v1/.well-known/ready >/dev/null; then
    echo "Weaviate ready on 127.0.0.1:18080 and gRPC 50052"
    exit 0
  fi
  sleep 1
done
echo "Weaviate did not become ready; inspect ${LOG_FILE}" >&2
exit 1


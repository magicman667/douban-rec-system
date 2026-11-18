#!/usr/bin/env bash
# Run on the authorized Docker operator account from the repository checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(realpath "${DOUBAN_DATA_ROOT:-/srv/douban-data}")
mountpoint -q "$root"
test -r "$root/processed/full-v1/counts.json"
export DOUBAN_DATA_ROOT="$root"
mkdir -p "$root/outputs/full-v1"
compose() { docker compose "$@"; }
compose up -d --no-build --wait --wait-timeout 420
compose exec -T master hdfs dfsadmin -report > "$root/outputs/full-v1/hdfs.txt"
grep -F 'Live datanodes (2)' "$root/outputs/full-v1/hdfs.txt"
for attempt in {1..24}; do
    compose exec -T master yarn node -list > "$root/outputs/full-v1/yarn.txt" 2>&1 || true
    if grep -Eq 'Total Nodes:[[:space:]]*2' "$root/outputs/full-v1/yarn.txt"; then break; fi
    sleep 5
done
grep -E 'Total Nodes:[[:space:]]*2' "$root/outputs/full-v1/yarn.txt"
compose exec -T master hdfs dfs -mkdir -p /staging/douban_full_v1
for table in movies train_ratings test_ratings; do
    compose exec -T master hdfs dfs -put -f "/handoff/processed/full-v1/$table.jsonl" /staging/douban_full_v1/
done
compose exec -T master spark-submit --master yarn --deploy-mode client --num-executors 2 --executor-memory 1g \
    /project/01_cluster_hive/publish_hive.py /staging/douban_full_v1 /warehouse/douban_full_v1 \
    > "$root/outputs/full-v1/spark_publish.log" 2>&1
compose exec -T hive beeline -u 'jdbc:hive2://localhost:10000/' -n platform -f /project/01_cluster_hive/tables.sql
compose exec -T hive beeline -u 'jdbc:hive2://localhost:10000/' -n platform -e \
    'SELECT COUNT(*) FROM douban_db.movies; SELECT COUNT(*) FROM douban_db.train_ratings; SELECT COUNT(*) FROM douban_db.test_ratings;'
compose exec -T master hdfs dfs -put -f /handoff/processed/full-v1/wordcount_input.tsv /staging/douban_full_v1/
compose exec -T master bash /project/01_cluster_hive/benchmark.sh /bench/full_v1 /staging/douban_full_v1/wordcount_input.tsv /handoff/outputs/full-v1

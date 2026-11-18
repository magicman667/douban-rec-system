#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(realpath "${DOUBAN_DATA_ROOT:-/srv/douban-data}")
mountpoint -q "$root"
export DOUBAN_DATA_ROOT="$root"
out="$root/outputs/full-v1"
data="$root/processed/full-v1"
mkdir -p "$out"
deps="$out/python-deps"
python3 -m pip install --disable-pip-version-check --target "$deps" 'pymongo>=4.10,<5'
PYTHONPATH="$deps" python3 01_cluster_hive/mongo_store.py "$data/movie_reviews.jsonl" > "$out/mongo_import.txt"
run="alert-$(date +%s)"
mkdir "$out/$run"
event_dir="$out/$run/events"
mkdir "$event_dir"
movie_id=-1
emitted=$(date +%s)
python3 - "$movie_id" "$emitted" > "$event_dir/test.json" <<'PY'
import json,sys
from datetime import datetime, timezone
stamp = datetime.fromtimestamp(int(sys.argv[2]), timezone.utc).isoformat()
for _ in range(5):
    print(json.dumps(dict(movie_id=int(sys.argv[1]), rating=1, event_time=stamp)))
PY
docker compose exec -T -e PYTHONPATH=/handoff/outputs/full-v1/python-deps \
    -e DOUBAN_MONGO_URI=mongodb://mongo:27017 master \
    spark-submit --master 'local[2]' --conf spark.sql.shuffle.partitions=4 \
    /project/01_cluster_hive/stream_alert.py "/handoff/outputs/full-v1/$run/events" \
    "/handoff/outputs/full-v1/$run/checkpoint" --once > "$out/$run/stream.log" 2>&1
PYTHONPATH="$deps:01_cluster_hive" python3 - "$movie_id" "$emitted" "$out/$run/stream.log" <<'PY' > "$out/$run/verification.txt"
import sys
from pathlib import Path
from mongo_store import collection, get_movie_reviews
movie_id = int(sys.argv[1])
result = get_movie_reviews(movie_id)
assert result["stream_alert"] is True and result["recent_drop_msg"]
records = [line for line in Path(sys.argv[3]).read_text().splitlines() if line.startswith("microbatch=")]
assert records
print(f"movie_id={movie_id}")
print(f"latency_sec={float(records[0].split('written_at=')[1]) - int(sys.argv[2]):.3f}")
# Remove the synthetic movie after validating its alert.
collection().delete_one({"movie_id": movie_id})
PY
cp "$data"/{rating_distribution.csv,movie_rating_count.csv} "$out/"

#!/usr/bin/env bash
set -euo pipefail
out=$(cd "$(dirname "$0")" && pwd)
r="$out/client-runtime"
export JAVA_HOME="$r/java/jre"
export HADOOP_CONF_DIR=
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_HOSTNAME=localhost
export SPARK_LOCAL_DIRS="$(mktemp -d)"
trap 'rm -rf -- "$SPARK_LOCAL_DIRS"' EXIT
query='SELECT COUNT(*) FROM douban_db.movies; SELECT COUNT(*) FROM douban_db.train_ratings; SELECT COUNT(*) FROM douban_db.test_ratings;'
counts=$("$JAVA_HOME/bin/java" -cp "$r/hive-jdbc-3.1.3-standalone.jar" org.apache.hive.beeline.BeeLine   -u jdbc:hive2://127.0.0.1:10000/default -n "$(id -un)" --silent=true --showHeader=false --outputformat=tsv2   -e "$query" </dev/null)
python3 - "$out/counts.json" "$counts" <<'PY'
import json,sys
from pathlib import Path
meta=json.loads(Path(sys.argv[1]).read_text())
expected=[meta[n] for n in ("movies_kept","train","test")]
actual=[int(x.strip()) for x in sys.argv[2].splitlines() if x.strip().isdigit()]
if expected != actual:
    raise SystemExit(f"Hive counts mismatch: {actual} != {expected}")
print(f"Hive counts: {actual}")
PY
"$r/spark/bin/spark-submit" --master 'local[2]' --conf spark.ui.enabled=false "$out/check_handoff.py" "$out"

#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
base=$1
input=$2
evidence=$3
script=$(realpath "$(dirname "$0")/wordcount.py")
mkdir -p "$evidence"
records=$(hdfs dfs -cat "$input" | wc -l)
printf 'engine,run_id,input_records,runtime_sec\n' > "$evidence/wordcount_runtime.csv"
for run in 1 2 3; do
    for engine in mapreduce spark; do
        dest="$base/$engine-$run"
        start=$(date +%s%N)
        if [ "$engine" = mapreduce ]; then
            hadoop jar "$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar" \
                -D stream.num.map.output.key.fields=2 -D mapreduce.job.reduces=2 \
                -files "$script" -mapper 'python3 wordcount.py map' -reducer 'python3 wordcount.py reduce' \
                -input "$input" -output "$dest" > "$evidence/$engine-$run.log" 2>&1
        else
            spark-submit --master yarn --deploy-mode client --num-executors 2 --executor-memory 1g \
                "$script" spark "$input" "$dest" > "$evidence/$engine-$run.log" 2>&1
        fi
        elapsed=$(awk -v a="$start" -v b="$(date +%s%N)" 'BEGIN {printf "%.3f", (b-a)/1e9}')
        printf '%s,%s,%s,%s\n' "$engine" "$run" "$records" "$elapsed" >> "$evidence/wordcount_runtime.csv"
        hdfs dfs -cat "$dest/part-*" | sort | sha256sum | cut -d ' ' -f 1 > "$evidence/$engine-$run.sha256"
    done
    diff -u "$evidence/mapreduce-$run.sha256" "$evidence/spark-$run.sha256"
done
awk -F, 'NR>1 {print $1,$4}' "$evidence/wordcount_runtime.csv" | sort -k1,1 -k2,2n | awk '{seen[$1]++; if(seen[$1]==2) print $1,$2}' > "$evidence/wordcount_medians.txt"

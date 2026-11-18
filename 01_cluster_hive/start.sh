#!/usr/bin/env bash
set -euo pipefail
role=$1
mkdir -p "$HADOOP_CONF_DIR"
python3 - <<'PY'
import os
from pathlib import Path
from xml.etree import ElementTree as ET

files = {
    "core-site": {"fs.defaultFS": "hdfs://master:8020"},
    "hdfs-site": {
        "dfs.replication": "2", "dfs.namenode.name.dir": "file:///state/name",
        "dfs.datanode.data.dir": "file:///state/data", "dfs.namenode.datanode.registration.ip-hostname-check": "false",
    },
    "mapred-site": {
        "mapreduce.framework.name": "yarn", "mapreduce.application.classpath":
        "/opt/hadoop/share/hadoop/mapreduce/*,/opt/hadoop/share/hadoop/mapreduce/lib/*",
    },
    "yarn-site": {
        "yarn.resourcemanager.hostname": "master", "yarn.nodemanager.aux-services": "mapreduce_shuffle",
        "yarn.nodemanager.resource.memory-mb": "4096", "yarn.nodemanager.resource.cpu-vcores": "2",
        "yarn.scheduler.maximum-allocation-mb": "4096", "yarn.nodemanager.vmem-check-enabled": "false",
        "yarn.nodemanager.env-whitelist": "JAVA_HOME,HADOOP_HOME,HADOOP_CONF_DIR,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_MAPRED_HOME,HADOOP_YARN_HOME,PATH,LANG,LC_ALL,PYTHONHASHSEED",
    },
    "hive-site": {
        "javax.jdo.option.ConnectionURL": "jdbc:derby:;databaseName=/state/metastore_db;create=true",
        "hive.metastore.warehouse.dir": "/warehouse",
        "hive.server2.enable.doAs": "false",
        "hive.metastore.uris": "thrift://metastore:9083" if os.getenv("DOUBAN_ROLE") != "metastore" else "",
    },
}
for name, config in files.items():
    root = ET.Element("configuration")
    for key, value in config.items():
        prop = ET.SubElement(root, "property")
        ET.SubElement(prop, "name").text = key
        ET.SubElement(prop, "value").text = value
    ET.ElementTree(root).write(Path(os.environ["HADOOP_CONF_DIR"]) / (name + ".xml"), encoding="utf-8")
PY
run_pair() {
    "$1" "$2" &
    first=$!
    "$3" "$4" &
    second=$!
    trap 'kill "$first" "$second" 2>/dev/null || true' EXIT
    trap 'exit 143' TERM INT
    wait -n "$first" "$second"
}
case "$role" in
    master)
        if [ ! -f /state/name/current/VERSION ]; then hdfs namenode -format -nonInteractive; fi
        run_pair hdfs namenode yarn resourcemanager
        ;;
    worker) run_pair hdfs datanode yarn nodemanager ;;
    metastore)
        if [ ! -d /state/metastore_db ]; then
            hdfs dfs -mkdir -p /tmp /warehouse /user/platform
            hdfs dfs -chmod 1777 /tmp
            /opt/hive/bin/schematool -dbType derby -initSchema
        fi
        exec /opt/hive/bin/hive --service metastore
        ;;
    hive) exec /opt/hive/bin/hiveserver2 ;;
    *) echo "Unknown role: $role" >&2; exit 2 ;;
esac

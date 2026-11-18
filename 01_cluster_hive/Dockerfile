FROM ghcr.io/apache/hadoop:3.3.6
USER root
ARG SPARK_VERSION=3.5.8
RUN set -eu; for tool in curl python3; do command -v "$tool" || exit 1; done; \
    for artifact in \
      "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
      "https://archive.apache.org/dist/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz"; do \
      name="${artifact##*/}"; \
      curl -fLSs --retry 4 "$artifact" -o "/tmp/$name"; \
      curl -fLSs --retry 4 "$artifact.sha512" -o "/tmp/$name.sha512"; \
      expected="$(grep -Eo '[a-fA-F0-9]{128}' "/tmp/$name.sha512" | head -1 | tr 'A-F' 'a-f')"; \
      test -n "$expected"; echo "$expected  /tmp/$name" | sha512sum -c -; \
      case "$name" in spark*) target=/opt/spark;; *) target=/opt/hive;; esac; \
      mkdir -p "$target"; tar -xf "/tmp/$name" --strip-components=1 -C "$target"; \
      rm "/tmp/$name" "/tmp/$name.sha512"; \
    done; \
    groupadd -g 17001 platform; useradd -m -u 17001 -g platform platform; \
    mkdir -p /state; chown platform:platform /state
ENV PATH="/opt/hadoop/bin:/opt/spark/bin:/opt/hive/bin:$PATH" HADOOP_HOME=/opt/hadoop HIVE_HOME=/opt/hive SPARK_HOME=/opt/spark HADOOP_CONF_DIR=/tmp/douban-config HIVE_CONF_DIR=/tmp/douban-config
COPY start.sh /usr/local/bin/douban-start
USER platform
ENTRYPOINT ["bash", "/usr/local/bin/douban-start"]

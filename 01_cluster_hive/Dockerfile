FROM ghcr.io/apache/hadoop:3.3.6 AS hadoop
FROM mirror.gcr.io/library/ubuntu:24.04
COPY --from=hadoop /opt/hadoop /opt/hadoop
COPY --from=hadoop /usr/lib/jvm/java-1.8.0-openjdk-1.8.0.212.b04-0.el7_6.x86_64 /opt/java
ENV JAVA_HOME=/opt/java
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://repo.huaweicloud.com/ubuntu/@g; s@http://security.ubuntu.com/ubuntu/@http://repo.huaweicloud.com/ubuntu/@g' /etc/apt/sources.list.d/ubuntu.sources;     apt-get -o Acquire::Retries=3 update;     apt-get -o Acquire::Retries=3 install -y --no-install-recommends ca-certificates curl python3 procps;     rm -rf /var/lib/apt/lists/*
ARG SPARK_VERSION=3.5.8
RUN set -eu; for tool in curl python3; do command -v "$tool" || exit 1; done; \
    for artifact in \
      "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz" \
      "https://archive.apache.org/dist/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz"; do \
      name="${artifact##*/}"; \
      curl -fLSs --retry 4 "$artifact" -o "/tmp/$name"; \
      case "$name" in spark*) checksum=sha512; width=128;; *) checksum=sha256; width=64;; esac; \
      curl -fLSs --retry 4 "$artifact.$checksum" -o "/tmp/$name.$checksum"; \
      expected="$(grep -Eo "[a-fA-F0-9]{$width}" "/tmp/$name.$checksum" | head -1 | tr 'A-F' 'a-f')"; \
      test -n "$expected"; echo "$expected  /tmp/$name" | "${checksum}sum" -c -; \
      case "$name" in spark*) target=/opt/spark;; *) target=/opt/hive;; esac; \
      mkdir -p "$target"; tar -xf "/tmp/$name" --strip-components=1 -C "$target"; \
      rm "/tmp/$name" "/tmp/$name.$checksum"; \
    done; \
    groupadd -g 17001 platform; useradd -m -u 17001 -g platform platform; \
    mkdir -p /state; chown platform:platform /state
ENV PATH="/opt/java/bin:/opt/hadoop/bin:/opt/spark/bin:/opt/hive/bin:$PATH" HADOOP_HOME=/opt/hadoop HIVE_HOME=/opt/hive SPARK_HOME=/opt/spark HADOOP_CONF_DIR=/tmp/douban-config HIVE_CONF_DIR=/tmp/douban-config
COPY start.sh /usr/local/bin/douban-start
USER platform
ENTRYPOINT ["bash", "/usr/local/bin/douban-start"]

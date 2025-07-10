FROM elasticsearch:7.17.9

RUN elasticsearch-plugin install --batch https://release.infinilabs.com/analysis-ik/stable/elasticsearch-analysis-ik-7.17.9.zip

ENV discovery.type=single-node
ENV ES_JAVA_OPTS="-Xms512m -Xmx512m"
ENV xpack.security.enabled=false

EXPOSE 9200 9300

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=5 \
  CMD curl -f http://localhost:9200/_cluster/health || exit 1

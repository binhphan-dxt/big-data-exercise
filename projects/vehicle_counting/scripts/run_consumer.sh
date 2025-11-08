#!/bin/bash
# ==========================================
# Script: run_consumer.sh
# Run vehicle counting Spark Structured Streaming Consumer
# - Reads video frames from Kafka
# - Performs YOLO vehicle detection
# - Writes results to PostgreSQL
# ==========================================

set -e

echo "[Consumer] Starting vehicle counting Spark Streaming job..."

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0 \
  --master local[2] \
  --driver-memory 3g \
  --executor-memory 3g \
  --conf spark.sql.streaming.checkpointLocation=/opt/airflow/checkpoints/vehicle_counting_checkpoint \
  /opt/airflow/projects/vehicle_counting/scripts/vehicle_consumer.py

status=$?
if [ $status -eq 0 ]; then
  echo "[Consumer] Completed successfully."
else
  echo "[Consumer] Failed with exit code $status."
fi

exit $status

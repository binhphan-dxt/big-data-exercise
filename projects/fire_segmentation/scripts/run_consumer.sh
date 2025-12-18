#!/bin/bash
set -e

echo "[Consumer] Starting fire detection consumer..."

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.postgresql:postgresql:42.6.0 \
  --master local[2] \
  --driver-memory 3g \
  --executor-memory 3g \
  --conf spark.sql.streaming.checkpointLocation=/opt/airflow/checkpoints/fire_detection_checkpoint \
  /opt/airflow/projects/fire_segmentation/scripts/fire_consumer_streaming.py

status=$?
if [ $status -eq 0 ]; then
  echo "[Consumer] Completed"
else
  echo "[Consumer] Failed: $status"
fi

exit $status

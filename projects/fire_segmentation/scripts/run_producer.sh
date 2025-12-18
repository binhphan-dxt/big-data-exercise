#!/bin/bash
set -e

echo "[Producer] Starting fire video producer..."
python /opt/airflow/projects/fire_segmentation/scripts/fire_producer.py

status=$?
if [ $status -eq 0 ]; then
  echo "[Producer] Completed"
else
  echo "[Producer] Failed: $status"
fi

exit $status

#!/bin/bash
# ==========================================
# Script: run_producer.sh
# Run video producer for vehicle counting
# - Streams frames from camera1.mp4 and camera2.mp4 to Kafka
# - Uses threading for parallel processing
# ==========================================

set -e

echo "[Producer] Starting video frame producer (2 cameras)..."
python /opt/airflow/projects/vehicle_counting/scripts/video_producer.py

status=$?
if [ $status -eq 0 ]; then
  echo "[Producer] Completed successfully."
else
  echo "[Producer] Failed with exit code $status."
fi

exit $status

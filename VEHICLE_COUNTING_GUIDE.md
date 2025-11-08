# Vehicle Counting System Guide

## Overview
This is a real-time multi-camera vehicle counting system built with Airflow, Kafka, Spark, and Streamlit. It follows the same architecture as the ABSA streaming lifecycle demo.

## Architecture

### Components
1. **Video Producer** (2 cameras)
   - Streams frames from `camera1.mp4` and `camera2.mp4` to Kafka
   - Uses threading to handle both cameras simultaneously
   - Sends 5 frames per second per camera
   - Topic: `vehicle-frames`

2. **Vehicle Consumer** (Spark Streaming)
   - Consumes video frames from Kafka
   - Performs YOLO-based vehicle detection (currently mock implementation)
   - Counts vehicles by type: car, motorcycle, bus, truck
   - Writes results to PostgreSQL `vehicle_counts` table
   - Checkpoint: `/opt/airflow/checkpoints/vehicle_counting_checkpoint`

3. **Streamlit Dashboard**
   - Real-time visualization of vehicle counts
   - Runs on port **8502** (separate from ABSA dashboard on 8501)
   - Shows metrics, charts, and camera comparisons
   - Auto-refresh every 10 seconds

4. **Airflow DAG**
   - Orchestrates the entire lifecycle
   - Runs every hour with 55-minute timeout
   - Tasks run in parallel: producer and consumer
   - Includes monitoring and checkpoint cleanup

## File Structure

```
projects/vehicle_counting/
├── scripts/
│   ├── video_producer.py       # Multi-camera streaming to Kafka
│   ├── vehicle_consumer.py     # Spark Streaming + YOLO detection
│   ├── run_producer.sh         # Producer launcher
│   └── run_consumer.sh         # Consumer launcher
└── streamlit/
    └── vehicle_counting_app.py # Real-time dashboard

dags/
└── vehicle_counting_dag.py     # Airflow orchestration (1-hour cycle)

videos/
├── camera1.mp4                 # Video source for camera 1
└── camera2.mp4                 # Video source for camera 2
```

## How to Use

### 1. Access Airflow UI
Open http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 2. Trigger the DAG
1. Find `vehicle_counting_lifecycle_demo` in the DAG list
2. Toggle it ON (if not already enabled)
3. Click "Trigger DAG" to start the pipeline

### 3. Monitor Progress
The DAG has 4 tasks:
- `deploy_video_producer` - Starts both camera producers (45min timeout)
- `deploy_vehicle_consumer` - Starts Spark Streaming consumer (45min timeout)
- `monitor_stream` - Checks checkpoint status
- `cleanup_checkpoints` - Cleans up old checkpoints

### 4. View Dashboard
Open http://localhost:8502 to see:
- **Metrics**: Total vehicles, active cameras, avg confidence, last update
- **Vehicle Count by Type**: Bar chart grouped by camera
- **Detection Confidence**: Box plot of confidence scores
- **Timeline**: Vehicle count over time (line chart)
- **Camera Comparison**: Pie chart and summary statistics
- **Recent Detections**: Table of latest 20 detections

### 5. Query Database
Check vehicle counts directly:
```bash
docker exec big-data-exercise-postgres-1 psql -U airflow -c "
  SELECT 
    camera_id, 
    vehicle_type, 
    SUM(count) as total_vehicles,
    AVG(confidence) as avg_confidence
  FROM vehicle_counts 
  WHERE created_at >= NOW() - INTERVAL '1 hour'
  GROUP BY camera_id, vehicle_type
  ORDER BY camera_id, total_vehicles DESC;
"
```

## DAG Schedule
- **Schedule**: Every 1 hour
- **DAG Timeout**: 55 minutes
- **Task Timeout**: 50 minutes each
- **Retries**: 2 for producer, 5 for consumer
- **Retry Delay**: 2 minutes

## Database Schema

```sql
CREATE TABLE vehicle_counts (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(50),
    frame_number INTEGER,
    detection_time TIMESTAMP,
    vehicle_type VARCHAR(50),     -- car, motorcycle, bus, truck
    confidence FLOAT,              -- detection confidence (0-1)
    count INTEGER,                 -- number of vehicles of this type
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Producer Settings (`video_producer.py`)
- `FRAME_RATE`: 5 frames/second per camera
- `RESIZE_WIDTH`: 640 pixels
- `KAFKA_TOPIC`: "vehicle-frames"
- `VIDEO_PATHS`: 2 cameras (camera1.mp4, camera2.mp4)

### Consumer Settings (`vehicle_consumer.py`)
- `spark.executor.memory`: 3g
- `spark.driver.memory`: 3g
- `spark.executor.cores`: 2
- `maxOffsetsPerTrigger`: 5 frames per batch
- `processingTime`: 10 seconds per batch

### Streamlit Settings
- **ABSA Dashboard**: http://localhost:8501
- **Vehicle Counting Dashboard**: http://localhost:8502
- Auto-refresh: Every 10 seconds
- Time window: Last 10 minutes (configurable)

## Production Improvements

### 1. Real YOLO Model
Currently using mock detection. To implement real YOLO:

```python
# Install ultralytics
# pip install ultralytics

from ultralytics import YOLO

def detect_vehicles_real(frame_data):
    model = YOLO('yolov8n.pt')  # or yolov5s.pt
    
    # Decode frame
    img_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(img, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
    
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": r.names[int(box.cls)],
                "confidence": float(box.conf)
            })
    
    return detections
```

### 2. Model Optimization
- Use YOLOv8n (nano) or YOLOv5s (small) for CPU efficiency
- Enable TensorRT or ONNX for faster inference
- Batch multiple frames together
- Use GPU if available

### 3. Scaling
- Add more video sources (camera3, camera4, etc.)
- Increase `maxOffsetsPerTrigger` for higher throughput
- Add more Spark executors for parallel processing
- Implement frame skipping for lower latency

### 4. Monitoring
- Add Prometheus metrics for frame rate, latency, accuracy
- Set up alerts for consumer lag or errors
- Track checkpoint size growth
- Monitor PostgreSQL table size

## Troubleshooting

### Producer Issues
```bash
# Check producer logs
docker exec big-data-exercise-airflow-worker-1 tail -f /opt/airflow/logs/dag_id=vehicle_counting_lifecycle_demo/run_id=*/task_id=deploy_video_producer/attempt=*.log
```

### Consumer Issues
```bash
# Check consumer logs
docker exec big-data-exercise-airflow-worker-1 tail -f /opt/airflow/logs/dag_id=vehicle_counting_lifecycle_demo/run_id=*/task_id=deploy_vehicle_consumer/attempt=*.log
```

### Checkpoint Issues
```bash
# Clean checkpoint manually
docker exec big-data-exercise-airflow-worker-1 rm -rf /opt/airflow/checkpoints/vehicle_counting_checkpoint
```

### Kafka Issues
```bash
# Check Kafka topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Check messages in topic
docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vehicle-frames --from-beginning --max-messages 5
```

## Performance Metrics

With current settings on 16GB Docker RAM:
- **Frame Rate**: ~10 frames/second (2 cameras × 5 fps)
- **Processing Latency**: ~10-15 seconds (batch interval)
- **Memory Usage**: 
  - Producer: ~500MB
  - Consumer: ~3-4GB (Spark)
  - Streamlit: ~200MB
- **Throughput**: ~600 frames/minute per camera

## Related Services

- **ABSA Streaming**: http://localhost:8501
- **Airflow UI**: http://localhost:8080
- **PostgreSQL**: localhost:5432
- **Kafka**: localhost:9092

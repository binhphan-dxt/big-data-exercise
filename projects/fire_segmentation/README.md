# 🔥 Real-Time Fire Detection System with U-Net Segmentation

## 📋 Project Overview

A real-time fire detection and segmentation system built on Apache Spark Streaming, using a U-Net deep learning model for accurate fire detection in video streams. The system processes video frames through a distributed streaming pipeline, performs semantic segmentation to identify fire regions, and provides real-time visualization through an interactive dashboard.

## 🎯 Key Features

- **Real-Time Processing**: Processes video frames in real-time using Apache Spark Streaming
- **Deep Learning Segmentation**: U-Net architecture for precise fire region detection
- **Distributed Architecture**: Kafka-based message queue for scalable frame distribution
- **Live Dashboard**: Interactive Streamlit dashboard with fire detection statistics and visualization
- **Image Storage**: Stores annotated images with fire mask overlays for analysis
- **Automated Pipeline**: Apache Airflow orchestration for producer/consumer lifecycle management

## 🏗️ System Architecture

```
┌─────────────────┐
│  Fire Video     │
│  (fire_clip.mp4)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fire Producer   │ ──► Reads video frames
│ (Kafka)         │ ──► Encodes as base64
└────────┬────────┘ ──► Sends to Kafka topic
         │
         ▼
┌─────────────────┐
│ Kafka Topic     │
│ "fire-frames"   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Spark Consumer  │ ──► Batch processing (every 10s)
│ + U-Net Model   │ ──► Fire segmentation inference
└────────┬────────┘ ──► Mask overlay creation
         │
         ▼
┌─────────────────┐
│  PostgreSQL     │ ──► Stores detection results
│  Database       │ ──► Stores annotated images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Streamlit UI    │ ──► Real-time statistics
│ Dashboard       │ ──► Fire image gallery
└─────────────────┘ ──► Detection timeline
```

## 🛠️ Technologies Used

### Big Data & Streaming
- **Apache Spark 3.5.1**: Distributed stream processing
- **Apache Kafka**: Message queue for frame distribution
- **Apache Airflow 2.9.0**: Workflow orchestration

### Deep Learning
- **PyTorch**: Deep learning framework
- **U-Net**: Convolutional neural network for semantic segmentation
- **OpenCV**: Image processing and manipulation

### Data Storage & Visualization
- **PostgreSQL**: Relational database for detection records
- **Streamlit**: Interactive dashboard for real-time monitoring
- **Plotly**: Data visualization

### Infrastructure
- **Docker Compose**: Container orchestration
- **Python 3.12**: Primary programming language

## 📊 U-Net Model Architecture

```
Input (256x256x3)
      ↓
[Encoder Block 1] → 32 filters
      ↓ MaxPool
[Encoder Block 2] → 64 filters
      ↓ MaxPool
[Encoder Block 3] → 128 filters
      ↓ MaxPool
[Bottleneck] → 256 filters
      ↓ Upsample + Skip Connection
[Decoder Block 3] → 128 filters
      ↓ Upsample + Skip Connection
[Decoder Block 2] → 64 filters
      ↓ Upsample + Skip Connection
[Decoder Block 1] → 32 filters
      ↓
Output (256x256x1) → Fire Mask
```

**Model Details:**
- Input Size: 256×256×3 RGB images
- Output: Binary segmentation mask
- Loss Function: BCE with Logits Loss
- Threshold: 0.5 for binary mask
- Fire Detection: >1% of frame contains fire

## 📁 Project Structure

```
fire_segmentation/
├── scripts/
│   ├── fire_consumer_streaming.py    # Spark streaming consumer
│   ├── fire_producer.py               # Kafka video producer
│   ├── run_consumer.sh                # Consumer launcher
│   └── run_producer.sh                # Producer launcher
├── streamlit/
│   └── fire_detection_app.py          # Dashboard application
└── README.md                           # This file

dags/
└── fire_detection_dag.py               # Airflow DAG orchestration

models/
└── fire_unet.pt                        # Pre-trained U-Net weights

videos/
└── fire_clip.mp4                       # Test video with fire scenes

docker-compose.yaml                     # Container orchestration
```

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM recommended
- 10GB+ free disk space

### Setup & Installation

1. **Clone the repository**
   ```bash
   cd big-data-exercise
   ```

2. **Start all services**
   ```bash
   docker compose up -d
   ```

   This starts:
   - Apache Airflow (webserver, scheduler, workers)
   - Apache Kafka + Zookeeper
   - PostgreSQL database
   - Streamlit dashboard

3. **Access the web interfaces**
   - Airflow UI: http://localhost:8080 (admin/admin)
   - Streamlit Dashboard: http://localhost:8503

### Running the Pipeline

1. **Navigate to Airflow UI** (http://localhost:8080)

2. **Find the DAG**: `fire_detection_lifecycle`

3. **Trigger the DAG manually** (play button icon)

4. **Monitor execution**:
   - `deploy_fire_producer`: Streams video frames to Kafka
   - `deploy_fire_consumer`: Processes frames with U-Net model
   - `monitor_stream`: Checks checkpoint status
   - `cleanup_checkpoints`: Cleans up after completion

5. **View results** in Streamlit Dashboard (http://localhost:8503):
   - Camera statistics (frames processed, fire detection rate)
   - Fire detection examples with mask overlays
   - Recent detection log

## 📊 Database Schema

```sql
CREATE TABLE fire_detections (
    camera_id VARCHAR(50),
    frame_number INTEGER,
    detection_time TIMESTAMP,
    fire_detected BOOLEAN,
    fire_percentage DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    image_base64 TEXT
);
```

## 🔧 Configuration

### Producer Settings
- **Kafka Topic**: `fire-frames`
- **Frame Rate**: 30 FPS (frame skip: 1)
- **Image Resize**: 640px width
- **JPEG Quality**: 80%

### Consumer Settings
- **Batch Interval**: 10 seconds
- **Max Offsets per Trigger**: 5 frames
- **Image Storage**: Every 5th frame + all fire detections
- **JPEG Quality**: 85%
- **Spark Executor**: 2 cores, 3GB memory

### Model Parameters
- **Image Size**: 256×256 pixels
- **Threshold**: 0.5
- **Fire Detection Threshold**: >1% of frame
- **Device**: CPU (can be changed to GPU)

## 📈 Performance Metrics

- **Processing Latency**: ~2-3 seconds per batch (5 frames)
- **Throughput**: ~150-200 frames per minute
- **Model Inference**: ~100-200ms per frame on CPU
- **Storage**: ~50-100KB per stored image
- **Database Growth**: ~5-10 MB per hour of operation

## 🎨 Dashboard Features

### Statistics Panel
- Total frames processed
- Fire detection rate
- Frames processed by camera

### Fire Detection Gallery
- Shows up to 5 most recent fire detections
- Displays original image with red fire mask overlay
- Shows fire percentage and timestamp
- Auto-refreshes every 10 seconds

### Detection Log
- Scrollable table of recent detections
- Filterable by fire detection status
- Color-coded fire alerts (red background)
- Exportable as CSV

## 🔍 How It Works

1. **Video Streaming**: Producer reads `fire_clip.mp4`, encodes frames as base64, and publishes to Kafka topic "fire-frames"

2. **Spark Processing**: Consumer reads batches from Kafka every 10 seconds

3. **Fire Detection**:
   - Decode base64 frames
   - Resize to 256×256 pixels
   - Normalize pixel values (0-1)
   - Run U-Net inference
   - Apply sigmoid activation
   - Threshold at 0.5 to create binary mask
   - Calculate fire percentage and confidence

4. **Mask Overlay**:
   - Resize mask to original frame size
   - Create red overlay on fire regions
   - Blend original (60%) + overlay (40%)
   - Encode as JPEG base64

5. **Data Storage**:
   - Store detection results in PostgreSQL
   - Store images every 5th frame or when fire detected
   - Timestamp in detection_time column

6. **Visualization**: Streamlit queries PostgreSQL and displays results

## 🐛 Troubleshooting

### No images in dashboard
- Check if DAG is running: Airflow UI → fire_detection_lifecycle
- Verify database connection: `docker exec postgres psql -U airflow -d airflow -c "SELECT COUNT(*) FROM fire_detections WHERE image_base64 IS NOT NULL;"`
- Restart consumer: Stop and re-trigger DAG

### Consumer not processing
- Check Kafka connection: `docker logs kafka`
- Verify Kafka topic: `docker exec kafka kafka-topics --list --bootstrap-server localhost:9092`
- Check Spark logs: Airflow UI → Task Logs → deploy_fire_consumer

### Database connection errors
- Check PostgreSQL: `docker ps | grep postgres`
- Test connection: `docker exec postgres psql -U airflow -d airflow -c "SELECT 1;"`

## 📚 References

- **U-Net Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- **Apache Spark Streaming**: https://spark.apache.org/streaming/
- **Apache Kafka**: https://kafka.apache.org/
- **Streamlit**: https://streamlit.io/

## 👥 Team

**SE363 - Big Data Application Development**  
Faculty of Software Engineering  
University of Information Technology (UIT-VNU)

## 📄 License

This project is developed for educational purposes as part of the SE363 course.

---

**Note**: This system is designed for demonstration and educational purposes. For production deployment, additional considerations for scalability, security, and model accuracy would be required.

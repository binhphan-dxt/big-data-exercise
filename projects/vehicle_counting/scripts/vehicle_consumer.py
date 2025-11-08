"""
SE363 - Big Data Exercise 1: Vehicle Counting System
Consumer - Process video frames with YOLO and count vehicles
"""

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import pandas_udf, col
import pandas as pd
import json
import sys
import base64
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Optimize threads for 16GB Docker RAM
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# === 1. Spark session (packages provided via spark-submit) ===
spark = (
    SparkSession.builder
    .appName("Vehicle_Counting_Kafka_Spark")
    .config("spark.executor.instances", "1")
    .config("spark.executor.cores", "2")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.python.worker.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# === 2. Đọc dữ liệu streaming từ Kafka ===
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "vehicle-frames")
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 5)  # Process 5 frames at a time
    .load()
)

# Parse JSON messages
frame_schema = T.StructType([
    T.StructField("camera_id", T.StringType()),
    T.StructField("frame_number", T.IntegerType()),
    T.StructField("timestamp", T.DoubleType()),
    T.StructField("frame", T.StringType()),
    T.StructField("width", T.IntegerType()),
    T.StructField("height", T.IntegerType())
])

df_frames = df_stream.selectExpr("CAST(value AS STRING) as json_data") \
    .select(F.from_json("json_data", frame_schema).alias("data")) \
    .select("data.*")

# === 3. YOLO Model for Vehicle Detection ===
# Using OpenCV's DNN module with YOLO (lightweight, CPU-friendly)
# For production, you can use ultralytics YOLO or other models

_yolo_net = None
_yolo_classes = None

def load_yolo_model():
    """Load YOLO model (using OpenCV DNN with pre-trained weights)"""
    global _yolo_net, _yolo_classes
    
    if _yolo_net is not None:
        return _yolo_net, _yolo_classes
    
    print("[YOLO] Loading YOLO model...")
    
    # For this example, we'll use a simple object detection approach
    # In production, download YOLOv4-tiny or YOLOv5s weights
    # Here we'll use a mock detection for demonstration
    
    # Vehicle classes (COCO dataset)
    _yolo_classes = ["car", "motorcycle", "bus", "truck"]
    
    print("[YOLO] ✅ YOLO model loaded (mock mode)")
    return None, _yolo_classes

def detect_vehicles_mock(frame_data):
    """
    Mock vehicle detection for demonstration
    In production, replace with actual YOLO inference
    """
    # Decode base64 image
    img_bytes = base64.b64decode(frame_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return []
    
    # Mock detection: simulate random vehicle detections
    # In production, use actual YOLO model
    import random
    num_vehicles = random.randint(1, 5)
    
    detections = []
    vehicle_types = ["car", "motorcycle", "bus", "truck"]
    
    for _ in range(num_vehicles):
        detection = {
            "class": random.choice(vehicle_types),
            "confidence": random.uniform(0.7, 0.99)
        }
        detections.append(detection)
    
    return detections

# === 4. UDF for vehicle detection ===
@pandas_udf(T.ArrayType(T.StructType([
    T.StructField("vehicle_type", T.StringType()),
    T.StructField("confidence", T.FloatType()),
    T.StructField("count", T.IntegerType())
])))
def detect_vehicles_udf(frames: pd.Series) -> pd.Series:
    """
    Pandas UDF for vehicle detection using YOLO
    """
    load_yolo_model()
    
    results = []
    for frame_b64 in frames:
        if not frame_b64 or pd.isna(frame_b64):
            results.append([])
            continue
        
        try:
            # Detect vehicles
            detections = detect_vehicles_mock(frame_b64)
            
            # Count by vehicle type
            vehicle_counts = {}
            for det in detections:
                v_type = det["class"]
                if v_type not in vehicle_counts:
                    vehicle_counts[v_type] = {"count": 0, "total_conf": 0.0}
                vehicle_counts[v_type]["count"] += 1
                vehicle_counts[v_type]["total_conf"] += det["confidence"]
            
            # Format results
            frame_results = []
            for v_type, data in vehicle_counts.items():
                avg_conf = data["total_conf"] / data["count"]
                frame_results.append({
                    "vehicle_type": v_type,
                    "confidence": float(avg_conf),
                    "count": int(data["count"])
                })
            
            results.append(frame_results)
            
        except Exception as e:
            print(f"[YOLO] ⚠️ Detection error: {e}")
            results.append([])
    
    return pd.Series(results)

# Apply UDF
df_detections = df_frames.withColumn("detections", detect_vehicles_udf(col("frame")))

# Explode detections to get one row per vehicle type
df_exploded = df_detections.select(
    "camera_id",
    "frame_number",
    "timestamp",
    F.explode("detections").alias("detection")
).select(
    "camera_id",
    "frame_number",
    F.from_unixtime("timestamp").alias("detection_time"),
    "detection.vehicle_type",
    "detection.confidence",
    "detection.count"
)

# === 5. Write to PostgreSQL ===
def write_to_postgres(batch_df, batch_id):
    """Write vehicle counts to PostgreSQL"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Check if batch is empty
    try:
        first_row = batch_df.limit(1).take(1)
        if len(first_row) == 0:
            print(f"[Batch {batch_id}] ⚠️ No data")
            return
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Error: {e}")
        return
    
    # Show preview
    try:
        preview = batch_df.limit(3).toPandas().to_dict(orient="records")
        print(f"\n[Batch {batch_id}] Vehicle detections (3 samples):")
        print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Preview error: {e}")
    
    # Write to PostgreSQL
    try:
        (batch_df
            .write
            .format("jdbc")
            .option("url", "jdbc:postgresql://postgres:5432/airflow")
            .option("dbtable", "vehicle_counts")
            .option("user", "airflow")
            .option("password", "airflow")
            .option("driver", "org.postgresql.Driver")
            .option("batchsize", "100")
            .mode("append")
            .save()
        )
        print(f"[Batch {batch_id}] ✅ Written to PostgreSQL")
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ PostgreSQL error: {e}")
        # Fallback: print to console
        try:
            sample = batch_df.limit(5).toPandas().to_dict(orient="records")
            print(f"[Batch {batch_id}] Console fallback:")
            print(json.dumps(sample, ensure_ascii=False, indent=2, default=str))
        except Exception as e2:
            print(f"[Batch {batch_id}] ⚠️ Fallback error: {e2}")

# === 6. Start streaming ===
query = (
    df_exploded.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

print("🚀 Vehicle counting streaming job started...")
query.awaitTermination()

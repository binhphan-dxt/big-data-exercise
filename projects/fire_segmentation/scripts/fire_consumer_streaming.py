"""
SE363 - Fire Detection System with U-Net
Spark Streaming consumer for real-time fire segmentation
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
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Spark Configuration
spark = (
    SparkSession.builder
    .appName("FireSegmentation_Streaming")
    .config("spark.executor.instances", "1")
    .config("spark.executor.cores", "2")
    .config("spark.driver.maxResultSize", "2g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .config("spark.python.worker.memory", "1g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Kafka Stream
df_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "fire-frames")
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 5)
    .load()
)

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

# U-Net Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        b = self.bottleneck(self.pool(e3))
        
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.final(d1)


MODEL_PATH = "/opt/airflow/models/fire_unet.pt"
IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = "cpu"

_model = None

def load_fire_model():
    global _model
    if _model is not None:
        return _model
    
    print("[FireModel] Loading U-Net...")
    _model = UNet(in_channels=3, out_channels=1)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            _model.load_state_dict(state_dict)
            print(f"[FireModel] ✅ Loaded weights from {MODEL_PATH}")
        except Exception as e:
            print(f"[FireModel] ⚠️ Error loading weights: {e}")
    else:
        print(f"[FireModel] ⚠️ Model file not found at {MODEL_PATH}")
    
    _model.to(DEVICE).eval()
    torch.set_grad_enabled(False)
    return _model


@pandas_udf(T.StructType([
    T.StructField("fire_detected", T.BooleanType()),
    T.StructField("fire_percentage", T.FloatType()),
    T.StructField("confidence", T.FloatType()),
    T.StructField("image_data", T.StringType())
]))
def detect_fire_udf(frames: pd.Series) -> pd.DataFrame:
    model = load_fire_model()
    results = []
    frame_idx = 0
    
    for frame_b64 in frames:
        frame_idx += 1
        save_image = (frame_idx % 5 == 0)  # Save every 5th frame for visualization
        
        if not frame_b64 or pd.isna(frame_b64):
            results.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
            continue
        
        try:
            # Decode image
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
                continue
            
            # Preprocess
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            img_norm = img_resized.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Calculate metrics
            pred_mask = (pred_prob > THRESHOLD).astype(np.uint8)
            fire_pct = float((pred_mask.sum() / pred_mask.size) * 100)
            confidence = float(pred_prob.mean())
            fire_detected = fire_pct > 1.0  # Fire if >1% of frame
            
            # Create visualization if needed
            image_data_b64 = None
            if save_image or fire_detected:
                # Resize mask to original size
                h, w = img.shape[:2]
                mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
                # Create overlay
                overlay = img_rgb.copy()
                overlay[mask_resized > 0] = [255, 0, 0]  # Red for fire
                result_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_data_b64 = base64.b64encode(buffer).decode('utf-8')
            
            results.append({
                "fire_detected": fire_detected,
                "fire_percentage": fire_pct,
                "confidence": confidence,
                "image_data": image_data_b64
            })
            
        except Exception as e:
            print(f"[FireDetection] ⚠️ Error: {e}")
            results.append({"fire_detected": False, "fire_percentage": 0.0, "confidence": 0.0, "image_data": None})
    
    return pd.DataFrame(results)


# Apply detection - single UDF call
df_detections = df_frames.withColumn("detection", detect_fire_udf(col("frame")))

df_exploded = df_detections.select(
    "camera_id",
    "frame_number",
    F.from_unixtime("timestamp").cast("timestamp").alias("detection_time"),
    "detection.fire_detected",
    "detection.fire_percentage",
    "detection.confidence",
    F.col("detection.image_data").alias("image_base64")
)


def write_to_postgres(batch_df, batch_id):
    """Write fire detections to PostgreSQL"""
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
    print(f"\n{'='*60}")
    print(f"[Batch {batch_id}] Processing fire detections...")
    print(f"Columns: {batch_df.columns}")
    
    # DEBUG: Check image data before writing
    image_count = batch_df.filter(F.col("image_base64").isNotNull()).count()
    total_count = batch_df.count()
    print(f"[DEBUG] Batch has {image_count}/{total_count} rows with images")
    
    # Show sample row with image info
    sample_with_image = batch_df.filter(F.col("image_base64").isNotNull()).limit(1).collect()
    if sample_with_image:
        row = sample_with_image[0]
        img_len = len(row["image_base64"]) if row["image_base64"] else 0
        print(f"[DEBUG] Sample row: camera={row['camera_id']}, frame={row['frame_number']}, fire={row['fire_detected']}, img_len={img_len}")
    
    print(f"{'='*60}")
    batch_df.drop("image_base64").show(5, truncate=False)
    
    try:
        (batch_df.write
            .format("jdbc")
            .option("url", "jdbc:postgresql://postgres:5432/airflow")
            .option("dbtable", "fire_detections")
            .option("user", "airflow")
            .option("password", "airflow")
            .option("driver", "org.postgresql.Driver")
            .mode("append")
            .save()
        )
        print(f"[Batch {batch_id}] ✅ Written {total_count} rows to PostgreSQL ({image_count} with images)")
    except Exception as e:
        print(f"[Batch {batch_id}] ⚠️ Database error: {e}")


query = (
    df_exploded.writeStream
    .foreachBatch(write_to_postgres)
    .outputMode("append")
    .trigger(processingTime="10 seconds")
    .start()
)

print("🔥 Fire detection streaming started...")
query.awaitTermination()

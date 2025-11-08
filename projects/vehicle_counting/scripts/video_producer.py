"""
SE363 - Big Data Exercise 1: Vehicle Counting System
Video Producer - Stream frames from video files to Kafka
"""

import cv2
import json
import time
import base64
import sys
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os

# Configuration
KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "vehicle-frames"
VIDEO_PATHS = [
    "/opt/airflow/videos/camera1.mp4",
    "/opt/airflow/videos/camera2.mp4"
]
FRAME_RATE = 5  # Send 5 frames per second (reduce to avoid overwhelming system)
RESIZE_WIDTH = 640  # Resize frames for faster processing

def create_kafka_producer():
    """Create Kafka producer with retry logic"""
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                max_request_size=10485760,  # 10MB
                buffer_memory=33554432,  # 32MB
                compression_type='gzip'
            )
            print(f"[Producer] ✅ Connected to Kafka at {KAFKA_BROKER}")
            return producer
        except KafkaError as e:
            retry_count += 1
            print(f"[Producer] ⚠️ Failed to connect to Kafka (attempt {retry_count}/{max_retries}): {e}")
            if retry_count < max_retries:
                time.sleep(5)
            else:
                raise

def stream_video_frames(video_path, camera_id, producer):
    """Stream frames from a video file to Kafka"""
    print(f"[Producer] Opening video: {video_path} (Camera {camera_id})")
    
    if not os.path.exists(video_path):
        print(f"[Producer] ❌ Video file not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[Producer] ❌ Failed to open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // FRAME_RATE)  # Skip frames to achieve desired frame rate
    frame_count = 0
    sent_count = 0
    
    print(f"[Producer] Camera {camera_id}: FPS={fps}, Skip={frame_skip}, Target FPS={FRAME_RATE}")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"[Producer] Camera {camera_id}: End of video, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                frame_count = 0
                continue
            
            frame_count += 1
            
            # Skip frames to reduce load
            if frame_count % frame_skip != 0:
                continue
            
            # Resize frame
            height, width = frame.shape[:2]
            new_width = RESIZE_WIDTH
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Create message
            message = {
                "camera_id": camera_id,
                "frame_number": sent_count,
                "timestamp": time.time(),
                "frame": frame_base64,
                "width": new_width,
                "height": new_height
            }
            
            # Send to Kafka
            try:
                producer.send(KAFKA_TOPIC, value=message)
                sent_count += 1
                
                if sent_count % 50 == 0:
                    print(f"[Producer] Camera {camera_id}: Sent {sent_count} frames")
                
            except Exception as e:
                print(f"[Producer] ❌ Error sending frame: {e}")
            
            # Control frame rate
            time.sleep(1.0 / FRAME_RATE)
            
    except KeyboardInterrupt:
        print(f"[Producer] Camera {camera_id}: Stopped by user")
    except Exception as e:
        print(f"[Producer] ❌ Camera {camera_id} error: {e}")
    finally:
        cap.release()
        print(f"[Producer] Camera {camera_id}: Released video capture")

def main():
    """Main function to run both video producers"""
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("🎥 Starting Video Frame Producer for Vehicle Counting")
    print("=" * 60)
    
    # Create Kafka producer
    producer = create_kafka_producer()
    
    # Import threading to run multiple videos
    import threading
    
    threads = []
    for i, video_path in enumerate(VIDEO_PATHS):
        camera_id = f"camera{i+1}"
        thread = threading.Thread(
            target=stream_video_frames,
            args=(video_path, camera_id, producer),
            daemon=True
        )
        thread.start()
        threads.append(thread)
        print(f"[Producer] Started thread for {camera_id}")
    
    # Wait for all threads
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n[Producer] Shutting down...")
    finally:
        producer.close()
        print("[Producer] ✅ Producer closed")

if __name__ == "__main__":
    main()

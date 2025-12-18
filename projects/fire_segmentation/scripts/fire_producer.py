"""
Fire Video Frame Producer - Stream frames to Kafka
"""

import cv2
import json
import time
import base64
import sys
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os

KAFKA_BROKER = "kafka:9092"
KAFKA_TOPIC = "fire-frames"
VIDEO_PATH = "/opt/airflow/videos/fire_clip.mp4"
FRAME_RATE = 3
RESIZE_WIDTH = 640


def create_kafka_producer():
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                max_request_size=10485760,
                buffer_memory=33554432,
                compression_type='gzip'
            )
            print(f"[Producer] ✅ Connected to Kafka")
            return producer
        except KafkaError as e:
            retry_count += 1
            print(f"[Producer] ⚠️ Connection attempt {retry_count}/{max_retries}")
            if retry_count < max_retries:
                time.sleep(5)
            else:
                raise


def stream_video_frames(video_path, camera_id, producer):
    print(f"[Producer] Opening: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"[Producer] ❌ Video not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[Producer] ❌ Cannot open video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = max(1, fps // FRAME_RATE)
    frame_count = 0
    sent_count = 0
    
    print(f"[Producer] FPS={fps}, Skip={frame_skip}")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"[Producer] End of video, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                continue
            
            frame_count += 1
            
            if frame_count % frame_skip != 0:
                continue
            
            # Resize
            height, width = frame.shape[:2]
            new_width = RESIZE_WIDTH
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            message = {
                "camera_id": camera_id,
                "frame_number": sent_count,
                "timestamp": time.time(),
                "frame": frame_base64,
                "width": new_width,
                "height": new_height
            }
            
            try:
                producer.send(KAFKA_TOPIC, value=message)
                sent_count += 1
                
                if sent_count % 30 == 0:
                    print(f"[Producer] Sent {sent_count} frames")
                
            except Exception as e:
                print(f"[Producer] ❌ Error: {e}")
            
            time.sleep(1.0 / FRAME_RATE)
            
    except KeyboardInterrupt:
        print(f"[Producer] Stopped by user")
    finally:
        cap.release()
        print(f"[Producer] Released video")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("🔥 Starting Fire Detection Video Producer")
    print("=" * 60)
    
    producer = create_kafka_producer()
    stream_video_frames(VIDEO_PATH, "fire_camera1", producer)
    producer.close()
    print("[Producer] ✅ Closed")


if __name__ == "__main__":
    main()

import cv2
import time
import logging
import threading
import base64
import psutil
import os
import json
import asyncio
from aiokafka import AIOKafkaProducer
from fogverse.util import get_timestamp_str

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

cpu_usage = 0
memory_usage = 0

KAFKA_SERVER = 'localhost:9094'

def setup(video_sources):
    video_captures = []
    for source in video_sources:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Error opening video source {source}")
            continue
        video_captures.append(cap)
    return video_captures

async def kafka_producer(data, source_number):
    topic = "input_" + str(source_number)
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    await producer.start()
    try:
        await producer.send_and_wait(topic, data)
        logger.info(f"Message sent to {topic}")
    except Exception as e:
        logger.error(f"Error sending message to {topic}: {e}")
    finally:
        await producer.stop()

async def send_frame(cap, source_number):
    global cpu_usage
    global memory_usage

    frame_id = 1
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from source {source_number}")
                break

            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            payload = {
                'uav_id': source_number,
                'frame': encoded_frame,
                'frame_id': str(frame_id),
                'input_timestamp': get_timestamp_str(),
                'input_cpu_usage': str(cpu_usage),
                'input_memory_usage': str(memory_usage)
            }
            await kafka_producer(payload, source_number)
            logger.info(f"Frame sent for source {source_number}")
            frame_id += 1
        except Exception as e:
            logger.error(f"Error processing frame from source {source_number}: {e}")
        await asyncio.sleep(0.033)

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    global cpu_usage
    global memory_usage

    while True:
        cpu_usage = process.cpu_percent(interval=interval) / psutil.cpu_count()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Process CPU Usage: {cpu_usage:.2f}%, Process Memory Usage: {memory_usage:.2f} MB")
        time.sleep(interval)

async def main():
    # Replace IPs with video sources (e.g., camera IDs or file paths)
    video_sources = ["./input/sources/documentation.mp4"]  # 0 refers to the default camera
    video_captures = setup(video_sources)

    if not video_captures:
        logger.error("No valid video sources found. Exiting...")
        return

    video_tasks = []
    for index, cap in enumerate(video_captures):
        video_task = asyncio.create_task(send_frame(cap, index + 1))
        video_tasks.append(video_task)

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    await asyncio.gather(*video_tasks)

if __name__ == '__main__':
    asyncio.run(main())

import cv2
import time
import logging
import threading
import base64
from djitellopy import Tello, TelloSwarm
from fogverse.util import get_timestamp_str
import psutil
import os
import json
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

cpu_usage = 0
memory_usage = 0

KAFKA_SERVER = 'localhost:9094'  # Replace with your Kafka server address

async def kafka_producer(data, drone_number):
    topic = "input_" + str(drone_number)
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    await producer.start()
    try:
        await producer.send_and_wait(topic, data)
        logger.info(f"Message sent to {topic}")
    finally:
        await producer.stop()

async def send_frame(webcam):
    global cpu_usage
    global memory_usage

    frame_id = 1
    while True:
        try:
            ret, frame = webcam.read()
            if not ret:
                logger.error("Failed to grab frame")
                continue
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            payload = {
                'drone_id': 1,
                'frame': encoded_frame,
                'uav_id': str(1),
                'frame_id': str(frame_id),
                'input_timestamp': get_timestamp_str(),
                'input_cpu_usage': str(cpu_usage),
                'input_memory_usage': str(memory_usage)
            }
            await kafka_producer(payload, 1)
            logger.info(f"Frame sent for drone {1}")
            frame_id += 1
        except Exception as e:
            logger.error(f"Error grabbing frame from drone {1}: {e}")
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
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        logger.error("Error: Could not open video device")
        return

    video_task = asyncio.create_task(send_frame(vid))

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    await video_task

if __name__ == '__main__':
    asyncio.run(main())

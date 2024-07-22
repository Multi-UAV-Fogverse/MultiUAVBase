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
from aiokafka import AIOKafkaProducer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

cpu_usage = 0
memory_usage = 0

KAFKA_SERVER = 'localhost:9094'

def setup():
    listIp = ["192.168.0.101", "192.168.0.102", "192.168.0.103"] 
    telloSwarm = TelloSwarm.fromIps(listIp)
    for index, tello in enumerate(telloSwarm.tellos):
        tello.LOGGER.setLevel(logging.ERROR)
        tello.connect()
        tello.streamon()
        tello.change_vs_udp(8881 + index)
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)
    return telloSwarm

async def kafka_producer(data, drone_number):
    topic = "uav_" + str(drone_number)
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

async def send_frame(tello, drone_number):
    global cpu_usage
    global memory_usage

    frame_id = 1
    while True:
        try:
            frame = tello.get_frame_read().frame
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            payload = {
                'drone_id': drone_number,
                'frame': encoded_frame,
                'uav_id': str(drone_number),
                'frame_id': str(frame_id),
                'input_timestamp': get_timestamp_str(),
                'input_cpu_usage': str(cpu_usage),
                'input_memory_usage': str(memory_usage)
            }
            await kafka_producer(payload, drone_number)
            logger.info(f"Frame sent for drone {drone_number}")
            frame_id += 1
        except Exception as e:
            logger.error(f"Error grabbing frame from drone {drone_number}: {e}")
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
    telloSwarm = setup()
    video_tasks = []
    for index, tello in enumerate(telloSwarm.tellos):
        tello_video_task = asyncio.create_task(send_frame(tello, index + 1))
        logger.info(f'Tello {index + 1} Battery: {tello.get_battery()}')
        video_tasks.append(tello_video_task)

    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    await asyncio.gather(*video_tasks)

if __name__ == '__main__':
    asyncio.run(main())

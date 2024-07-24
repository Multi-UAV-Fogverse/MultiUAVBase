import asyncio
from aiokafka import AIOKafkaConsumer
import cv2
import numpy as np
import base64
import json
import requests
import torch
from ultralytics import YOLO
import logging
from fogverse.util import get_timestamp_str
import psutil
import os

logger = logging.getLogger()

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    logger.info("CUDA not available. Model using CPU.")

cpu_usage = 0

async def consume_kafka_messages(drone_id):
    topic_name = f'input_{drone_id}'
    consumer = AIOKafkaConsumer(
        topic_name,
        bootstrap_servers='kafka-broker:9092',  # Replace with your Kafka broker address
        group_id=f'group_{drone_id}',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    await consumer.start()
    try:
        async for message in consumer:
            frame_data = message.value
            process_frame(drone_id, frame_data)
    finally:
        await consumer.stop()

def process_frame(drone_id, frame_data):
    process = psutil.Process(os.getpid())
    # Decode base64 frame data
    decode_frame = base64.b64decode(frame_data['frame'])
    nparr = np.frombuffer(decode_frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if data is not None and len(data.shape) == 3 and data.shape[2] == 3:  # Check for 3-channel image
        results = model(data)  # Get the results from the model
        annotated_frame = data.copy()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if cls_id == 0:  # Only consider the human class (class ID 0)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract additional information from frame_data
        uav_id = frame_data['uav_id']
        frame_id = frame_data['frame_id']
        input_timestamp = frame_data['input_timestamp']
        input_cpu_usage = frame_data['input_cpu_usage']
        input_memory_usage = frame_data['input_memory_usage']
        
        # Prepare the payload to send to the Flask app
        payload = {
            'frame': frame_base64,
            'uav_id': uav_id,
            'frame_id': frame_id,
            'input_timestamp': input_timestamp,
            'input_cpu_usage': input_cpu_usage,
            'input_memory_usage': input_memory_usage
        }
        payload['executor_timestamp'] = get_timestamp_str()
        payload['executor_cpu_usage'] = str(cpu_usage)
        payload['executor_memory_usage'] = str(process.memory_info().rss / 1024 / 1024)
        payload['executor_gpu_memory_reserved'] = str(torch.cuda.memory_reserved(0) / 1024 / 1024)
        payload['executor_gpu_memory_allocated'] = str(torch.cuda.memory_allocated(0) / 1024 / 1024)

        # Send frame to Flask app via HTTP POST
        response = requests.post(f'http://base_client:5001/receive_frame/{drone_id}', json=payload)
        if response.status_code != 200:
            print(f"Failed to send frame: {response.text}")
    else:
        print("Error: Failed to decode frame")

async def monitor_resources(interval=1):
    global cpu_usage
    process = psutil.Process(os.getpid())

    while True:
        cpu_usage = process.cpu_percent(interval=interval) / psutil.cpu_count()
        await asyncio.sleep(interval)

async def main(drone_ids):
    tasks = []
    for drone_id in drone_ids:
        tasks.append(asyncio.create_task(consume_kafka_messages(drone_id)))
    tasks.append(asyncio.create_task(monitor_resources()))
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    drone_ids = ['1', '2', '3', '4']  # List of drone IDs to create consumers for
    asyncio.run(main(drone_ids))

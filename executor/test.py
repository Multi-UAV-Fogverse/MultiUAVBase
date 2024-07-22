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

logger = logging.getLogger()

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    logger.info("CUDA not available. Model using CPU.")

async def consume_kafka_messages():
    consumer = AIOKafkaConsumer(
        'input_1',
        bootstrap_servers='localhost:9094',  # Replace with your Kafka broker address
        group_id='my-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    await consumer.start()
    try:
        async for message in consumer:
            frame_data = message.value
            process_frame(frame_data)
    finally:
        await consumer.stop()

def process_frame(frame_data):
    # Decode base64 frame data
    decode_frame = base64.b64decode(frame_data['frame'])
    nparr = np.frombuffer(decode_frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if data is not None:
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
        
        # Send frame to Flask app via HTTP POST
        drone_id = frame_data.get('drone_id', 'unknown')  # Replace 'unknown' with a default value if needed
        response = requests.post(f'http://localhost:5001/receive_frame/{drone_id}', json={'frame': frame_base64})
        if response.status_code != 200:
            print(f"Failed to send frame: {response.text}")
    else:
        print("Error: Failed to decode frame")

def start_kafka_consumer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(consume_kafka_messages())

if __name__ == '__main__':
    start_kafka_consumer()

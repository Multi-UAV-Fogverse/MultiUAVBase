import cv2
import base64
import numpy as np
import torch
import logging
from fogverse.util import get_timestamp_str
import psutil
import os
import websockets
import asyncio
from ultralytics import YOLO
import imagezmq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    logger.info("CUDA not available. Model using CPU.")

cpu_usage = 0
frame_id = 1

# Initialize ImageHub for receiving frames from drones
image_hub = imagezmq.ImageHub(open_port='tcp://0.0.0.0:5000')

async def send_image(websocket, drone_id, data):
    global cpu_usage
    global frame_id
    process = psutil.Process(os.getpid())

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

        # Encode the frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Send frame to the client using WebSocket
        await websocket.send(jpg_as_text)
        logger.info('Processed frame %d from drone %s', frame_id, drone_id)
        logger.info('Sending image data of length: %d', len(jpg_as_text))

        frame_id += 1

async def main():
    async with websockets.connect('ws://base_client:8000/ws/image/') as websocket:
        while True:
            drone_id, frame_data = image_hub.recv_image()
            await send_image(websocket, drone_id, frame_data)
            image_hub.send_reply(b'OK')

if __name__ == '__main__':
    asyncio.run(main())

from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import eventlet
import base64
import logging
import psutil, os, threading, time
from fogverse.util import get_timestamp_str

eventlet.monkey_patch()
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    logger.info("CUDA not available. Model using CPU.")

frames = {}
headers_store = {}
drone_ids = set()

cpu_usage = 0

@app.route('/process_frame', methods=['POST'])
def process_frame():
    process = psutil.Process(os.getpid())
    global cpu_usage

    frame_data = request.get_json()
    drone_number = frame_data['drone_id']
    drone_ids.add(drone_number)
    decode_frame = base64.b64decode(frame_data['frame'])
    nparr = np.frombuffer(decode_frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    required_keys = ['uav_id', 'frame_id', 'input_timestamp', 'input_cpu_usage', 'input_memory_usage']
    missing_keys = [key for key in required_keys if key not in frame_data]
    if missing_keys:
        logger.error(f"Missing keys: {missing_keys}")
        return jsonify({'status': 'Missing keys', 'missing': missing_keys}), 400

    if len(data.shape) == 3 and data.shape[2] == 3:  # Check for 3-channel image
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
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frames[drone_number] = buffer.tobytes()
        
        # Append additional data to headers
        new_headers = {key: value for key, value in frame_data.items()}
        new_headers.pop("frame", "ok")
        new_headers['executor_timestamp'] = get_timestamp_str()
        new_headers['executor_cpu_usage'] = str(cpu_usage)
        new_headers['executor_memory_usage'] = str(process.memory_info().rss / 1024 / 1024)
        new_headers['executor_gpu_memory_reserved'] = str(torch.cuda.memory_reserved(0) / 1024 / 1024)
        new_headers['executor_gpu_memory_allocated'] = str(torch.cuda.memory_allocated(0) / 1024 / 1024)
        headers_store[drone_number] = new_headers
        
        logger.info(f"Processed and stored frame for drone {drone_number}")
        return jsonify({'status': 'Frame processed'}), 200
    else:
        logger.error("Error: Input data is not in the correct format.")
        return jsonify({'status': 'Incorrect format'}), 404

@app.route('/get_frame/<int:drone_id>', methods=['GET'])
def get_frame(drone_id):
    if drone_id in frames:
        frame = frames.pop(drone_id)
        response = Response(frame, content_type='image/jpeg')
        for key, value in headers_store[drone_id].items():
            response.headers[key] = value
        return response
    else:
        return 'No frame', 404

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    global cpu_usage

    while True:
        cpu_usage = process.cpu_percent(interval=interval) / psutil.cpu_count()

        time.sleep(interval)

@app.route('/get_drones', methods=['GET'])
def get_drone_total():
    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    return jsonify(list(drone_ids))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

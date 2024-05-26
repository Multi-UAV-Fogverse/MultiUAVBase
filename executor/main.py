from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import eventlet
import base64

eventlet.monkey_patch()
app = Flask(__name__)

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("CUDA not available. Model using CPU.")

frames = {}
drone_ids = set()

@app.route('/process_frame', methods=['POST'])
def process_frame():
    frame_data = request.get_json()
    drone_number = frame_data['drone_id']
    drone_ids.add(drone_number)
    decode_frame = base64.b64decode(frame_data['frame'])
    nparr = np.frombuffer(decode_frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ensure the data is in the correct format
    if len(data.shape) == 3 and data.shape[2] == 3:  # Check for 3-channel image
        results = model(data)  # Get the results from the model
        annotated_frame = data.copy()

        # Extract the bounding boxes, confidence scores, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Filter for human class (class ID 0) and annotate the frame
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if cls_id == 0:  # Only consider the human class (class ID 0)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frames[drone_number] = buffer.tobytes()
        print(f"Processed and stored frame for drone {drone_number}")
        return jsonify({'status': 'Frame processed'}), 200
    else:
        print("Error: Input data is not in the correct format.")
        return jsonify({'status': 'Incorrect format'}), 404

@app.route('/get_frame/<int:drone_id>', methods=['GET'])
def get_frame(drone_id):
    if drone_id in frames:
        return frames[drone_id]
    else:
        return 'No frame', 404

@app.route('/get_drones', methods=['GET'])
def get_drone_total():
    return list(drone_ids)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

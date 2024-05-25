from flask import Flask, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import torch
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("CUDA not available. Model using CPU.")

frames = {}

@socketio.on('frame')
def process_frame(data):
    drone_number = data['drone_id']
    nparr = np.frombuffer(request.data, np.uint8)
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
        emit('processed_frame', {'drone_id': drone_number, 'frame': buffer.tobytes()})
    else:
        print("Error: Input data is not in the correct format.")


@app.route('/get_frame/<int:drone_id>', methods=['GET'])
def get_frame(drone_id):
    if drone_id in frames:
        return frames[drone_id]
    else:
        return 'No frame', 404

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

from djitellopy import Tello, TelloSwarm
import cv2
import threading
import time, logging
from ultralytics import YOLO
import base64
import socketio

sio = socketio.Client()

# Set the logging level to ERROR to suppress info and debug logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)

fly = False
video = True
landed = False   
weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)

def setup():
    listIp = ["192.168.0.102"]
    telloSwarm = TelloSwarm.fromIps(listIp)

    for index, tello in enumerate(telloSwarm.tellos):
        tello.LOGGER.setLevel(logging.ERROR) 
        tello.connect()
        print(f'Tello {index+1} Battery : {tello.get_battery()}')
        tello.change_vs_udp(9000 + index)  # Change the ports to avoid conflicts
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)

    return telloSwarm

def process(data):
    if len(data.shape) == 3 and data.shape[2] == 3:
        results = model(data)
        annotated_frame = data.copy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated_frame
    else:
        return data

def tello_video(tello, drone_number):
    while not landed:
        try:
            frame = tello.get_frame_read().frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = process(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            event = tello.address[0]
            print(f"Emitting frame for {event}")  # Debug statement
            sio.emit("192.168.0.102", frame_data)
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in tello_video for drone {drone_number}: {e}")
            break

def stream_on(telloSwarm):
    telloSwarm.parallel(lambda drone, tello: tello.streamon())
    videoThreads = []
    if video:
        for index, tello in enumerate(telloSwarm.tellos):
            tello_video_thread = threading.Thread(target=tello_video, args=(tello, index + 1), daemon=True)
            tello_video_thread.start()
            videoThreads.append(tello_video_thread)
        time.sleep(3)
    return videoThreads

def stream_off(videoThreads, telloSwarm):
    if video:
        for tello_video in videoThreads:
            tello_video.join()
    telloSwarm.parallel(lambda drone, tello: tello.streamoff())

def main():
    telloSwarm = setup()
    videoThreads = stream_on(telloSwarm)
    stream_off(videoThreads, telloSwarm)
    telloSwarm.end()

def start_tello():
    main()

# Event handler for connection
@sio.event
def connect():
    print('Connected to Flask-SocketIO server')
    # Start the Tello setup and video streaming in a separate thread
    tello_thread = threading.Thread(target=start_tello, daemon=True)
    tello_thread.start()

# Event handler for disconnection
@sio.event
def disconnect():
    print('Disconnected from Flask-SocketIO server')

if __name__ == '__main__':
    print("Attempting to connect to Flask-SocketIO server")
    # Attempt to connect to the Flask-SocketIO server
    try:
        sio.connect('http://localhost:5000')
        sio.wait()  # Wait for the connection to be established
    except socketio.exceptions.ConnectionError as e:
        print(f"Connection failed: {e}")

    print("Waiting for connection event")

from djitellopy import Tello, TelloSwarm
import torch
import cv2
import threading
import time, logging
import psutil
import os
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
# Using fogverse just for logging and util
from fogverse.fogverse_logging import FogVerseLogging
from fogverse.util import get_timestamp, get_timestamp_str, timestamp_to_datetime

# Set the logging level to ERROR to suppress info and debug logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)

fly = False
video = True
landed = False   
weights_path = 'yolo-Weights/yolov8n.pt'

cpu_usage = 0
memory_usage = 0
gpu_memory_reserved = 0
gpu_memory_allocated = 0


model = YOLO(weights_path)
if torch.cuda.is_available():
    model.to('cuda')
    print("Model moved to GPU.")
else:
    print("CUDA not available. Model using CPU.")

def setup():
    listIp = ["192.168.0.102"]
    telloSwarm = TelloSwarm.fromIps(listIp)

    for index, tello in enumerate(telloSwarm.tellos):
        # Change the logging level to ERROR only, ignore all INFO feedback from DJITELLOPY
        tello.LOGGER.setLevel(logging.ERROR) 

        tello.connect()

        print(f'Tello {index+1} Battery : {tello.get_battery()}')

        # Change the video stream port to 888x, so that they will not be conflicting with each other, the original port 11111.
        tello.change_vs_udp(8881+index)
        # Set resolution and bitrate low to make sure it can show video
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)

    return telloSwarm

def compress_image(data, quality=30):
    # Convert the OpenCV image (BGR) to PIL image (RGB)
    data_rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(data_rgb)

    # Compress the image using PIL
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)  # Adjust quality as needed
    buffer.seek(0)

    # Convert the compressed PIL image back to OpenCV format (BGR)
    compressed_image = Image.open(buffer)
    compressed_image = np.array(compressed_image)
    compressed_image = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR)

    return compressed_image

def process(data):
    # Compress the image at the beginning
    data = compress_image(data)
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

        return annotated_frame  # Return the annotated image
    else:
        print("Error: Input data is not in the correct format.")
        return data

def tello_video(tello, drone_number):
    global cpu_usage
    global memory_usage
    global gpu_memory_reserved
    global gpu_memory_allocated

    uav_headers_logger = ["uav_id", "frame_id", "cpu_usage", "memory_usage", "gpu_memory_reserved", "gpu_memory_allocated", "input_timestamp", "client_timestamp", "latency"]
    fogverse_uav_logger = FogVerseLogging(
        name=f'uav_{drone_number}_scenario_1',
        dirname="uav-logs",
        csv_header=uav_headers_logger,
        level= logging.INFO + 2
    )
    frame_id = 0
    while not landed:
        input_timestamp = get_timestamp()

        frame = tello.get_frame_read().frame
        frame_id += 1
        frame = process(frame)
        cv2.imshow(f'Tello {drone_number}' , frame)
        cv2.moveWindow(f'Tello {drone_number}', (drone_number - 1) * 900, 50)

        client_timestamp = get_timestamp()
        latency = client_timestamp - input_timestamp
        frame_log = [drone_number, frame_id, cpu_usage, memory_usage, gpu_memory_reserved, gpu_memory_allocated, input_timestamp, client_timestamp, latency]
        fogverse_uav_logger.csv_log(frame_log)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow(f'Tello {drone_number}')
            break
        
        time.sleep(0.01)  # Small delay to allow other threads to run
    
    cv2.destroyAllWindows()

def stream_on(telloSwarm):
    telloSwarm.parallel(lambda drone, tello: tello.streamon())

    videoThreads = []
    if video:
        for index, tello in enumerate(telloSwarm.tellos):
            tello_video_new = threading.Thread(target=tello_video, args=(tello, index+1), daemon=True)
            tello_video_new.start()
            videoThreads.append(tello_video_new)

        time.sleep(3)
    return videoThreads

def stream_off(videoThreads, telloSwarm):
    if video:    
        for tello_video in videoThreads:
            tello_video.join()

    telloSwarm.parallel(lambda drone, tello: tello.streamoff())

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    global cpu_usage
    global memory_usage
    global gpu_memory_reserved
    global gpu_memory_allocated

    while not landed:
        cpu_usage = process.cpu_percent(interval=interval) / psutil.cpu_count()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        if torch.cuda.is_available():
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024 / 1024  # Convert to MB
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024  # Convert to MB
            print(f"Process CPU Usage: {cpu_usage:.2f}%, Process Memory Usage: {memory_usage:.2f} MB, GPU Memory Reserved: {gpu_memory_reserved:.2f} MB, GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        else:
            print(f"Process CPU Usage: {cpu_usage:.2f}%, Process Memory Usage: {memory_usage:.2f} MB")
        time.sleep(interval)

def main():
    telloSwarm = setup()
    
    # Start resource monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    videoThreads = stream_on(telloSwarm)
    stream_off(videoThreads, telloSwarm)
    telloSwarm.end()

if __name__ == '__main__':
    main()

from djitellopy import Tello, TelloSwarm
import cv2
import threading
import time, logging
import asyncio

from ultralytics import YOLO
# Set the logging level to ERROR to suppress info and debug logs
logging.getLogger('ultralytics').setLevel(logging.ERROR)

fly = False
video = True
landed = False   
weights_path = 'yolo-Weights/yolov8n.pt'
model = YOLO(weights_path)

def setup():
    listIp = ["192.168.0.101"]
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

def process(data):
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
    while not landed:  
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame = process(frame)
        cv2.imshow(f'Tello {drone_number}' , frame)
        cv2.moveWindow(f'Tello {drone_number}', (drone_number - 1)*900, 50)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cv2.destroyWindow(f'Tello {drone_number}')
            break

def tello_flip(tello, direction):
    tello.flip(direction)
    
def tello_mpad(tello, x, y, z, speed, mpad):
    tello.enable_mission_pads
    tello.go_xyz_speed_mid(x, y, z, speed, mpad)

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

def fly(telloSwarm):
    telloSwarm.send_rc_control(0,0,0,0)
    telloSwarm.takeoff()
   
    telloSwarm.land()
    landed = True

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


if __name__ == '__main__':
    main()
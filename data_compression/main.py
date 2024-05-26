import cv2
import time
import logging
import threading
import requests
import base64
from djitellopy import Tello, TelloSwarm

def setup():
    listIp = ["192.168.0.102"]  # Add more IPs as needed
    telloSwarm = TelloSwarm.fromIps(listIp)
    for index, tello in enumerate(telloSwarm.tellos):
        tello.LOGGER.setLevel(logging.ERROR)
        tello.connect()
        tello.streamon()
        tello.change_vs_udp(8881 + index)
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)
    return telloSwarm

def send_frame(tello, drone_number):
    while True:
        try:
            frame = tello.get_frame_read().frame
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])  # Reduce quality to 70%
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            payload = {'drone_id': drone_number, 'frame': encoded_frame}
            response = requests.post('http://localhost:5000/process_frame', json=payload)
            if response.status_code != 200:
                print(f"Failed to send frame for drone {drone_number}: {response.text}")
        except Exception as e:
            print(f"Error grabbing frame from drone {drone_number}: {e}")
        time.sleep(0.1)

def main():
    telloSwarm = setup()
    videoThreads = []
    for index, tello in enumerate(telloSwarm.tellos):
        tello_video_new = threading.Thread(target=send_frame, args=(tello, index + 1), daemon=True)
        print(f'Tello {index + 1} Battery: {tello.get_battery()}')
        tello_video_new.start()
        videoThreads.append(tello_video_new)
    for t in videoThreads:
        t.join()

if __name__ == '__main__':
    main()
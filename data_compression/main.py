import cv2
import time, logging
import requests
from djitellopy import Tello, TelloSwarm
import threading
import socketio

sio = socketio.Client()

def setup():
    listIp = ["192.168.0.102", "192.168.0.103"]  # Add more IPs as needed
    telloSwarm = TelloSwarm.fromIps(listIp)
    for index, tello in enumerate(telloSwarm.tellos):
        tello.LOGGER.setLevel(logging.ERROR)
        tello.connect()
        tello.change_vs_udp(8881+index)
        tello.set_video_resolution(Tello.RESOLUTION_480P)
        tello.set_video_bitrate(Tello.BITRATE_1MBPS)
    return telloSwarm

@sio.event
def connect():
    print('Connection established')

@sio.event
def disconnect():
    print('Disconnected from server')

def send_frame(tello, drone_number):
    while True:
        frame = tello.get_frame_read().frame
        _, buffer = cv2.imencode('.jpg', frame)
        sio.emit('frame', {'drone_id': drone_number, 'frame': buffer.tobytes()})
        time.sleep(0.1)

def main():
    sio.connect('http://localhost:5000')
    telloSwarm = setup()
    videoThreads = []
    for index, tello in enumerate(telloSwarm.tellos):
        tello_video_new = threading.Thread(target=send_frame, args=(tello, index+1), daemon=True)
        tello_video_new.start()
        videoThreads.append(tello_video_new)
    for t in videoThreads:
        t.join()

if __name__ == '__main__':
    main()

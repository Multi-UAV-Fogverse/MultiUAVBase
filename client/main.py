from flask import Flask, render_template, jsonify, Response
import requests
from threading import Thread
import time
import eventlet
import eventlet.wsgi

eventlet.monkey_patch()

app = Flask(__name__)

frame_store = {}

def fetch_frame(drone_id):
    while True:
        try:
            response = requests.get(f'http://base_executor:5000/get_frame/{drone_id}')
            if response.status_code == 200:
                frame_store[drone_id] = response.content
                print(f"Fetched frame for drone {drone_id}")
            else:
                print(f"Failed to get frame for drone {drone_id}: {response.text}")
        except Exception as e:
            print(f"Error fetching frame for drone {drone_id}: {e}")
        time.sleep(1)

@app.route('/')
def control_center():
    response = requests.get('http://base_executor:5000/get_drones')
    if response.status_code == 200:
        uav_list = response.json()
    else:
        uav_list = []
    return render_template('control_center.html', uav_list=uav_list)

@app.route('/<int:drone_id>')
def index(drone_id):
    if drone_id not in frame_store:
        Thread(target=fetch_frame, args=(drone_id,), daemon=True).start()
    return render_template('index.html', drone_id=drone_id)

@app.route('/get_frame/<int:drone_id>', methods=['GET'])
def get_frame(drone_id):
    if drone_id in frame_store:
        return Response(frame_store[drone_id], content_type='image/jpeg')
    else:
        return 'No frame', 404

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 5001)), app)
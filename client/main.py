from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import requests

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def control_center():
    response = requests.get('http://process_frame:5000/get_drones')
    if response.status_code == 200:
        uav_list = response.json()
    else:
        uav_list = []
    return render_template('control_center.html', uav_list=uav_list)

@app.route('/<int:drone_id>')
def index(drone_id):
    return render_template('index.html', drone_id=drone_id)

@socketio.on('processed_frame')
def handle_processed_frame(data):
    emit('frame_update', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
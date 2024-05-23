from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import yaml

app = Flask(__name__)
socketio = SocketIO(app)

def page_not_found(*args):
  return render_template('404.html'), 404

@app.route('/')
def control_center():
    total_uav = get_total_uav()
    return render_template('control_center.html', uav_list=total_uav)

@app.route('/<uav_ip>/')
def index(uav_ip=None):
    if not uav_ip:
        return page_not_found()
    return render_template('uav.html', uav_ip=uav_ip)

@socketio.on('<uav_ip>')
def handle_video_frame(data):
    print(f"Received frame for {uav_ip}")  # Debug statement
    emit('video_frame', {'uav_ip': uav_ip, 'frame': data}, broadcast=True)

def get_total_uav():
    # Open the YAML file
    with open('config.yaml', 'r') as file:
        # Load the YAML data into a Python object
        data = yaml.safe_load(file)

    # Access the data
    return data['DRONE_IP']

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

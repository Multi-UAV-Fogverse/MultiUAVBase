from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import eventlet
import psutil, threading, os, time, logging
from fogverse.util import get_timestamp_str, get_timestamp, timestamp_to_datetime
from fogverse.fogverse_logging import FogVerseLogging
import requests

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for WebSocket

frame_store = {}
cpu_usage = 0

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    global cpu_usage

    while True:
        cpu_usage = process.cpu_percent(interval=interval) / psutil.cpu_count()
        time.sleep(interval)

@app.route('/receive_frame/<string:drone_id>', methods=['POST'])
def receive_frame(drone_id):
    data = request.json
    frame_base64 = data['frame']
    frame_store[drone_id] = frame_base64
    global cpu_usage

    # Setup FogVerse logger
    uav_headers_logger = ["uav_id", "frame_id", "cpu_usage", "memory_usage", "gpu_memory_reserved", "gpu_memory_allocated", "input_timestamp", "client_timestamp", "latency"]
    fogverse_uav_logger = FogVerseLogging(
        name=f'uav_{drone_id}_scenario_2',
        dirname="uav-logs",
        csv_header=uav_headers_logger,
        level=logging.INFO + 2
    )

    # Emit the frame via WebSocket
    socketio.emit('frame_update', {'drone_id': drone_id, 'frame': frame_base64})

    # Logging
    process = psutil.Process(os.getpid())

    data = {key: value for key, value in data.items()}
    client_timestamp = get_timestamp()
    input_timestamp = timestamp_to_datetime(data['input_timestamp'])
    latency = client_timestamp - input_timestamp
    cpu_usage_total = float(data['input_cpu_usage']) + float(data['executor_cpu_usage']) + float(cpu_usage)
    memory_usage_total = float(data['input_memory_usage']) + float(data['executor_memory_usage']) + float(process.memory_info().rss / 1024 / 1024)
    
    frame_log = [
        data['uav_id'], 
        data['frame_id'],
        cpu_usage_total, 
        memory_usage_total, 
        data["executor_gpu_memory_reserved"],
        data["executor_gpu_memory_allocated"],
        data['input_timestamp'],
        get_timestamp_str(date=client_timestamp),
        latency
    ]

    logger.info(f"Received frame: {frame_log}")
    fogverse_uav_logger.csv_log(frame_log)

    return jsonify({'status': 'Frame received'}), 200

@app.route('/')
def control_center():
    uav_list = ['1','2','3','4']
    return render_template('control_center.html', uav_list=uav_list)

@app.route('/<int:drone_id>')
def index(drone_id):
    return render_template('index.html', drone_id=drone_id)

if __name__ == '__main__':
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    socketio.run(app, host='0.0.0.0', port=5001)



# def fetch_frame(drone_id):
#     process = psutil.Process(os.getpid())
#     monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
#     monitor_thread.start()
#     global cpu_usage

#     uav_headers_logger = ["uav_id", "frame_id", "cpu_usage", "memory_usage", "gpu_memory_reserved", "gpu_memory_allocated", "input_timestamp", "client_timestamp", "latency"]
#     fogverse_uav_logger = FogVerseLogging(
#         name=f'uav_{drone_id}_scenario_2',
#         dirname="uav-logs",
#         csv_header=uav_headers_logger,
#         level=logging.INFO + 2
#     )
#     while True:
#         try:
#             response = requests.get(f'http://base_executor:5000/get_frame/{drone_id}')
#             if response.status_code == 200:
#                 frame_store[drone_id] = response.content
#                 print(f"Fetched frame for drone {drone_id}")

#                 # Logging
#                 headers = response.headers
#                 headers = {key: value for key, value in headers.items()}
#                 client_timestamp = get_timestamp()
#                 input_timestamp = timestamp_to_datetime(headers['input_timestamp'])
#                 latency = client_timestamp - input_timestamp
#                 cpu_usage_total = float(headers['input_cpu_usage']) + float(headers['executor_cpu_usage']) + float(cpu_usage)
#                 memory_usage_total = float(headers['input_memory_usage']) + float(headers['executor_memory_usage']) + float(process.memory_info().rss / 1024 / 1024)
#                 frame_log = [
#                     headers['uav_id'], 
#                     headers['frame_id'],
#                     cpu_usage_total, 
#                     memory_usage_total, 
#                     headers["executor_gpu_memory_reserved"],
#                     headers["executor_gpu_memory_allocated"],
#                     headers['input_timestamp'],
#                     get_timestamp_str(date=client_timestamp),
#                     latency
#                 ]
#                 fogverse_uav_logger.csv_log(frame_log)

#                 # Emit the frame via WebSocket
#                 socketio.emit('frame_update', {'drone_id': drone_id, 'frame': response.content})
#             else:
#                 print(f"Failed to get frame for drone {drone_id}: {response.text}")
#         except Exception as e:
#             print(f"Error fetching frame for drone {drone_id}: {e}")
#         time.sleep(0.033)
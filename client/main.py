from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import eventlet
import psutil, threading, os, time, logging, csv
from fogverse.util import get_timestamp_str, get_timestamp, timestamp_to_datetime
from concurrent.futures import ThreadPoolExecutor

eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for WebSocket

frame_store = {}
cpu_usage = 0

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# Thread pool executor for non-blocking file operations
executor = ThreadPoolExecutor(max_workers=4)

def setup_csv_logging(uav_id):
    # CSV file setup
    csv_file_path = f'logs/log_csv_uav_{uav_id}_scenario_1.csv'
    csv_headers = ["uav_id", "frame_id", "cpu_usage", "memory_usage", "gpu_memory_reserved", "gpu_memory_allocated", "input_timestamp", "client_timestamp", "latency"]

    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Create CSV file if it doesn't exist and write the header
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
    
    return csv_file_path, csv_headers

def append_to_csv(csv_file_path, frame_log):
    # Append log to CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(frame_log)

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

    # Get CSV file path and headers
    csv_file_path, _ = setup_csv_logging(data['uav_id'])

    # Append log to CSV file in a separate thread
    executor.submit(append_to_csv, csv_file_path, frame_log)

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
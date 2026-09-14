import sys
import cv2
import os
import glob
import time
import json
import requests
from threading import Thread, Lock
from flask import Flask, render_template_string, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Parse camera index from terminal (default: 9)
CAMERA_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 9

# Configuration Parameters
CONFIDENCE_THRESHOLD = 0.50
REQUIRED_STREAK = 3
MAX_STORAGE_FILES = 10
STORAGE_DIR = "captured_events"

BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"
QUEUE_FILE = "offline_alert_queue.json"

os.makedirs(STORAGE_DIR, exist_ok=True)

# State Variables
lock = Lock()
streak_count = 0
last_alert_time = 0
ALERT_COOLDOWN = 10
latest_logs = []

model = YOLO('yolov8n.pt')

def add_log(msg):
    global latest_logs
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    with lock:
        latest_logs.append(entry)
        if len(latest_logs) > 15:
            latest_logs.pop(0)

# FIFO Storage Management
def enforce_fifo_storage():
    files = sorted(glob.glob(os.path.join(STORAGE_DIR, "*.jpg")), key=os.path.getmtime)
    while len(files) > MAX_STORAGE_FILES:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file)
            add_log(f"FIFO Purge: Removed old image {os.path.basename(oldest_file)}")
        except Exception as e:
            add_log(f"FIFO Error: {e}")

# Connectivity & Sync Engine
def check_internet(url="https://api.telegram.org", timeout=3):
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except Exception:
        return False

def send_alert_payload(alert):
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        add_log(f"[SIMULATED ALERT] {alert['message']}")
        return True

    img_path = alert.get("image_path")
    try:
        if img_path and os.path.exists(img_path):
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(img_path, 'rb') as photo:
                res = requests.post(url, data={'chat_id': CHAT_ID, 'caption': alert["message"]}, files={'photo': photo}, timeout=5)
        else:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            res = requests.post(url, data={'chat_id': CHAT_ID, 'text': alert["message"]}, timeout=5)
        return res.status_code == 200
    except Exception as e:
        add_log(f"Dispatch failure: {e}")
        return False

def queue_offline_alert(alert):
    queue = []
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, 'r') as f:
                queue = json.load(f)
        except Exception:
            queue = []
    queue.append(alert)
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f)
    add_log("Offline mode: Alert queued locally in database index.")

def background_sync_worker():
    while True:
        if check_internet() and os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, 'r') as f:
                    queue = json.load(f)
                if queue:
                    add_log(f"Network Active: Syncing {len(queue)} offline alert(s)...")
                    remaining = []
                    for item in queue:
                        if send_alert_payload(item):
                            add_log(f"Synced queued alert from {item['timestamp']}")
                        else:
                            remaining.append(item)
                    with open(QUEUE_FILE, 'w') as f:
                        json.dump(remaining, f)
            except Exception as e:
                add_log(f"Sync Engine Error: {e}")
        time.sleep(10)

Thread(target=background_sync_worker, daemon=True).start()

# Video Stream Generator
def generate_frames():
    global streak_count, last_alert_time
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]
        person_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                person_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if person_detected:
            streak_count += 1
        else:
            streak_count = 0

        cv2.putText(frame, f"Streak: {streak_count}/{REQUIRED_STREAK}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if streak_count >= REQUIRED_STREAK:
            cv2.putText(frame, "INTRUSION CONFIRMED!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            now = time.time()
            if (now - last_alert_time) > ALERT_COOLDOWN:
                last_alert_time = now
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(STORAGE_DIR, f"intruder_{timestamp_str}.jpg")
                
                cv2.imwrite(img_path, frame)
                enforce_fifo_storage()

                alert_data = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"🚨 INTRUSION CONFIRMED!\nStreak: {streak_count} consecutive frames.",
                    "image_path": img_path
                }

                if check_internet():
                    add_log("Sending real-time Telegram push notification...")
                    Thread(target=send_alert_payload, args=(alert_data,)).start()
                else:
                    queue_offline_alert(alert_data)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart IDS Portal</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #fff; text-align: center; margin: 20px; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
        .card { background: #1e1e1e; padding: 15px; border-radius: 8px; }
        #logs { text-align: left; background: #000; color: #00ff00; padding: 10px; font-family: monospace; height: 250px; overflow-y: scroll; }
    </style>
</head>
<body>
    <h1>Smart Intrusion Detection System</h1>
    <div class="container">
        <div class="card">
            <h3>Live Detection Stream</h3>
            <img src="/video_feed" width="640" height="480">
        </div>
        <div class="card" style="width: 400px;">
            <h3>System Activity Logs</h3>
            <div id="logs">Loading logs...</div>
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/api/logs').then(r => r.json()).then(data => {
                document.getElementById('logs').innerHTML = data.logs.join('<br>');
            });
        }, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    with lock:
        return jsonify({"logs": latest_logs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

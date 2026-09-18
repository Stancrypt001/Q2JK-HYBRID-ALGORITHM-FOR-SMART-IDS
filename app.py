import sys
import cv2
import numpy as np
import os
import glob
import time
import json
import requests
from threading import Thread, Lock
from flask import Flask, render_template_string, Response, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# System State & Locks
lock = Lock()
cap = None
current_camera_index = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0

# Configuration Parameters
CONFIDENCE_THRESHOLD = 0.50
REQUIRED_STREAK = 3
MAX_STORAGE_FILES = 10
STORAGE_DIR = "captured_events"

BOT_TOKEN = "8972666318:AAEGqEo1GBUcOJJe1E6OVydRbylC2Ho-fRU"
CHAT_ID = "-1004435269123"
QUEUE_FILE = "offline_alert_queue.json"

os.makedirs(STORAGE_DIR, exist_ok=True)

# Intrusion Tracker State
streak_count = 0
last_alert_time = 0
ALERT_COOLDOWN = 10
latest_logs = []

# Load YOLO Model
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

# --- SAFE CAMERA ENGINE & SCANNING (WITH TTL CACHING) ---

def release_camera_safely():
    """Safely releases the active VideoCapture object without crashing."""
    global cap
    if cap is not None:
        try:
            if cap.isOpened():
                cap.release()
        except Exception as e:
            add_log(f"Camera Release Error: {e}")
        finally:
            cap = None

def get_camera_stream(index):
    """Attempts MSMF backend first (ideal for Windows & Iriun), then falls back."""
    release_camera_safely()
    capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not capture.isOpened():
        capture = cv2.VideoCapture(index)
    return capture

def get_available_cameras(max_tested=10):
    """Scans hardware and virtual indices safely using MSMF."""
    active_cams = []
    add_log("Scanning for active camera feeds...")
    for index in range(max_tested):
        temp_cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(index)
            
        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None and frame.size > 0:
                mean_val = np.mean(frame)
                std_val = np.std(frame)
                if mean_val > 5 and std_val > 2:
                    active_cams.append({
                        "index": index,
                        "status": "Active Feed",
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "brightness": round(float(mean_val), 1)
                    })
            temp_cap.release()
    return active_cams

# Simple TTL Camera Cache to prevent freezing during frequent camera polling
_camera_cache = {"data": None, "ts": 0}

def get_available_cameras_cached(max_tested=10, ttl=10):
    """Cache camera scans for 10 seconds to avoid repeating expensive hardware probes."""
    now = time.time()
    if _camera_cache["data"] is None or (now - _camera_cache["ts"]) > ttl:
        _camera_cache["data"] = get_available_cameras(max_tested)
        _camera_cache["ts"] = now
    return _camera_cache["data"]

# --- STORAGE & TELEGRAM ALERT ENGINE ---

def enforce_fifo_storage():
    files = sorted(glob.glob(os.path.join(STORAGE_DIR, "*.jpg")), key=os.path.getmtime)
    while len(files) > MAX_STORAGE_FILES:
        oldest_file = files.pop(0)
        try:
            os.remove(oldest_file)
            add_log(f"FIFO Purge: Removed old image {os.path.basename(oldest_file)}")
        except Exception as e:
            add_log(f"FIFO Error: {e}")

def check_internet(url="https://api.telegram.org"):
    try:
        return requests.get(url, timeout=3).status_code == 200
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
                res = requests.post(url, data={'chat_id': CHAT_ID, 'caption': alert["message"]}, files={'photo': photo})
        else:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            res = requests.post(url, data={'chat_id': CHAT_ID, 'text': alert["message"]})
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

# --- STREAM GENERATOR WITH YOLO DETECTION ---

def generate_frames():
    global streak_count, last_alert_time, current_camera_index, cap

    if cap is None or not cap.isOpened():
        cap = get_camera_stream(current_camera_index)
        add_log(f"Initialized stream on Camera Index {current_camera_index}")

    while True:
        if cap is not None and cap.isOpened():
            try:
                success, frame = cap.read()
                if not success or frame is None:
                    cv2.waitKey(10)
                    continue

                # Run YOLO Inference
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
                            add_log("Dispatching Telegram push notification...")
                            Thread(target=send_alert_payload, args=(alert_data,)).start()
                        else:
                            queue_offline_alert(alert_data)

                # Encode Frame for Web Display
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                add_log(f"Frame Processing Exception: {e}")
                break
        else:
            time.sleep(0.1)

# --- WEB UI & ROUTING ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🛡️ Smart IDS Portal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e14;
            --bg-secondary: #121721;
            --bg-card: #161b26;
            --bg-elevated: #1c2230;
            --border-color: #232a3a;
            --border-hover: #2f3849;
            --accent: #00d9ff;
            --accent-glow: rgba(0, 217, 255, 0.15);
            --success: #00e676;
            --warning: #ffb300;
            --danger: #ff3b5c;
            --text-primary: #e6edf3;
            --text-secondary: #8b98a9;
            --text-muted: #5a6678;
            --shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            background-image:
                radial-gradient(circle at 10% 0%, rgba(0, 217, 255, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 100%, rgba(255, 59, 92, 0.06) 0%, transparent 40%);
            background-attachment: fixed;
        }

        header {
            padding: 24px 40px;
            border-bottom: 1px solid var(--border-color);
            background: rgba(18, 23, 33, 0.7);
            backdrop-filter: blur(20px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .header-content {
            max-width: 1500px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }

        .logo { display: flex; align-items: center; gap: 14px; }
        .logo-icon {
            width: 44px; height: 44px; border-radius: 12px;
            background: linear-gradient(135deg, var(--accent), #0077ff);
            display: flex; align-items: center; justify-content: center;
            font-size: 22px; box-shadow: 0 0 24px var(--accent-glow);
        }

        .logo-text h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }
        .logo-text p { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }

        .status-pill {
            display: flex; align-items: center; gap: 10px;
            padding: 8px 16px; background: var(--bg-elevated);
            border: 1px solid var(--border-color); border-radius: 100px;
            font-size: 13px; font-weight: 500;
        }

        .status-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--success); box-shadow: 0 0 12px var(--success);
            animation: pulse 2s infinite;
        }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

        .main-container {
            max-width: 1500px; margin: 0 auto; padding: 32px 40px;
            display: grid; grid-template-columns: 1fr 420px; gap: 24px; align-items: start;
        }

        @media (max-width: 1100px) {
            .main-container { grid-template-columns: 1fr; padding: 20px; }
            header { padding: 16px 20px; }
        }

        .card {
            background: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 16px; overflow: hidden; box-shadow: var(--shadow);
        }

        .card-header {
            padding: 18px 22px; border-bottom: 1px solid var(--border-color);
            display: flex; align-items: center; justify-content: space-between; background: var(--bg-secondary);
        }

        .card-title { display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 600; }

        .controls-bar {
            display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
            padding: 16px 22px; background: var(--bg-secondary); border-bottom: 1px solid var(--border-color);
        }

        .control-group { display: flex; align-items: center; gap: 8px; flex: 1; min-width: 240px; }
        .control-group label { font-size: 12px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; }

        select {
            flex: 1; padding: 10px 14px; background: var(--bg-elevated); color: var(--text-primary);
            border: 1px solid var(--border-color); border-radius: 10px; font-size: 13px; outline: none;
        }

        button {
            padding: 10px 20px; background: linear-gradient(135deg, var(--accent), #0077ff);
            color: #fff; border: none; border-radius: 10px; font-size: 13px; font-weight: 600; cursor: pointer;
        }

        .card-body { padding: 22px; }

        .video-wrapper {
            position: relative; background: #000; border-radius: 12px;
            overflow: hidden; aspect-ratio: 4 / 3; border: 1px solid var(--border-color);
        }

        .video-wrapper img { width: 100%; height: 100%; object-fit: cover; display: block; }

        .badge {
            padding: 6px 12px; border-radius: 8px; font-size: 11px; font-weight: 700; text-transform: uppercase;
        }
        .badge-live { background: rgba(255, 59, 92, 0.9); color: #fff; position: absolute; top: 12px; left: 12px; }
        .badge-cam { background: rgba(0, 0, 0, 0.6); color: var(--accent); border: 1px solid rgba(0, 217, 255, 0.3); }

        .logs-container {
            background: #05070a; border-radius: 12px; border: 1px solid var(--border-color);
            padding: 16px; height: 520px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 12px;
        }

        .log-entry { padding: 6px 10px; border-radius: 6px; margin-bottom: 2px; color: #7dd3a0; }
        .log-entry.error { color: #ff6b8a; background: rgba(255, 59, 92, 0.05); }
        .log-entry.warning { color: #ffd54f; background: rgba(255, 179, 0, 0.05); }

        .stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }
        .stat-box { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; padding: 12px; text-align: center; }
        .stat-label { font-size: 10px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }
        .stat-value { font-size: 18px; font-weight: 700; color: var(--accent); font-family: 'JetBrains Mono', monospace; }

        footer { text-align: center; padding: 30px; color: var(--text-muted); font-size: 12px; margin-top: 40px; }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo">
                <div class="logo-icon">🛡️</div>
                <div class="logo-text">
                    <h1>Smart IDS Portal</h1>
                    <p>AI-Powered Intrusion Detection System</p>
                </div>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span id="systemStatus">System Online</span>
            </div>
        </div>
    </header>

    <main class="main-container">
        <section class="card">
            <div class="card-header">
                <div class="card-title"><span>📹 Live Detection Stream</span></div>
                <span class="badge badge-cam" id="activeCamBadge">CAM 0</span>
            </div>

            <div class="controls-bar">
                <div class="control-group">
                    <label>Feed</label>
                    <select id="cameraSelect">
                        <option value="">Scanning devices...</option>
                    </select>
                </div>
                <button onclick="changeCamera()">🔄 Switch Feed</button>
            </div>

            <div class="card-body">
                <div class="video-wrapper">
                    <span class="badge badge-live">LIVE</span>
                    <img id="streamImg" src="/video_feed" alt="Live Stream">
                </div>

                <div class="stats-row">
                    <div class="stat-box">
                        <div class="stat-label">Active Cam</div>
                        <div class="stat-value" id="statCamera">0</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Streak</div>
                        <div class="stat-value" id="statStreak">0/3</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Storage</div>
                        <div class="stat-value" id="statStorage">0/10</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Queue</div>
                        <div class="stat-value" id="statQueue">0</div>
                    </div>
                </div>
            </div>
        </section>

        <section class="card">
            <div class="card-header">
                <div class="card-title"><span>📋 Activity Logs</span></div>
                <span class="badge badge-cam" id="logCount">0</span>
            </div>
            <div class="card-body">
                <div class="logs-container" id="logs"></div>
            </div>
        </section>
    </main>

    <footer>Smart Intrusion Detection System &middot; Powered by YOLOv8 + OpenCV</footer>

    <script>
        function loadCameras() {
            fetch('/api/cameras')
                .then(res => res.json())
                .then(data => {
                    const select = document.getElementById('cameraSelect');
                    select.innerHTML = '';
                    if (!data.cameras || data.cameras.length === 0) {
                        select.innerHTML = '<option value="0">No Active Feeds (Default: 0)</option>';
                    } else {
                        data.cameras.forEach(cam => {
                            const opt = document.createElement('option');
                            opt.value = cam.index;
                            opt.textContent = `Camera ${cam.index} · ${cam.resolution}` + (cam.index === data.current ? '  ✓ ACTIVE' : '');
                            if (cam.index === data.current) opt.selected = true;
                            select.appendChild(opt);
                        });
                    }
                    document.getElementById('statCamera').textContent = data.current;
                    document.getElementById('activeCamBadge').textContent = 'CAM ' + data.current;
                });
        }

        function changeCamera() {
            const index = document.getElementById('cameraSelect').value;
            fetch('/api/set_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ camera_index: parseInt(index) })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('streamImg').src = '/video_feed?t=' + new Date().getTime();
                    loadCameras();
                }
            });
        }

        function fetchStats() {
            fetch('/api/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('statStreak').textContent = `${data.streak}/${data.required_streak}`;
                    document.getElementById('statStorage').textContent = `${data.storage_count}/${data.max_storage}`;
                    document.getElementById('statQueue').textContent = data.queue_size;
                });
        }

        function updateLogs() {
            fetch('/api/logs')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('logs');
                    const logs = data.logs || [];
                    document.getElementById('logCount').textContent = logs.length;
                    container.innerHTML = logs.map(line => `<div class="log-entry">${line}</div>`).join('');
                });
        }

        loadCameras();
        updateLogs();
        fetchStats();

        setInterval(updateLogs, 1000);
        setInterval(fetchStats, 1000);
        setInterval(loadCameras, 10000);
    </script>
</body>
</html>
"""

# --- ROUTES ---

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

@app.route('/api/stats')
def get_stats():
    """Exposes live stats for the UI dashboard."""
    global streak_count, current_camera_index
    with lock:
        queue_len = 0
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, 'r') as f:
                    queue_len = len(json.load(f))
            except Exception:
                queue_len = 0

        return jsonify({
            "streak": streak_count,
            "required_streak": REQUIRED_STREAK,
            "camera": current_camera_index,
            "storage_count": len(glob.glob(os.path.join(STORAGE_DIR, "*.jpg"))),
            "max_storage": MAX_STORAGE_FILES,
            "queue_size": queue_len,
        })

@app.route('/api/cameras')
def get_cameras():
    cams = get_available_cameras_cached()
    return jsonify({"cameras": cams, "current": current_camera_index})

@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    global current_camera_index, cap
    data = request.get_json() or {}
    new_index = data.get('camera_index')

    if new_index is not None:
        try:
            new_index = int(new_index)
            release_camera_safely()
            current_camera_index = new_index
            cap = get_camera_stream(current_camera_index)
            add_log(f"Switched active video stream to Camera Index: {current_camera_index}")
            return jsonify({"success": True, "active": current_camera_index})
        except Exception as e:
            add_log(f"Failed to switch camera: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False, "message": "Missing camera_index"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
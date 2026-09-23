import os
import sys
import glob
import time
import json
import threading
import requests
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

# --- ENVIRONMENT & CONFIGURATION ---

def load_env_file(filepath=".env"):
    """Lightweight .env loader without requiring external third-party dotenv package."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception as e:
            print(f"[WARN] Could not parse .env file: {e}")

load_env_file()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
REQUIRED_STREAK = int(os.getenv("REQUIRED_STREAK", 3))
ALERT_COOLDOWN = float(os.getenv("ALERT_COOLDOWN", 10))
MAX_STORAGE_FILES = int(os.getenv("MAX_STORAGE_FILES", 10))
DEFAULT_CAM_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else int(os.getenv("DEFAULT_CAMERA_INDEX", 0))

STORAGE_DIR = "captured_events"
QUEUE_FILE = "offline_alert_queue.json"
os.makedirs(STORAGE_DIR, exist_ok=True)

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__, template_folder="templates")

# Synchronization Locks
state_lock = threading.Lock()
queue_lock = threading.Lock()
camera_lock = threading.Lock()

# Global State
latest_logs = []

def add_log(msg):
    global latest_logs
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    with state_lock:
        latest_logs.append(entry)
        if len(latest_logs) > 25:
            latest_logs.pop(0)

# --- STORAGE & OFFLINE QUEUE HARDENING ---

def get_queued_image_paths():
    """Returns set of image paths currently referenced in offline queue to prevent premature deletion."""
    with queue_lock:
        if not os.path.exists(QUEUE_FILE):
            return set()
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                items = json.load(f)
                return {item.get("image_path") for item in items if item.get("image_path")}
        except Exception:
            return set()

def enforce_fifo_storage():
    """Enforces storage ceiling while protecting any image currently pending in offline queue."""
    queued_paths = get_queued_image_paths()
    files = sorted(glob.glob(os.path.join(STORAGE_DIR, "*.jpg")), key=os.path.getmtime)
    
    deletable_files = [f for f in files if os.path.abspath(f) not in {os.path.abspath(p) for p in queued_paths}]
    
    while len(files) > MAX_STORAGE_FILES and deletable_files:
        oldest_file = deletable_files.pop(0)
        files.remove(oldest_file)
        try:
            os.remove(oldest_file)
            add_log(f"FIFO Purge: Removed old snapshot {os.path.basename(oldest_file)}")
        except Exception as e:
            add_log(f"FIFO Error: {e}")

def check_internet(url="https://api.telegram.org"):
    try:
        return requests.get(url, timeout=3).status_code == 200
    except Exception:
        return False

def send_alert_payload(alert):
    """Sends Telegram alert with explicit network timeout to prevent thread hangs."""
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        add_log(f"[SIMULATED ALERT] {alert['message']}")
        return True

    img_path = alert.get("image_path")
    try:
        if img_path and os.path.exists(img_path):
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(img_path, 'rb') as photo:
                res = requests.post(
                    url,
                    data={'chat_id': CHAT_ID, 'caption': alert["message"]},
                    files={'photo': photo},
                    timeout=10
                )
        else:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            res = requests.post(
                url,
                data={'chat_id': CHAT_ID, 'text': alert["message"]},
                timeout=10
            )
        return res.status_code == 200
    except Exception as e:
        add_log(f"Dispatch failure: {e}")
        return False

def queue_offline_alert(alert):
    """Thread-safe enqueueing of offline alerts."""
    with queue_lock:
        queue = []
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, 'r', encoding="utf-8") as f:
                    queue = json.load(f)
            except Exception:
                queue = []
        queue.append(alert)
        with open(QUEUE_FILE, 'w', encoding="utf-8") as f:
            json.dump(queue, f, indent=2)
    add_log("Offline mode: Intrusion alert queued locally.")

def background_sync_worker():
    """Background worker for retrying queued offline alerts when network is active."""
    while True:
        try:
            if check_internet() and os.path.exists(QUEUE_FILE):
                with queue_lock:
                    queue = []
                    try:
                        with open(QUEUE_FILE, 'r', encoding="utf-8") as f:
                            queue = json.load(f)
                    except Exception:
                        queue = []

                if queue:
                    add_log(f"Network Active: Syncing {len(queue)} offline alert(s)...")
                    remaining = []
                    for item in queue:
                        if send_alert_payload(item):
                            add_log(f"Synced queued alert from {item.get('timestamp')}")
                        else:
                            remaining.append(item)

                    with queue_lock:
                        with open(QUEUE_FILE, 'w', encoding="utf-8") as f:
                            json.dump(remaining, f, indent=2)

                    # Now that synced files are dispatched, clean up any storage over ceiling
                    enforce_fifo_storage()
        except Exception as e:
            add_log(f"Sync Engine Error: {e}")
        time.sleep(10)

threading.Thread(target=background_sync_worker, daemon=True).start()

# --- DECOUPLED VIDEO CAPTURE & AI PROCESSING ENGINE ---

class VideoProcessingEngine:
    """
    Dedicated background camera capture & YOLO detection engine.
    Eliminates camera device collisions, prevents multi-tab stream contention,
    and isolates AI inference from the Flask HTTP response cycle.
    """
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = True
        self.latest_jpeg = None
        self.streak_count = 0
        self.last_alert_time = 0
        self.frame_width = 0
        self.frame_height = 0
        self.is_camera_active = False

        add_log("Loading YOLOv8 model...")
        self.model = YOLO('yolov8n.pt')
        self._init_camera()

        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def _init_camera(self):
        with camera_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            # Attempt DirectShow first (preferred on Windows), then Media Foundation / default
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if self.cap.isOpened():
                self.is_camera_active = True
                self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                add_log(f"Camera {self.camera_index} initialized ({self.frame_width}x{self.frame_height}).")
            else:
                self.is_camera_active = False
                add_log(f"[WARN] Failed to open Camera {self.camera_index}.")

    def switch_camera(self, new_index):
        with camera_lock:
            if self.camera_index == new_index and self.is_camera_active:
                return True
            self.camera_index = new_index
            self.streak_count = 0
            self.latest_jpeg = None
        self._init_camera()
        return self.is_camera_active

    def _run_loop(self):
        consecutive_drops = 0
        while self.is_running:
            frame = None
            with camera_lock:
                if self.cap is not None and self.cap.isOpened():
                    ret, raw_frame = self.cap.read()
                    if ret and raw_frame is not None and raw_frame.size > 0:
                        frame = raw_frame
                        consecutive_drops = 0
                    else:
                        consecutive_drops += 1
                else:
                    consecutive_drops += 1

            if frame is None:
                # Generate placeholder frame when camera is not ready
                if consecutive_drops >= 10:
                    self.is_camera_active = False
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"CAM {self.camera_index} OFFLINE / CONNECTING...",
                                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, buf = cv2.imencode('.jpg', placeholder)
                    with state_lock:
                        self.latest_jpeg = buf.tobytes()
                    time.sleep(1.0)
                    self._init_camera()
                else:
                    time.sleep(0.05)
                continue

            self.is_camera_active = True
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]

            # 1. Run YOLO Object Detection (Person class = 0)
            try:
                results = self.model(frame, verbose=False)[0]
                person_detected = False

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {conf:.2f}", (x1, max(15, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 2. Update Q2JK Streak Counter
                if person_detected:
                    self.streak_count += 1
                else:
                    self.streak_count = 0

                # 3. Status Overlay
                cv2.putText(frame, f"Streak: {self.streak_count}/{REQUIRED_STREAK}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # 4. Trigger Intrusion Event
                if self.streak_count >= REQUIRED_STREAK:
                    cv2.putText(frame, "INTRUSION CONFIRMED!", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                    now = time.time()
                    if (now - self.last_alert_time) > ALERT_COOLDOWN:
                        self.last_alert_time = now
                        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join(STORAGE_DIR, f"intruder_{timestamp_str}.jpg")

                        # Save snapshot evidence
                        cv2.imwrite(img_path, frame)
                        enforce_fifo_storage()

                        alert_data = {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "message": f"🚨 INTRUSION CONFIRMED!\nCamera: {self.camera_index}\nStreak: {self.streak_count} consecutive frames.",
                            "image_path": img_path
                        }

                        if check_internet():
                            add_log("Dispatching Telegram push notification...")
                            threading.Thread(target=send_alert_payload, args=(alert_data,), daemon=True).start()
                        else:
                            queue_offline_alert(alert_data)

            except Exception as e:
                add_log(f"Inference Loop Error: {e}")

            # 5. Compress to JPEG for Web Stream
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with state_lock:
                    self.latest_jpeg = buf.tobytes()

            # Slight sleep to control CPU load and stabilize frame pacing (~25 FPS)
            time.sleep(0.01)

# Start Video Engine Singleton
engine = VideoProcessingEngine(camera_index=DEFAULT_CAM_INDEX)

# --- SAFE CAMERA HARDWARE DISCOVERY ---

_camera_cache = {"data": None, "ts": 0}

def get_available_cameras(max_tested=8, ttl=15):
    """
    Safely probes connected cameras.
    CRITICAL FIX: Skips the currently active camera index to avoid hardware resource
    lockouts and crashes in Windows / DirectShow / MSMF.
    """
    now = time.time()
    if _camera_cache["data"] is not None and (now - _camera_cache["ts"]) < ttl:
        return _camera_cache["data"]

    active_cams = []
    current_idx = engine.camera_index

    # Report active camera first from engine state without probing
    if engine.is_camera_active:
        active_cams.append({
            "index": current_idx,
            "status": "Active Feed",
            "resolution": f"{engine.frame_width}x{engine.frame_height}",
            "brightness": "Online"
        })

    for index in range(max_tested):
        if index == current_idx:
            continue  # Do not open the device that is already actively capturing!

        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(index)

        if temp_cap.isOpened():
            ret, frame = temp_cap.read()
            if ret and frame is not None and frame.size > 0:
                mean_val = float(np.mean(frame))
                std_val = float(np.std(frame))
                if mean_val > 5 and std_val > 2:
                    active_cams.append({
                        "index": index,
                        "status": "Available",
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                        "brightness": round(mean_val, 1)
                    })
            temp_cap.release()

    _camera_cache["data"] = active_cams
    _camera_cache["ts"] = now
    return active_cams

# --- STREAM GENERATOR FOR FLASK CLIENTS ---

def generate_frames():
    """
    Multi-client safe stream generator.
    Clients simply read the latest JPEG buffer without contending for camera hardware.
    """
    last_frame = None
    while True:
        with state_lock:
            jpeg = engine.latest_jpeg

        if jpeg is not None and jpeg != last_frame:
            last_frame = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        
        # Pacing to ~25-30 FPS per client
        time.sleep(0.035)

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    with state_lock:
        return jsonify({"logs": latest_logs})

@app.route('/api/stats')
def get_stats():
    """Exposes real-time system metrics for the web UI."""
    with queue_lock:
        queue_len = 0
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, 'r', encoding="utf-8") as f:
                    queue_len = len(json.load(f))
            except Exception:
                queue_len = 0

    return jsonify({
        "streak": engine.streak_count,
        "required_streak": REQUIRED_STREAK,
        "camera": engine.camera_index,
        "is_running": engine.is_camera_active,
        "storage_count": len(glob.glob(os.path.join(STORAGE_DIR, "*.jpg"))),
        "max_storage": MAX_STORAGE_FILES,
        "queue_size": queue_len,
    })

@app.route('/api/cameras')
def get_cameras():
    cams = get_available_cameras()
    return jsonify({"cameras": cams, "current": engine.camera_index})

@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    data = request.get_json() or {}
    new_index = data.get('camera_index')

    if new_index is not None:
        try:
            new_index = int(new_index)
            success = engine.switch_camera(new_index)
            # Invalidate cache so UI picks up new active camera status
            _camera_cache["ts"] = 0
            add_log(f"Switched active video stream to Camera Index: {new_index}")
            return jsonify({"success": success, "active": new_index})
        except Exception as e:
            add_log(f"Failed to switch camera: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False, "message": "Missing camera_index"}), 400

if __name__ == "__main__":
    add_log("=== Starting Smart IDS Simulation Server ===")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
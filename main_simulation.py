import sys
import cv2
import time
import requests
from ultralytics import YOLO

# ==========================================
# DYNAMIC CAMERA INDEX SELECTION
# ==========================================
# 1. Check if index was passed as a command-line argument (e.g., python main_simulation.py 9)
if len(sys.argv) > 1:
    try:
        CAMERA_INDEX = int(sys.argv[1])
    except ValueError:
        print(f"[ERROR] Invalid camera index '{sys.argv[1]}'. Defaulting to 0.")
        CAMERA_INDEX = 0
else:
    # 2. Ask user interactively if no argument was passed
    user_input = input("Enter Camera Index (default 9): ").strip()
    CAMERA_INDEX = int(user_input) if user_input.isdigit() else 9

import os

def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

load_env_file()

# ==========================================
# CONFIGURATION & Q2JK PARAMETERS
# ==========================================
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
REQUIRED_STREAK = int(os.getenv("REQUIRED_STREAK", 3))
ALERT_COOLDOWN = float(os.getenv("ALERT_COOLDOWN", 10))

# Telegram Bot Setup
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

def send_telegram_alert(frame_image, message):
    """Sends photo evidence and caption to Telegram."""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print(f"[SIMULATED TELEGRAM ALERT] {message}")
        return
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    _, img_encoded = cv2.imencode('.jpg', frame_image)
    files = {'photo': ('alert.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'chat_id': CHAT_ID, 'caption': message}
    
    try:
        response = requests.post(url, data=data, files=files, timeout=5)
        if response.status_code == 200:
            print("[TELEGRAM] Alert sent successfully!")
        else:
            print(f"[TELEGRAM ERROR] Failed to send: {response.text}")
    except Exception as e:
        print(f"[TELEGRAM NETWORK ERROR] {e}")

# ==========================================
# INITIALIZATION
# ==========================================
print("[SYSTEM] Loading YOLOv8 model...")
model = YOLO('yolov8n.pt')

print(f"[SYSTEM] Opening Camera (Index {CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"[ERROR] Could not open camera at index {CAMERA_INDEX}.")
    print("[TIP] Verify camera connections or try indices 0, 1, 2, 3, or 4.")
    exit()

# Q2JK State Machine Variables
streak_count = 0
intrusion_active = False
last_alert_time = 0
ALERT_COOLDOWN = 10  # Seconds between alert triggers

print(f"\n=== SMART IDS SIMULATION RUNNING ON CAMERA INDEX {CAMERA_INDEX} ===")
print("Press 'q' in the video window to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"[ERROR] Failed to grab frame from camera at index {CAMERA_INDEX}.")
        break

    # 1. Run YOLO Object Detection
    results = model(frame, verbose=False)[0]
    person_detected = False
    
    # 2. Stage 1 Check: Look for Class 0 (Person) with Confidence > Threshold
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls_id == 0 and conf >= CONFIDENCE_THRESHOLD:
            person_detected = True
            # Draw bounding box on frame
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. Stage 2 Check: Q2JK Consecutive Frame Counter
    if person_detected:
        streak_count += 1
    else:
        streak_count = 0
        intrusion_active = False

    # Status Overlay
    status_text = f"Q2JK Streak: {streak_count}/{REQUIRED_STREAK}"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # 4. Trigger Intrusion Event
    if streak_count >= REQUIRED_STREAK:
        cv2.putText(frame, "INTRUSION CONFIRMED!", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        
        current_time = time.time()
        if (current_time - last_alert_time) > ALERT_COOLDOWN:
            print(f"\n[ALERT TRIGGERED] Q2JK Threshold Met! ({streak_count} consecutive frames)")
            send_telegram_alert(frame, "🚨 INTRUSION DETECTED! Confirmed by Q2JK Engine.")
            last_alert_time = current_time

    # Display Video Feed
    cv2.imshow("Smart IDS Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[SYSTEM] Simulation stopped.")

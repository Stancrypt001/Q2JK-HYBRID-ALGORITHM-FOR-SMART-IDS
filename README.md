# 🛡️ Smart Intrusion Detection System (IDS) with YOLO & OpenCV

An AI-powered Smart Intrusion Detection System built with Python, OpenCV, YOLOv8, and Flask. This project processes live video streams, tracks object intrusion streaks, generates real-time visual overlays, maintains local FIFO image storage, and sends instant alert notifications to Telegram (with offline queue support).

---

## 📌 Features

- **Real-time Detection**: Uses YOLOv8 (`yolov8n.pt`) to detect human intrusion across live video feeds.
- **Intrusion Streak Logic**: Confirms intrusion only after a specified number of consecutive detection frames to prevent false alarms.
- **Safe Camera Engine**: Built-in support for multiple cameras and virtual inputs (such as Iriun Webcam) using MSMF (Media Foundation) fallback engine to prevent OpenCV crashes.
- **Telegram Integration**: Direct push notifications with snapshots attached when an intrusion is confirmed.
- **Offline Alert Sync**: Local JSON-based queuing system that automatically retries sending alerts when internet connection is restored.
- **FIFO Storage Management**: Stores captured intrusion frames locally and automatically purges old images beyond a designated limit.
- **Web Dashboard**: Responsive web frontend served via Flask featuring live streaming, status logging, and dynamic camera switching.

---

## 📂 Project Structure

```text
smart_ids_simulation/
├── app.py                      # Main system application and Flask server
├── offline_alert_queue.json    # Auto-generated offline message queue
├── captured_events/            # Directory for saved intrusion snapshots
└── README.md                   # Project documentation
```

---

## 🛠️ Prerequisites & Requirements

Ensure you have the following installed on your system:

- Python 3.8+
- Pip (Python Package Installer)
- Web camera hardware or a virtual webcam application (e.g., Iriun Webcam)

---

## 🚀 Setup & Installation

Follow these steps step-by-step to set up and run the application.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository-name>.git
cd <your-repository-name>
```

### 2. Set Up a Virtual Environment (Recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies

Install all required Python packages inside your active virtual environment:

```bash
pip install opencv-python ultralytics flask requests numpy
```

> **Note:** If running on a headless server without a display interface, you can install `opencv-python-headless` instead of `opencv-python`.

---

## ⚙️ Configuration

Open `app.py` in your code editor to customize system settings:

### Telegram Alert Setup

To receive alert notifications on your phone, replace the placeholders with your credentials:

```python
BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Replace with your Telegram Bot Token
CHAT_ID = "YOUR_CHAT_ID_HERE"      # Replace with your Telegram Chat ID
```

> If left as default, alerts will be simulated in the console logs.

### Detection & Storage Thresholds

```python
CONFIDENCE_THRESHOLD = 0.50   # Minimum YOLO confidence score for person class
REQUIRED_STREAK = 3          # Consecutive frames required to confirm intrusion
ALERT_COOLDOWN = 10          # Cooldown time (in seconds) between alerts
MAX_STORAGE_FILES = 10       # Maximum stored snapshot images (FIFO purge)
```

---

## 🏃 Running the System

Ensure your virtual environment is activated.

Run the application script:

```bash
python app.py
```

**Optional:** You can specify a starting camera index via command line argument (e.g., camera index 1):

```bash
python app.py 1
```

Open your web browser and navigate to:

```text
http://localhost:5000
```

---

## 💻 Web Dashboard Usage

- **Live Stream**: View real-time YOLO object detection bounding boxes and current streak counts.
- **Camera Selector**: The system automatically scans for active local and virtual camera indices. Select a feed from the drop-down menu and click **Switch Feed** to change input on the fly.
- **Activity Logs**: Real-time logging of camera switches, detection events, FIFO purges, and Telegram dispatch updates.

# 🛡️ Smart Intrusion Detection System (IDS) with YOLO & OpenCV

An AI-powered Smart Intrusion Detection System built with Python, OpenCV, YOLOv8, and Flask. This project processes live video streams, tracks object intrusion streaks, generates real-time visual overlays, maintains local FIFO image storage, and sends instant alert notifications to Telegram (with offline queue support).

---

## 📌 Features

* **Real-time Detection**: Uses YOLOv8 (`yolov8n.pt`) to detect human intrusion across live video feeds.
* **Intrusion Streak Logic**: Confirms intrusion only after a specified number of consecutive detection frames to prevent false alarms.
* **Safe Camera Engine**: Built-in support for multiple cameras and virtual inputs (such as Iriun Webcam) using MSMF (Media Foundation) fallback engine to prevent OpenCV crashes.
* **Telegram Integration**: Direct push notifications with snapshots attached when an intrusion is confirmed.
* **Offline Alert Sync**: Local JSON-based queuing system that automatically retries sending alerts when internet connection is restored.
* **FIFO Storage Management**: Stores captured intrusion frames locally and automatically purges old images beyond a designated limit.
* **Web Dashboard**: Responsive web frontend served via Flask featuring live streaming, status logging, and dynamic camera switching.

---

## 📂 Project Structure

```text
smart_ids_simulation/
├── app.py                      # Main system application and Flask server
├── offline_alert_queue.json    # Auto-generated offline message queue
├── captured_events/            # Directory for saved intrusion snapshots
└── README.md                   # Project documentation

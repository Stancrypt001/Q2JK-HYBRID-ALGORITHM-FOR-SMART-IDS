import cv2
import time
import threading

class AutoReconnectCamera:
    """
    Thread-safe camera manager using DirectShow (cv2.CAP_DSHOW) on Windows.
    Automatically handles dropped frames and attempts reconnection on failure.
    """
    def __init__(self, camera_index=0, max_consecutive_failures=5):
        self.camera_index = camera_index
        self.max_failures = max_consecutive_failures
        self.cap = None
        self.lock = threading.Lock()
        self.is_running = True
        self.failed_reads = 0
        self._connect()

    def _connect(self):
        """Attempts to open the camera using DirectShow backend."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            
            # Using CAP_DSHOW avoids MSMF lockups and index out-of-range errors
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if self.cap.isOpened():
                self.failed_reads = 0
                print(f"[INFO] Camera {self.camera_index} initialized successfully.")
            else:
                print(f"[WARN] Failed to open Camera {self.camera_index}.")

    def read_frame(self):
        """Reads a frame with automatic failover and reconnect logic."""
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                self._reconnect()
                return False, None

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.failed_reads += 1
                print(f"[WARN] Dropped frame count: {self.failed_reads}/{self.max_failures}")

                if self.failed_reads >= self.max_failures:
                    print("[ERROR] Max frame drops reached. Triggering camera auto-refresh...")
                    self._reconnect()
                return False, None

            # Reset failure count on a successful frame read
            self.failed_reads = 0
            return True, frame

    def set_camera_index(self, new_index):
        """Switches to a different camera index cleanly."""
        with self.lock:
            self.camera_index = new_index
            self.failed_reads = 0
        self._connect()

    def _reconnect(self):
        """Safely releases the current device handle and attempts reconnection."""
        print(f"[INFO] Reconnecting camera {self.camera_index}...")
        if self.cap:
            self.cap.release()
            self.cap = None
        
        time.sleep(1.0)  # OS time buffer to release hardware resource lock
        self._connect()

    def release(self):
        """Cleanly releases the camera hardware on shutdown."""
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
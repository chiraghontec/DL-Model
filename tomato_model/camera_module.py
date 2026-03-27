#!/usr/bin/env python3
"""
camera_module.py

Non-blocking USB/Pi camera capture for the tomato spraying rover.

Runs a background thread that continuously reads frames so that the main
inference loop always gets the latest available frame without blocking on
camera I/O.  Stale frames are dropped automatically (only the newest is
kept in the ring buffer).

Usage:
    from camera_module import CameraModule

    cam = CameraModule(device_id=0, width=640, height=480, fps=30)
    cam.start()

    frame = cam.read()          # Latest BGR frame, or None if not ready
    annotated = cam.annotate(frame, label="early_blight", conf=0.986,
                             zone="TOP", spray_ms=204, should_spray=True)
    cam.stop()

On Raspberry Pi with libcamera / PiCamera2, set device_id=0 and
OpenCV will use V4L2 (ensure libcamera-v4l2 module is loaded).
"""

import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ─── colour palette (BGR) ──────────────────────────────────────────────────────
_COLOURS = {
    "early_blight": (0,  165, 255),   # orange
    "late_blight":  (0,  0,   255),   # red
    "healthy":      (0,  200, 50),    # green
    "default":      (200, 200, 200),  # grey
}
_ZONE_COLOURS = {
    "TOP":    (255, 80,  80),
    "MID":    (80,  255, 80),
    "BOTTOM": (80,  80,  255),
}


class CameraModule:
    """
    Thread-safe, non-blocking camera reader.

    Attributes:
        device_id (int): OpenCV camera index (0 = first USB cam).
        width, height (int): Capture resolution.
        fps (int): Requested capture framerate.
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        auto_exposure: bool = True,
    ):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_exposure = auto_exposure

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Telemetry
        self._frame_count = 0
        self._drop_count = 0
        self._t_start: float = 0.0

    # ─── Public API ──────────────────────────────────────────────────────────

    def start(self) -> "CameraModule":
        """Open the camera and start the background capture thread."""
        self._cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            # Fallback: try default backend (AVFOUNDATION on macOS, DSHOW on Win)
            self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device_id={self.device_id}. "
                "Check USB connection or device_id in config.json."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if self.auto_exposure:
            # CAP_PROP_AUTO_EXPOSURE: some backends use 0.75 for auto, 0.25 for manual
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        self._running = True
        self._t_start = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait up to 2 s for the first frame
        deadline = time.time() + 2.0
        while self._frame is None and time.time() < deadline:
            time.sleep(0.05)

        if self._frame is None:
            self.stop()
            raise RuntimeError("Camera started but no frame received within 2 s.")

        print(f"[CameraModule] Started: {self.width}×{self.height} @{self.fps}fps "
              f"device={self.device_id}")
        return self

    def stop(self):
        """Stop capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        print(f"[CameraModule] Stopped. Frames captured: {self._frame_count}, "
              f"dropped: {self._drop_count}")

    def read(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame (BGR, HWC uint8), or None if unavailable.

        Non-blocking — always returns immediately.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_open(self) -> bool:
        """Return True if the camera is open and running."""
        return self._running and self._cap is not None and self._cap.isOpened()

    def actual_fps(self) -> float:
        """Return measured capture FPS since start()."""
        elapsed = time.time() - self._t_start
        if elapsed < 0.1:
            return 0.0
        return self._frame_count / elapsed

    def stats(self) -> dict:
        return {
            "frames_captured": self._frame_count,
            "frames_dropped":  self._drop_count,
            "actual_fps":      round(self.actual_fps(), 1),
            "resolution":      f"{self.width}×{self.height}",
        }

    # ─── Annotation helper ────────────────────────────────────────────────────

    @staticmethod
    def annotate(
        frame: np.ndarray,
        class_name: str = "",
        confidence: float = 0.0,
        zone: Optional[str] = None,
        spray_ms: int = 0,
        should_spray: bool = False,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        zone_boundaries_frac: Tuple[float, float] = (0.333, 0.667),
    ) -> np.ndarray:
        """
        Return a copy of frame with inference overlay drawn.

        Args:
            frame: BGR image (H, W, 3).
            class_name: Predicted class label.
            confidence: Softmax confidence [0-1].
            zone: Zone name ("TOP" / "MID" / "BOTTOM") or None.
            spray_ms: Computed spray duration in ms.
            should_spray: Whether the system will actuate the relay.
            bbox: (x, y, w, h) bounding box in pixels, or None.
            zone_boundaries_frac: (upper, lower) Y-fraction zone splits.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw zone boundary lines
        for i, frac in enumerate(zone_boundaries_frac):
            y_line = int(h * frac)
            cv2.line(vis, (0, y_line), (w, y_line), (200, 200, 200), 1, cv2.LINE_AA)
            label = ("TOP", "MID", "BOTTOM")[i]
            cv2.putText(vis, label, (4, y_line - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw bounding box (if disease + GradCAM)
        if bbox is not None and should_spray:
            bx, by, bw, bh = bbox
            colour = _COLOURS.get(class_name, _COLOURS["default"])
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), colour, 2)
            # Zone colour indicator
            if zone:
                zone_col = _ZONE_COLOURS.get(zone, (255, 255, 255))
                cv2.rectangle(vis, (bx, by - 2), (bx + bw, by), zone_col, -1)

        # Top bar background
        bar_h = 50
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

        # Class + confidence
        colour = _COLOURS.get(class_name, _COLOURS["default"])
        cv2.putText(vis, f"{class_name}  {confidence:.0%}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)

        # Zone + spray info
        if should_spray and zone:
            spray_str = f"SPRAY {zone}  {spray_ms}ms"
            cv2.putText(vis, spray_str, (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _ZONE_COLOURS.get(zone, (255, 255, 255)), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "NO SPRAY", (8, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 50), 1, cv2.LINE_AA)

        return vis

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Background thread: grab frames as fast as possible; keep only the newest."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # Camera disconnected; signal main thread
                print("[CameraModule] WARNING: failed to read frame — retrying")
                time.sleep(0.1)
                continue

            self._frame_count += 1
            with self._lock:
                if self._frame is not None:
                    self._drop_count += 1   # previous frame was never consumed
                self._frame = frame

    # ─── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "CameraModule":
        return self.start()

    def __exit__(self, *_):
        self.stop()


# ─── CLI demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CameraModule live demo")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps",    type=int, default=30)
    args = ap.parse_args()

    with CameraModule(args.device, args.width, args.height, args.fps) as cam:
        print("Press 'q' to quit.")
        while True:
            frame = cam.read()
            if frame is None:
                continue
            vis = CameraModule.annotate(frame, "healthy", 0.99)
            cv2.imshow("Camera Feed", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        print(cam.stats())

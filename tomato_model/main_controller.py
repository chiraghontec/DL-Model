#!/usr/bin/env python3
"""
main_controller.py

Main event loop for the Scout-Map-Spray tomato disease rover.

Pipeline per frame:
  1.  CameraModule → grab latest frame (non-blocking)
  2.  InferenceEngine → classify + GradCAM + zone/VRA
  3.  RelayController → open solenoid valve if disease detected
  4.  DataLogger      → append row to CSV log
  5.  CameraModule.annotate → draw overlay on frame
  6.  cv2.imshow (optional) → live preview

Usage examples:
    # Live camera + real spray (on Pi):
    python3 main_controller.py

    # Mock relay + live camera:
    python3 main_controller.py --mock-relay

    # Offline test with static images (no camera):
    python3 main_controller.py --no-camera --test-images test/

    # Headless (no cv2 window):
    python3 main_controller.py --no-display
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# ── Local modules (same directory) ────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from camera_module import CameraModule
from inference_engine import InferenceEngine
from kinematic_mapper import KinematicMapper
from relay_controller import RelayController
from stepper_controller import StepperController

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main_controller")


# ─── Data Logger ──────────────────────────────────────────────────────────────

class DataLogger:
    """Persists per-frame telemetry to a CSV file, flushing every write."""

    COLUMNS = [
        "timestamp", "elapsed_s", "frame_no",
        "class_name", "confidence", "should_spray",
        "y_center_px", "y_target_mm", "steps_commanded",
        "spray_duration_ms",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "inference_ms",
    ]

    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        log.info(f"[DataLogger] Logging to {log_path}")

    def write(self, row: dict):
        self._writer.writerow(row)

    def close(self):
        self._file.flush()
        self._file.close()
        log.info(f"[DataLogger] Log file closed: {self.log_path}")


# ─── Controller ───────────────────────────────────────────────────────────────

class Controller:
    """Orchestrates camera → infer → relay → log pipeline."""

    def __init__(self, args):
        self.args = args
        self._running = False
        self.frame_no = 0
        self.start_time = 0.0

        # Load config
        config_path = args.config
        with open(config_path) as f:
            self.cfg = json.load(f)

        # Components
        self.cam = self._make_camera()
        self.engine = self._make_engine(config_path)
        self.relay = self._make_relay()
        self.stepper = self._make_stepper(config_path)
        self.kinematics = KinematicMapper(config_path)
        self.logger = DataLogger(args.log)

        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    # ── Factory helpers ───────────────────────────────────────────────────────

    def _make_camera(self) -> CameraModule:
        cam_cfg = self.cfg.get("camera", {})
        res = cam_cfg.get("resolution", [640, 480])
        return CameraModule(
            device_id=0,
            width=res[0],
            height=res[1],
            fps=cam_cfg.get("fps", 30),
        )

    def _make_engine(self, config_path: str) -> InferenceEngine:
        engine = InferenceEngine(config_path)
        model_cfg = self.cfg.get("model", {})
        onnx_rel = model_cfg.get("onnx_path", "")
        onnx_abs = str(_HERE / onnx_rel) if onnx_rel else None

        pt_rel = model_cfg.get("checkpoint_path", "")
        pt_abs = str(_HERE / pt_rel) if pt_rel else None

        if onnx_abs and Path(onnx_abs).exists():
            engine.load_onnx_model(onnx_abs)
            log.info(f"[Engine] ONNX model loaded: {onnx_abs}")
        else:
            log.warning("[Engine] ONNX model not found — inference will fail unless PyTorch loaded")

        # PyTorch model needed for GradCAM
        if not self.args.no_gradcam and pt_abs and Path(pt_abs).exists():
            try:
                engine.load_pytorch_model(pt_abs)
                log.info(f"[Engine] PyTorch model loaded: {pt_abs}")
            except Exception as e:
                log.warning(f"[Engine] PyTorch load failed (GradCAM disabled): {e}")
                self.args.no_gradcam = True
        else:
            log.info("[Engine] GradCAM disabled (no PyTorch checkpoint or --no-gradcam)")
            self.args.no_gradcam = True

        return engine

    def _make_relay(self) -> RelayController:
        stepper_cfg = self.cfg.get("stepper", {})
        nozzle_pin = stepper_cfg.get("nozzle_gpio_pin", 17)
        pump_pin = stepper_cfg.get("pump_gpio_pin", 23)
        cooldown = self.cfg.get("vra", {}).get("cooldown_ms", 200)
        return RelayController(
            nozzle_pin=nozzle_pin,
            pump_pin=pump_pin,
            cooldown_ms=cooldown,
        )

    def _make_stepper(self, config_path: str) -> StepperController:
        sc = StepperController(config_path)
        sc.connect()
        return sc

    # ── Image iterator (test mode) ─────────────────────────────────────────────

    def _iter_test_images(self):
        """Yield (frame, path) tuples from --test-images directory."""
        ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        test_dir = Path(self.args.test_images)
        paths = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in ext)
        if not paths:
            log.error(f"No images found in {test_dir}")
            return
        log.info(f"[TestMode] Found {len(paths)} image(s) in {test_dir}")
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                yield img, str(p)
            else:
                log.warning(f"[TestMode] Cannot read {p}")

    # ── Main processing (single frame) ────────────────────────────────────────

    def _process_frame(self, frame) -> dict:
        """
        Run inference on one frame.

        Returns a result dict with at minimum:
          class_name, confidence, should_spray, y_center_px, y_target_mm,
          spray_duration_ms, bbox (tuple or None), inference_ms
        """
        t0 = time.time()
        try:
            result = self.engine.infer(
                frame,
                run_gradcam=not self.args.no_gradcam,
            )
        except Exception as e:
            log.error(f"[Engine] Inference error: {e}")
            result = {
                "class_name": "error",
                "confidence": 0.0,
                "should_spray": False,
                "y_center_px": None,
                "y_target_mm": None,
                "spray_duration_ms": 0,
                "bbox": None,
            }
        result["inference_ms"] = (time.time() - t0) * 1000
        return result

    def _log_row(self, result: dict):
        bbox = result.get("bbox") or (0, 0, 0, 0)
        self.logger.write({
            "timestamp":      datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_s":      round(time.time() - self.start_time, 2),
            "frame_no":       self.frame_no,
            "class_name":     result.get("class_name", ""),
            "confidence":     round(float(result.get("confidence", 0)), 4),
            "should_spray":   result.get("should_spray", False),
            "y_center_px":    result.get("y_center_px", ""),
            "y_target_mm":    result.get("y_target_mm", ""),
            "steps_commanded": result.get("steps_commanded", ""),
            "spray_duration_ms": result.get("spray_duration_ms", 0),
            "bbox_x": bbox[0], "bbox_y": bbox[1],
            "bbox_w": bbox[2], "bbox_h": bbox[3],
            "inference_ms":   round(result.get("inference_ms", 0), 1),
        })

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(self):
        self._running = True
        self.start_time = time.time()

        use_camera = not self.args.no_camera

        # Setup relay and home actuator
        self.relay.setup()
        log.info("[Stepper] Homing actuator …")
        self.stepper.home()
        log.info("[Stepper] Home complete")

        # Open display window (unless headless)
        if not self.args.no_display:
            cv2.namedWindow("Rover Vision", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rover Vision", 800, 600)

        # Start camera (only if needed)
        if use_camera:
            self.cam.start()
            log.info("[Camera] Live capture started")
            time.sleep(0.5)  # warm-up

        frame_times = []

        try:
            if not use_camera:
                # ── Offline test mode ─────────────────────────────────────────
                for frame, img_path in self._iter_test_images():
                    if not self._running:
                        break
                    self.frame_no += 1
                    t_frame = time.time()

                    print(f"\n[Frame {self.frame_no:04d}] {Path(img_path).name}")
                    result = self._process_frame(frame)
                    self._print_result(result)

                    # Stepper positioning then nozzle actuation
                    if result.get("should_spray"):
                        steps = 0
                        y_px = result.get("y_center_px")
                        if y_px is not None:
                            steps = self.kinematics.pixel_to_steps(y_px)
                        result["steps_commanded"] = steps
                        self.stepper.goto(steps)
                        self.relay.spray(result.get("spray_duration_ms", 200))

                    self._log_row(result)

                    # Annotate + display
                    annotated = CameraModule.annotate(
                        frame.copy(),
                        class_name=result.get("class_name", ""),
                        confidence=result.get("confidence", 0.0),
                        zone=f"{result.get('y_target_mm', 0):.0f}mm" if result.get("y_target_mm") else None,
                        spray_ms=result.get("spray_duration_ms", 0),
                        should_spray=result.get("should_spray", False),
                        bbox=result.get("bbox"),
                    )
                    if not self.args.no_display:
                        cv2.imshow("Rover Vision", annotated)
                        key = cv2.waitKey(500)   # 500ms between test images
                        if key == ord("q") or key == 27:
                            log.info("Quit key pressed")
                            break

                    frame_times.append(time.time() - t_frame)

            else:
                # ── Live camera mode ──────────────────────────────────────────
                min_interval = 1.0 / max(self.args.fps, 1)
                last_infer_t = 0.0

                while self._running:
                    frame = self.cam.read()
                    if frame is None:
                        time.sleep(0.01)
                        continue

                    now = time.time()
                    if (now - last_infer_t) < min_interval:
                        # Show last annotated frame at full camera rate
                        if not self.args.no_display and self.frame_no > 0:
                            cv2.imshow("Rover Vision", self._last_annotated)
                            key = cv2.waitKey(1)
                            if key == ord("q") or key == 27:
                                break
                        continue

                    last_infer_t = now
                    self.frame_no += 1
                    t_frame = time.time()
                    
                    # Periodically refresh distance to adjust FOV mapping linearly (assume 1cm = 20mm FOV scale)
                    if self.frame_no % 30 == 0:
                        dist_data = self.stepper.get_distance()
                        front_dist = dist_data.get("front_cm", 15)
                        if front_dist < 100:  # Valid reading
                            # Approx geometry: 15cm distance gives 300mm FOV (ratio 20x)
                            self.kinematics.camera_fov_height_mm = front_dist * 20.0

                    result = self._process_frame(frame)

                    if result.get("should_spray"):
                        steps = 0
                        y_px = result.get("y_center_px")
                        if y_px is not None:
                            steps = self.kinematics.pixel_to_steps(y_px)
                        result["steps_commanded"] = steps
                        self.stepper.goto(steps)
                        self.relay.spray(result.get("spray_duration_ms", 200))

                    self._log_row(result)

                    annotated = CameraModule.annotate(
                        frame.copy(),
                        class_name=result.get("class_name", ""),
                        confidence=result.get("confidence", 0.0),
                        zone=f"{result.get('y_target_mm', 0):.0f}mm" if result.get("y_target_mm") else None,
                        spray_ms=result.get("spray_duration_ms", 0),
                        should_spray=result.get("should_spray", False),
                        bbox=result.get("bbox"),
                    )
                    self._last_annotated = annotated

                    if not self.args.no_display:
                        cv2.imshow("Rover Vision", annotated)
                        key = cv2.waitKey(1)
                        if key == ord("q") or key == 27:
                            log.info("Quit key pressed")
                            break

                    frame_times.append(time.time() - t_frame)

                    # Print summary every 30 frames
                    if self.frame_no % 30 == 0:
                        self._print_stats(frame_times)

        finally:
            self._shutdown(frame_times)

    def _last_annotated(self):
        pass  # placeholder; overwritten at runtime

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self, frame_times):
        log.info("[Controller] Shutting down …")
        self._running = False

        if not self.args.no_camera:
            self.cam.stop()

        self.stepper.stop()
        self.stepper.disconnect()
        self.relay.all_off()
        self.relay.teardown()
        self.logger.close()

        if not self.args.no_display:
            cv2.destroyAllWindows()

        # Final summary
        if frame_times:
            avg_ms = 1000 * sum(frame_times) / len(frame_times)
            print(f"\n{'='*50}")
            print(f"  Session summary")
            print(f"  Frames processed : {self.frame_no}")
            print(f"  Avg frame time   : {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
            print(f"  Relay stats      : {self.relay.stats()}")
            print(f"  Log saved to     : {self.args.log}")
            print(f"{'='*50}\n")

    def _handle_exit(self, signum, frame):
        log.warning(f"Signal {signum} received — stopping …")
        self._running = False

    # ── Console output ────────────────────────────────────────────────────────

    @staticmethod
    def _print_result(result: dict):
        if result.get("should_spray"):
            mm = result.get("y_target_mm")
            steps = result.get("steps_commanded", "?")
            spray_str = (
                f"SPRAY  target={mm:.0f}mm  steps={steps}  "
                f"{result['spray_duration_ms']:.0f}ms"
                if mm is not None
                else f"SPRAY  {result['spray_duration_ms']:.0f}ms"
            )
        else:
            spray_str = "no spray"
        print(
            f"  class={result.get('class_name','?'):<14}"
            f"  conf={result.get('confidence', 0):.3f}"
            f"  {spray_str}"
            f"  infer={result.get('inference_ms', 0):.1f}ms"
        )

    @staticmethod
    def _print_stats(frame_times):
        n = len(frame_times)
        if n < 2:
            return
        recent = frame_times[-30:]
        avg_ms = 1000 * sum(recent) / len(recent)
        print(f"  [stats] frames={n}  avg={avg_ms:.1f}ms  fps={1000/avg_ms:.1f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Tomato rover — Scout-Map-Spray controller"
    )
    ap.add_argument(
        "--config",
        default=str(_HERE / "config.json"),
        help="Path to config.json (default: next to this script)",
    )
    ap.add_argument(
        "--log",
        default=str(_HERE / "logs" / f"spray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"),
        help="CSV log output path",
    )
    ap.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable live camera; use --test-images instead",
    )
    ap.add_argument(
        "--test-images",
        default=str(_HERE / "test"),
        help="Directory of images to run in offline test mode",
    )
    ap.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress cv2.imshow (headless / Pi without monitor)",
    )
    ap.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip GradCAM (faster; classification-only mode)",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=6.0,
        help="Target inference rate in FPS (default: 6.0)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("  Scout-Map-Spray Tomato Disease Controller")
    print("=" * 60)
    print(f"  Config    : {args.config}")
    print(f"  Mode      : {'TEST IMAGE' if args.no_camera else 'LIVE CAMERA'}")
    print(f"  GradCAM   : {'OFF' if args.no_gradcam else 'ON'}")
    print(f"  Display   : {'OFF' if args.no_display else 'ON'}")
    print(f"  Target FPS: {args.fps}")
    print()

    ctrl = Controller(args)
    ctrl.run()

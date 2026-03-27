#!/usr/bin/env python3
"""
stepper_controller.py

Raspberry Pi ↔ Arduino UART bridge for the Z-axis linear actuator.

Sends JSON-framed commands to the Arduino's rover_controller firmware:

  Pi → Arduino   {"cmd": "GOTO", "steps": 4800}
  Arduino → Pi   {"status": "ARRIVED", "steps": 4800}

  Pi → Arduino   {"cmd": "HOME"}
  Arduino → Pi   {"status": "HOMED"}

  Pi → Arduino   {"cmd": "STOP"}
  Arduino → Pi   {"status": "STOPPED", "steps": <current>}

  Pi → Arduino   {"cmd": "POS"}
  Arduino → Pi   {"status": "POS", "steps": <current>}

On non-Pi dev machines (macOS / CI) the serial port is absent, so a
MockStepperController is substituted automatically.

Usage:
    from stepper_controller import StepperController
    sc = StepperController("config.json")
    sc.connect()
    sc.home()
    sc.goto(4800)     # blocks until ARRIVED or timeout
    sc.stop()
"""

import json
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

try:
    import serial  # pyserial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


class StepperController:
    """
    Serial bridge from Pi to Arduino stepper firmware.

    Falls back to mock mode when pyserial is unavailable or the port
    cannot be opened (development / macOS environment).
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        nav = cfg.get("navigation", {})
        stepper = cfg.get("stepper", {})

        self.port: str = nav.get("serial_port", "/dev/ttyUSB0")
        self.baud: int = nav.get("baud_rate", 115200)
        self.goto_timeout_s: float = stepper.get("goto_timeout_s", 10.0)
        self.homing_timeout_s: float = stepper.get("homing_timeout_s", 15.0)
        self.max_steps: int = round(
            stepper.get("travel_mm", 800) * stepper.get("steps_per_mm", 80)
        )

        self._ser: Optional[object] = None
        self._lock = threading.Lock()
        self._current_steps: int = 0
        self._is_mock: bool = False
        self._connected: bool = False

    # ─── Connection ───────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the serial port. Returns True on success, False in mock mode."""
        if not _SERIAL_AVAILABLE:
            log.warning("[StepperController] pyserial not installed — mock mode")
            self._is_mock = True
            self._connected = True
            return False

        try:
            self._ser = serial.Serial(
                self.port,
                baudrate=self.baud,
                timeout=1.0,
            )
            time.sleep(2.0)  # allow Arduino to reboot after DTR reset
            self._ser.reset_input_buffer()
            self._connected = True
            log.info(f"[StepperController] Connected: {self.port} @ {self.baud}")
            return True
        except Exception as exc:
            log.warning(f"[StepperController] Cannot open {self.port}: {exc} — mock mode")
            self._is_mock = True
            self._connected = True
            return False

    def disconnect(self):
        """Close the serial port."""
        if self._ser and hasattr(self._ser, "close"):
            try:
                self._ser.close()
            except Exception:
                pass
        self._connected = False

    # ─── Commands ─────────────────────────────────────────────────────────────

    def home(self) -> bool:
        """
        Drive actuator to home position (end-stop).
        Blocks until HOMED ACK or timeout.
        Returns True on success.
        """
        if not self._connected:
            raise RuntimeError("StepperController not connected. Call connect() first.")

        if self._is_mock:
            log.info("[StepperController] MOCK HOME → position 0")
            self._current_steps = 0
            return True

        self._send({"cmd": "HOME"})
        ack = self._wait_for(["HOMED", "ERROR"], timeout=self.homing_timeout_s)
        if ack and ack.get("status") == "HOMED":
            self._current_steps = 0
            log.info("[StepperController] Homed successfully")
            return True

        log.error(f"[StepperController] Homing failed: {ack}")
        return False

    def goto(self, steps: int, blocking: bool = True) -> bool:
        """
        Move actuator to absolute step position.

        Args:
            steps:    Absolute step count from home (0 = home end).
            blocking: Wait for ARRIVED ACK (default True).

        Returns:
            True if ARRIVED (or fire-and-forget when blocking=False).
        """
        if not self._connected:
            raise RuntimeError("StepperController not connected. Call connect() first.")

        steps = max(0, min(self.max_steps, steps))

        if self._is_mock:
            log.info(f"[StepperController] MOCK GOTO {steps} steps")
            self._current_steps = steps
            return True

        self._send({"cmd": "GOTO", "steps": steps})

        if not blocking:
            return True

        ack = self._wait_for(["ARRIVED", "ERROR"], timeout=self.goto_timeout_s)
        if ack and ack.get("status") == "ARRIVED":
            self._current_steps = ack.get("steps", steps)
            return True

        log.error(f"[StepperController] GOTO failed: {ack}")
        return False

    def stop(self) -> bool:
        """Emergency stop. Blocks until STOPPED ACK."""
        if not self._connected:
            return False

        if self._is_mock:
            log.info("[StepperController] MOCK STOP")
            return True

        self._send({"cmd": "STOP"})
        ack = self._wait_for(["STOPPED", "ERROR"], timeout=2.0)
        if ack:
            self._current_steps = ack.get("steps", self._current_steps)
        return ack is not None

    def get_position(self) -> int:
        """Query current absolute step position from firmware."""
        if self._is_mock:
            return self._current_steps

        self._send({"cmd": "POS"})
        ack = self._wait_for(["POS"], timeout=2.0)
        if ack:
            self._current_steps = ack.get("steps", self._current_steps)
        return self._current_steps

    def get_distance(self) -> dict:
        """Query ultrasound distance in cm from firmware."""
        if self._is_mock:
            return {"front_cm": 15, "rear_cm": 15}
            
        self._send({"cmd": "DIST"})
        ack = self._wait_for(["DIST"], timeout=2.0)
        if ack:
            return {"front_cm": ack.get("front_cm", 999), "rear_cm": ack.get("rear_cm", 999)}
        return {"front_cm": 999, "rear_cm": 999}

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _send(self, payload: dict):
        """Serialize payload as JSON + newline and write to serial port."""
        line = json.dumps(payload) + "\n"
        with self._lock:
            self._ser.write(line.encode("utf-8"))
            self._ser.flush()
        log.debug(f"[StepperController] TX: {payload}")

    def _wait_for(self, expected_statuses: list, timeout: float) -> Optional[dict]:
        """
        Read lines from serial until a JSON response with one of the
        expected_statuses arrives, or timeout expires.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with self._lock:
                    raw = self._ser.readline()
                if not raw:
                    continue
                msg = json.loads(raw.decode("utf-8").strip())
                log.debug(f"[StepperController] RX: {msg}")
                if msg.get("status") in expected_statuses:
                    return msg
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as exc:
                log.error(f"[StepperController] Read error: {exc}")
                break

        log.warning(
            f"[StepperController] Timeout waiting for {expected_statuses} "
            f"({timeout:.1f}s)"
        )
        return None

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def position_steps(self) -> int:
        return self._current_steps

    def __repr__(self) -> str:
        mode = "mock" if self._is_mock else f"serial:{self.port}"
        return f"StepperController({mode}, pos={self._current_steps})"

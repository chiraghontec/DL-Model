#!/usr/bin/env python3
"""
relay_controller.py

Single-channel GPIO relay control for the tomato spraying rover's nozzle.

Controls one 12V solenoid nozzle valve and one diaphragm pump via a
2-channel optocoupler-isolated relay module driven from Raspberry Pi GPIO.

Hardware notes:
  - Active-LOW relay: RPi pulls GPIO LOW → relay coil energised → valve opens
  - Flyback diodes (1N4007) protect against solenoid back-EMF
  - Cooldown (default 200ms) prevents solenoid overheating

On non-Pi systems (dev / macOS / CI), a mock mode is substituted
automatically when RPi.GPIO is unavailable.

Usage:
    from relay_controller import RelayController

    rc = RelayController(nozzle_pin=17, pump_pin=23, cooldown_ms=200)
    rc.setup()
    rc.spray(duration_ms=204)   # non-blocking
    rc.teardown()
"""

import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# ── Try to import real RPi.GPIO; fall back to mock ───────────────────────────
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    _GPIO_AVAILABLE = False


class RelayController:
    """
    Controls single-channel nozzle relay for the spray gantry.

    If Raspberry Pi GPIO is unavailable (dev machine), a software mock is
    used automatically — all API calls succeed and are logged to stdout.

    Args:
        nozzle_pin:  BCM GPIO pin for the nozzle solenoid relay.
        pump_pin:    BCM GPIO pin for the diaphragm pump relay.
        active_low:  True if relay is active-LOW (standard opto-relay board).
        cooldown_ms: Minimum time between successive actuations (ms).
    """

    def __init__(
        self,
        nozzle_pin: int = 17,
        pump_pin: int = 23,
        active_low: bool = True,
        cooldown_ms: int = 200,
    ):
        self.nozzle_pin = nozzle_pin
        self.pump_pin = pump_pin
        self.active_low = active_low
        self.cooldown_ms = cooldown_ms / 1000.0  # convert to seconds

        self._all_pins = [nozzle_pin, pump_pin]
        self._lock = threading.Lock()
        self._last_spray_time: float = 0.0
        self._nozzle_timer: Optional[threading.Timer] = None

        # Telemetry
        self._spray_count: int = 0
        self._total_spray_ms: float = 0.0
        self._is_mock = not _GPIO_AVAILABLE
        self._setup_done = False

        if self._is_mock:
            log.warning("[RelayController] RPi.GPIO unavailable — using mock mode")

    # ─── Setup / Teardown ─────────────────────────────────────────────────────

    def setup(self):
        """Configure GPIO pins as outputs and set to safe (off) state."""
        if self._is_mock:
            print(f"[RelayController] MOCK setup — nozzle_pin={self.nozzle_pin}, "
                  f"pump_pin={self.pump_pin}")
            self._setup_done = True
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in self._all_pins:
            GPIO.setup(pin, GPIO.OUT)
            self._set_pin(pin, False)   # ensure relays are OFF at startup

        self._setup_done = True
        print(f"[RelayController] GPIO setup complete. "
              f"nozzle={self.nozzle_pin}, pump={self.pump_pin}")

    def teardown(self):
        """Cancel any pending timers and safely release all GPIO pins."""
        if self._nozzle_timer is not None:
            self._nozzle_timer.cancel()

        if not self._is_mock and self._setup_done:
            for pin in self._all_pins:
                self._set_pin(pin, False)   # close valve and pump
            GPIO.cleanup(self._all_pins)

        print("[RelayController] Teardown complete.")

    # ─── Public API ───────────────────────────────────────────────────────────

    def spray(
        self,
        duration_ms: float,
        with_pump: bool = True,
    ) -> bool:
        """
        Open the nozzle solenoid for duration_ms milliseconds.

        Non-blocking: schedules a threading.Timer to close the valve.
        Respects cooldown to protect solenoid coil.

        Args:
            duration_ms: How long to keep the nozzle valve open (ms).
            with_pump:   Also activate the pump relay simultaneously.

        Returns:
            True if actuation was triggered, False if skipped (cooldown).
        """
        if not self._setup_done:
            log.warning("[RelayController] spray() called before setup()")
            return False

        with self._lock:
            now = time.time()
            since_last = now - self._last_spray_time
            if since_last < self.cooldown_ms:
                log.debug(
                    f"[RelayController] In cooldown "
                    f"({since_last*1000:.0f}ms < {self.cooldown_ms*1000:.0f}ms)"
                )
                return False

            self._last_spray_time = now
            self._spray_count += 1
            self._total_spray_ms += duration_ms

        dur_sec = max(duration_ms / 1000.0, 0.05)

        if self._is_mock:
            print(f"[RelayController] MOCK spray  "
                  f"nozzle_pin={self.nozzle_pin}  dur={duration_ms:.0f}ms  pump={with_pump}")
        else:
            log.info(
                f"[RelayController] SPRAY  pin={self.nozzle_pin}  "
                f"dur={duration_ms:.0f}ms"
            )

        # Activate nozzle (and pump)
        self._set_pin(self.nozzle_pin, True)
        if with_pump:
            self._set_pin(self.pump_pin, True)

        # Schedule deactivation
        def _close():
            self._set_pin(self.nozzle_pin, False)
            if with_pump:
                self._set_pin(self.pump_pin, False)

        with self._lock:
            if self._nozzle_timer is not None:
                self._nozzle_timer.cancel()
            t = threading.Timer(dur_sec, _close)
            self._nozzle_timer = t
        t.start()
        return True

    def all_off(self):
        """Immediately close nozzle and pump. Use for emergency stop."""
        if self._nozzle_timer is not None:
            self._nozzle_timer.cancel()
        for pin in self._all_pins:
            self._set_pin(pin, False)
        print("[RelayController] All relays OFF (emergency stop)")

    def stats(self) -> dict:
        """Return actuation telemetry."""
        return {
            "spray_count":    self._spray_count,
            "total_spray_ms": round(self._total_spray_ms, 1),
            "mock_mode":      self._is_mock,
        }

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _set_pin(self, pin: int, state: bool):
        """Set a GPIO pin, honouring active_low polarity."""
        if self._is_mock:
            return
        physical_state = (not state) if self.active_low else state
        GPIO.output(pin, GPIO.LOW if physical_state else GPIO.HIGH)

    # ─── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "RelayController":
        self.setup()
        return self

    def __exit__(self, *_):
        self.teardown()


# ─── CLI self-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== RelayController self-test ===")
    with RelayController() as rc:
        print("\nSpraying nozzle for 200ms...")
        rc.spray(200)
        time.sleep(0.4)

        print("Spraying nozzle for 350ms...")
        rc.spray(350)
        time.sleep(0.6)

        print("\nStats:", rc.stats())
    print("Done.")

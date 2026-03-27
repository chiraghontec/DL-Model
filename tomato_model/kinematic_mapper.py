#!/usr/bin/env python3
"""
kinematic_mapper.py

Converts Grad-CAM bbox Y-center pixel coordinates to physical actuator
position in millimetres, then to stepper motor step counts.

Kinematic chain:
  NEMA 17 (200 steps/rev, 1.8°) with 1/16 microstepping
  → 3200 steps/rev
  GT2 belt, 2 mm pitch × 20-tooth pulley
  → 40 mm/rev
  → 0.0125 mm/step  (= 80 steps/mm)

  H_mm = (y_center_px / frame_px) × camera_fov_height_mm
  steps = round(H_mm × steps_per_mm)

Usage:
  from kinematic_mapper import KinematicMapper
  km = KinematicMapper("config.json")
  steps = km.pixel_to_steps(y_center_px=112)
"""

import json
import math


class KinematicMapper:
    """
    Maps image pixel coordinates to stepper motor step counts.

    All positions are measured from the HOME (top) end-stop downward.
    """

    def __init__(self, config_path: str = "config.json"):
        with open(config_path) as f:
            cfg = json.load(f)

        stepper = cfg.get("stepper", {})
        self.steps_per_mm: float = stepper.get("steps_per_mm", 80.0)
        self.travel_mm: float = stepper.get("travel_mm", 800.0)
        self.home_offset_mm: float = stepper.get("home_offset_mm", 20.0)
        self.camera_fov_height_mm: float = stepper.get("camera_fov_height_mm", 300.0)

        model_cfg = cfg.get("model", {})
        self.frame_px: int = model_cfg.get("input_size", 224)

        # Derived limits
        self.min_steps: int = 0
        self.max_steps: int = round(self.travel_mm * self.steps_per_mm)

    # ─── Core conversions ────────────────────────────────────────────────────

    def pixel_to_mm(self, y_center_px: int) -> float:
        """
        Convert bbox Y-center pixel (0..frame_px) to physical height in mm.
        Accounts for the 30px offset caused by Resize(256) -> CenterCrop(224).
        """
        # Calculate resize padding to find true position in FOV
        # 640x480 resized to 256 shortest edge -> 256H. CenterCrop(224) drops 16px from top/bottom.
        crop_padding = (256 - self.frame_px) / 2.0  # 16 px in Resized space
        
        # True fractional position down the original FOV
        fraction_y = (y_center_px + crop_padding) / 256.0
        
        mm = fraction_y * self.camera_fov_height_mm
        return round(mm, 2)

    def mm_to_steps(self, h_mm: float) -> int:
        """Convert physical height (mm from HOME) to absolute step count."""
        return round(h_mm * self.steps_per_mm)

    def pixel_to_steps(self, y_center_px: int) -> int:
        """Full pipeline: pixel → mm → steps (clamped to travel limits)."""
        h_mm = self.pixel_to_mm(y_center_px)
        raw_steps = self.mm_to_steps(h_mm)
        return self.clamp_steps(raw_steps)

    def clamp_steps(self, steps: int) -> int:
        """Clamp step count to valid actuator travel range."""
        return max(self.min_steps, min(self.max_steps, steps))

    # ─── Inverse (for calibration / display) ─────────────────────────────────

    def steps_to_mm(self, steps: int) -> float:
        """Convert absolute step count to physical height in mm."""
        return round(steps / self.steps_per_mm, 2)

    # ─── Info ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"KinematicMapper("
            f"steps_per_mm={self.steps_per_mm}, "
            f"travel_mm={self.travel_mm}, "
            f"fov_height_mm={self.camera_fov_height_mm}, "
            f"frame_px={self.frame_px})"
        )

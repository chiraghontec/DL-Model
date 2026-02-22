#!/usr/bin/env python3
"""
inference_engine.py

Unified inference engine that combines:
  - ONNX Runtime for fast classification (FP32 or INT8)
  - PyTorch Grad-CAM for spatial localization (requires autograd)
  - Zone mapping (bbox Y → TOP/MID/BOTTOM)
  - VRA spray duration calculation (bbox area → ms)

This module is the primary entry point for the rover's perception pipeline.
On Raspberry Pi, it reads from the PiCamera; on dev machines, it processes
static images or webcam feed.

Architecture:
  ┌─────────────┐
  │  Camera Feed │
  └──────┬──────┘
         │
  ┌──────▼──────┐    ┌────────────────┐
  │ Preprocess   │───►│ ONNX Runtime   │──► class + confidence
  │ (resize,     │    │ (INT8 model)   │
  │  normalize)  │    └────────────────┘
  └──────┬──────┘
         │ (if class is diseased & confidence > threshold)
  ┌──────▼──────┐
  │ PyTorch      │──► heatmap + bounding box
  │ Grad-CAM     │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │ Zone Mapper  │──► primary zone + GPIO pin
  │ + VRA Calc   │──► spray duration (ms)
  └──────────────┘

Usage:
  # As a module:
  from inference_engine import InferenceEngine
  engine = InferenceEngine("config.json")
  result = engine.infer_from_file("leaf.jpg")

  # As a CLI:
  python inference_engine.py --image leaf.jpg --config config.json
  python inference_engine.py --camera --config config.json
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models, transforms

from gradcam import GradCAM, overlay_heatmap, draw_bbox_and_zone


class InferenceEngine:
    """
    Unified inference engine for the tomato blight spraying system.

    This class wraps:
      - ONNX Runtime session (for fast classification)
      - PyTorch model + Grad-CAM (for localization, loaded on demand)
      - Zone/VRA computation
    """

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize engine from configuration file.

        Args:
            config_path: Path to config.json.
        """
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Model config
        model_cfg = self.config["model"]
        self.class_names = sorted(model_cfg["classes"])
        self.input_size = model_cfg.get("input_size", 224)
        self.model_name = model_cfg.get("architecture", "resnet18")

        # Inference config
        inf_cfg = self.config.get("inference", {})
        self.confidence_threshold = inf_cfg.get("confidence_threshold", 0.70)
        self.gradcam_layer = inf_cfg.get("gradcam_layer", "layer4")
        self.heatmap_threshold = inf_cfg.get("heatmap_threshold", 0.5)

        # VRA config
        vra_cfg = self.config.get("vra", {})
        self.vra_alpha = vra_cfg.get("alpha", 0.025)
        self.vra_beta = vra_cfg.get("beta", 30.0)
        self.vra_min_ms = vra_cfg.get("min_spray_ms", 50)
        self.vra_max_ms = vra_cfg.get("max_spray_ms", 500)

        # Zone config
        zone_cfg = self.config.get("zone_mapping", {})
        self.zones = zone_cfg.get("zones", [
            {"name": "TOP", "gpio_pin": 17},
            {"name": "MID", "gpio_pin": 27},
            {"name": "BOTTOM", "gpio_pin": 22},
        ])
        self.zone_boundaries = [0.333, 0.667]

        # Camera config
        cam_cfg = self.config.get("camera", {})
        self.frame_width = cam_cfg.get("resolution", [640, 480])[0]
        self.frame_height = cam_cfg.get("resolution", [640, 480])[1]

        # Preprocessing transforms (same as training validation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Initialize models
        self.onnx_session = None
        self.pytorch_model = None
        self.gradcam = None

        # Telemetry
        self._inference_count = 0
        self._total_latency_ms = 0

    def load_onnx_model(self, onnx_path: str):
        """Load ONNX model for classification."""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        providers = ["CPUExecutionProvider"]

        # Check for CUDA/TensorRT on non-Pi systems
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
        self.onnx_input_name = self.onnx_session.get_inputs()[0].name

        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[InferenceEngine] ONNX model loaded: {onnx_path} ({size_mb:.1f} MB)")
        print(f"[InferenceEngine] Providers: {self.onnx_session.get_providers()}")

    def load_pytorch_model(self, checkpoint_path: str, device: str = "auto"):
        """
        Load PyTorch model for Grad-CAM.

        Note: Grad-CAM requires autograd, so this must be a PyTorch model
        (not ONNX). On resource-constrained devices, this is loaded on
        demand only when disease is detected.
        """
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # Grad-CAM backward hooks may not work on MPS
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        num_classes = ckpt.get("num_classes", len(self.class_names))

        if self.model_name == "resnet18":
            model = models.resnet18(weights=None)
        elif self.model_name == "resnet50":
            model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(self.device)
        model.eval()

        self.pytorch_model = model
        self.gradcam = GradCAM(model, target_layer_name=self.gradcam_layer)

        print(f"[InferenceEngine] PyTorch model loaded for Grad-CAM (device: {self.device})")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ONNX Runtime inference.

        Args:
            image: BGR image from OpenCV (H, W, 3).

        Returns:
            Preprocessed numpy array (1, 3, 224, 224).
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor.unsqueeze(0).numpy()

    def preprocess_torch(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for PyTorch Grad-CAM.

        Args:
            image: BGR image from OpenCV (H, W, 3).

        Returns:
            Preprocessed torch tensor (1, 3, 224, 224).
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor.unsqueeze(0).to(self.device)

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run ONNX classification on a single image.

        Args:
            image: BGR image from OpenCV.

        Returns:
            dict: class_name, class_idx, confidence, all_probs, latency_ms
        """
        if self.onnx_session is None:
            raise RuntimeError("ONNX model not loaded. Call load_onnx_model() first.")

        preprocessed = self.preprocess(image)

        start = time.time()
        outputs = self.onnx_session.run(None, {self.onnx_input_name: preprocessed})[0]
        latency_ms = (time.time() - start) * 1000

        # Softmax
        exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probs = exp_out / exp_out.sum(axis=1, keepdims=True)

        pred_idx = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, pred_idx])

        self._inference_count += 1
        self._total_latency_ms += latency_ms

        return {
            "class_name": self.class_names[pred_idx],
            "class_idx": pred_idx,
            "confidence": confidence,
            "all_probs": {
                name: float(probs[0, i])
                for i, name in enumerate(self.class_names)
            },
            "latency_ms": latency_ms,
        }

    def localize(self, image: np.ndarray, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Grad-CAM localization on a single image.

        Args:
            image: BGR image from OpenCV.
            target_class: Override class for Grad-CAM (optional).

        Returns:
            dict with heatmap, bbox, zone, spray_duration, etc.
        """
        if self.gradcam is None:
            raise RuntimeError("PyTorch model not loaded. Call load_pytorch_model() first.")

        tensor = self.preprocess_torch(image)

        start = time.time()
        result = self.gradcam(
            tensor,
            target_class=target_class,
            heatmap_threshold=self.heatmap_threshold,
            frame_height=self.frame_height,
            zone_boundaries=self.zone_boundaries,
            vra_alpha=self.vra_alpha,
            vra_beta=self.vra_beta,
            vra_min_ms=self.vra_min_ms,
            vra_max_ms=self.vra_max_ms,
        )
        latency_ms = (time.time() - start) * 1000

        result["localization_latency_ms"] = latency_ms

        # Map zone name to GPIO pin
        if result["primary_zone"]:
            for zone in self.zones:
                if zone["name"] == result["primary_zone"]:
                    result["gpio_pin"] = zone["gpio_pin"]
                    break

        return result

    def infer(self, image: np.ndarray, run_gradcam: bool = True) -> Dict[str, Any]:
        """
        Full inference pipeline: classify → (optional) localize → zone/VRA.

        Args:
            image: BGR image from OpenCV.
            run_gradcam: If True and disease is detected, run Grad-CAM.

        Returns:
            Complete inference result dict.
        """
        total_start = time.time()

        # Step 1: ONNX classification
        cls_result = self.classify(image)

        result = {
            **cls_result,
            "should_spray": False,
            "heatmap": None,
            "bbox": None,
            "primary_zone": None,
            "all_zones": [],
            "spray_duration_ms": 0,
            "gpio_pin": None,
            "bbox_area": 0,
            "localization_latency_ms": 0,
        }

        # Step 2: Determine if spraying needed
        is_diseased = cls_result["class_name"] != "healthy"
        above_threshold = cls_result["confidence"] >= self.confidence_threshold

        if is_diseased and above_threshold:
            result["should_spray"] = True

            # Step 3: Grad-CAM localization (if model loaded)
            if run_gradcam and self.gradcam is not None:
                loc_result = self.localize(image, target_class=cls_result["class_idx"])
                result.update({
                    "heatmap": loc_result["heatmap"],
                    "bbox": loc_result["bbox"],
                    "primary_zone": loc_result["primary_zone"],
                    "all_zones": loc_result["all_zones"],
                    "spray_duration_ms": loc_result["spray_duration_ms"],
                    "gpio_pin": loc_result.get("gpio_pin"),
                    "bbox_area": loc_result["bbox_area"],
                    "localization_latency_ms": loc_result["localization_latency_ms"],
                })

        result["total_latency_ms"] = (time.time() - total_start) * 1000

        return result

    def infer_from_file(self, image_path: str, run_gradcam: bool = True) -> Dict[str, Any]:
        """Convenience: load image from file and run full pipeline."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        result = self.infer(image, run_gradcam)
        result["image_path"] = image_path
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return telemetry stats."""
        avg_latency = (
            self._total_latency_ms / self._inference_count
            if self._inference_count > 0
            else 0
        )
        return {
            "inference_count": self._inference_count,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_fps": round(1000 / avg_latency, 1) if avg_latency > 0 else 0,
        }

    def visualize_result(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create annotated visualization of inference result.

        Args:
            image: Original BGR image.
            result: Inference result dict from self.infer().
            save_path: Optional path to save the visualization.

        Returns:
            Annotated BGR image.
        """
        # Resize to model input size for consistency with heatmap
        display = cv2.resize(image, (self.input_size, self.input_size))

        if result.get("heatmap") is not None:
            display = overlay_heatmap(display, result["heatmap"], alpha=0.4)

        annotated = draw_bbox_and_zone(
            display,
            result.get("bbox"),
            result.get("primary_zone"),
            result.get("spray_duration_ms", 0),
            result.get("class_name", "unknown"),
            result.get("confidence", 0),
            self.zone_boundaries,
        )

        # Add latency info
        latency_text = f"Cls: {result.get('latency_ms', 0):.0f}ms"
        if result.get("localization_latency_ms", 0) > 0:
            latency_text += f" | Loc: {result['localization_latency_ms']:.0f}ms"
        cv2.putText(
            annotated, latency_text, (5, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            cv2.imwrite(save_path, annotated)

        return annotated


# ---------------------------------------------------------------------------
# CLI: single-image demo and webcam feed
# ---------------------------------------------------------------------------

def print_result(result: Dict[str, Any], class_names: List[str]):
    """Pretty-print inference result."""
    print(f"\n{'─'*50}")
    print(f"  Class:        {result['class_name']}")
    print(f"  Confidence:   {result['confidence']:.4f}")
    print(f"  Should Spray: {'YES' if result['should_spray'] else 'NO'}")

    if result["should_spray"]:
        print(f"  BBox:         {result.get('bbox')}")
        print(f"  BBox Area:    {result.get('bbox_area', 0)} px²")
        print(f"  Zone:         {result.get('primary_zone')} (zones: {result.get('all_zones', [])})")
        print(f"  GPIO Pin:     {result.get('gpio_pin')}")
        print(f"  Spray Duration: {result.get('spray_duration_ms', 0)} ms")

    print(f"  Latency:")
    print(f"    Classification: {result.get('latency_ms', 0):.1f} ms")
    print(f"    Localization:   {result.get('localization_latency_ms', 0):.1f} ms")
    print(f"    Total:          {result.get('total_latency_ms', 0):.1f} ms")

    # All class probabilities
    if "all_probs" in result:
        print(f"  Probabilities:")
        for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 30)
            print(f"    {cls:15s} {prob:.4f} {bar}")
    print(f"{'─'*50}")


def run_webcam(engine: InferenceEngine):
    """Run inference on live webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, engine.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, engine.frame_height)

    print("Press 'q' to quit, 's' to save current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = engine.infer(frame)
        annotated = engine.visualize_result(frame, result)

        # Add FPS counter
        stats = engine.get_stats()
        cv2.putText(
            annotated, f"FPS: {stats['avg_fps']:.1f}",
            (annotated.shape[1] - 100, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

        cv2.imshow("Tomato Blight Detector", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = f"capture_{int(time.time())}.png"
            cv2.imwrite(save_path, annotated)
            print(f"Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession stats: {engine.get_stats()}")


def main():
    parser = argparse.ArgumentParser(
        description="Tomato Blight Inference Engine"
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--onnx", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--checkpoint", type=str, default=None, help="PyTorch checkpoint (for Grad-CAM)")
    parser.add_argument("--image", type=str, default=None, help="Single image inference")
    parser.add_argument("--image_dir", type=str, default=None, help="Run on all images in directory")
    parser.add_argument("--camera", action="store_true", help="Run webcam feed")
    parser.add_argument("--output_dir", type=str, default="inference_output", help="Output directory for visualizations")
    parser.add_argument("--no_gradcam", action="store_true", help="Skip Grad-CAM (classification only)")
    args = parser.parse_args()

    # Initialize engine
    engine = InferenceEngine(args.config)

    # Load ONNX model
    if args.onnx:
        engine.load_onnx_model(args.onnx)
    else:
        # Try to find ONNX model from config directory
        config_dir = os.path.dirname(args.config)
        for candidate in [
            os.path.join(config_dir, "models", "resnet18_tomato_int8.onnx"),
            os.path.join(config_dir, "models", "resnet18_tomato_fp32.onnx"),
        ]:
            if os.path.exists(candidate):
                engine.load_onnx_model(candidate)
                break
        else:
            print("Error: No ONNX model found. Use --onnx to specify path.")
            return

    # Load PyTorch model for Grad-CAM (optional)
    if not args.no_gradcam and args.checkpoint:
        engine.load_pytorch_model(args.checkpoint)
    elif not args.no_gradcam:
        # Try to find checkpoint
        config_dir = os.path.dirname(args.config)
        ckpt_dir = os.path.join(config_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            # Find best checkpoint
            best_ckpts = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("best_")],
                key=lambda x: float(x.split("_acc_")[1].replace(".pth", "")) if "_acc_" in x else 0,
                reverse=True,
            )
            if best_ckpts:
                ckpt_path = os.path.join(ckpt_dir, best_ckpts[0])
                engine.load_pytorch_model(ckpt_path)

    # Run inference
    if args.camera:
        run_webcam(engine)

    elif args.image:
        result = engine.infer_from_file(args.image, run_gradcam=not args.no_gradcam)
        print_result(result, engine.class_names)

        # Save visualization
        image = cv2.imread(args.image)
        os.makedirs(args.output_dir, exist_ok=True)
        vis_path = os.path.join(args.output_dir, os.path.basename(args.image))
        engine.visualize_result(image, result, save_path=vis_path)
        print(f"Visualization saved: {vis_path}")

    elif args.image_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        image_files = [
            f for f in os.listdir(args.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"Processing {len(image_files)} images from {args.image_dir}...")
        results_log = []

        for fname in image_files:
            fpath = os.path.join(args.image_dir, fname)
            result = engine.infer_from_file(fpath, run_gradcam=not args.no_gradcam)

            # Save visualization
            image = cv2.imread(fpath)
            vis_path = os.path.join(args.output_dir, fname)
            engine.visualize_result(image, result, save_path=vis_path)

            # Log (without numpy arrays)
            log_entry = {k: v for k, v in result.items() if k != "heatmap"}
            if log_entry.get("bbox"):
                log_entry["bbox"] = list(log_entry["bbox"])
            results_log.append(log_entry)

            status = "SPRAY" if result["should_spray"] else "SKIP"
            print(f"  [{status}] {fname}: {result['class_name']} ({result['confidence']:.3f})")

        # Save results log
        log_path = os.path.join(args.output_dir, "inference_log.json")
        with open(log_path, "w") as f:
            json.dump(results_log, f, indent=2, default=str)

        print(f"\nResults: {log_path}")
        print(f"Stats: {engine.get_stats()}")

    else:
        print("Specify --image, --image_dir, or --camera. Use --help for details.")


if __name__ == "__main__":
    main()

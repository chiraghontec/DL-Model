#!/usr/bin/env python3
"""
gradcam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for
spatial localization of disease regions in tomato leaf images.

Pipeline:
  1. Forward pass → class prediction + confidence
  2. Backward pass → gradients w.r.t. target conv layer activations
  3. Global Average Pool gradients → channel weights
  4. Weighted sum of feature maps → heatmap (7×7 for ResNet layer4)
  5. ReLU → bilinear upscale → normalize [0, 1]
  6. Threshold → binary mask → largest connected component → bounding box

Reference:
  Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization. ICCV.

Usage:
  # As a module:
  from gradcam import GradCAM
  cam = GradCAM(model, target_layer="layer4")
  result = cam(image_tensor)

  # As a script (single-image demo):
  python gradcam.py --image path/to/leaf.jpg --checkpoint checkpoints/best.pth --output heatmap.png
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class GradCAM:
    """
    Grad-CAM implementation for ResNet-family models.

    Attributes:
        model: The PyTorch model (ResNet-18/50).
        target_layer: The convolutional layer to extract activations from.
        activations: Stored forward-pass activations.
        gradients: Stored backward-pass gradients.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "layer4"):
        """
        Args:
            model: A trained ResNet model.
            target_layer_name: Name of the target conv layer (default: "layer4").
        """
        self.model = model
        self.model.eval()

        # Get target layer by name
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W).
            target_class: Class index to generate heatmap for.
                         If None, uses the predicted class.

        Returns:
            heatmap: Normalized heatmap array, shape (H, W), values in [0, 1].
            pred_class: Predicted class index.
            confidence: Prediction confidence (softmax probability).
        """
        # Ensure gradients are enabled
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for the target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Grad-CAM computation
        # gradients shape: (1, C, h, w) — e.g., (1, 512, 7, 7) for ResNet-18 layer4
        # activations shape: same
        gradients = self.gradients[0]  # (C, h, w)
        activations = self.activations[0]  # (C, h, w)

        # Global Average Pool the gradients → channel weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=activations.dtype, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upscale to input resolution
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.resize(cam, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

        return heatmap, target_class, confidence

    def extract_bbox(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract bounding box from Grad-CAM heatmap.

        Args:
            heatmap: Normalized heatmap, shape (H, W), values in [0, 1].
            threshold: Threshold for binary mask (default: 0.5).

        Returns:
            bbox: (x, y, w, h) in pixel coordinates, or None if no region found.
        """
        # Binary threshold
        binary_mask = (heatmap >= threshold).astype(np.uint8)

        if binary_mask.sum() == 0:
            # Try lower threshold
            binary_mask = (heatmap >= threshold * 0.5).astype(np.uint8)
            if binary_mask.sum() == 0:
                return None

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        if num_labels <= 1:
            return None

        # Find largest component (skip background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        w = stats[largest_label, cv2.CC_STAT_WIDTH]
        h = stats[largest_label, cv2.CC_STAT_HEIGHT]

        return (x, y, w, h)

    def compute_zone(
        self,
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        zone_boundaries: List[float] = [0.333, 0.667],
    ) -> Tuple[str, List[str]]:
        """
        Map bounding box Y-coordinate to spray zone.

        Args:
            bbox: (x, y, w, h) in pixel coordinates.
            frame_height: Total frame height in pixels.
            zone_boundaries: Y-fraction boundaries [top/mid, mid/bottom].

        Returns:
            primary_zone: The main zone ("TOP", "MID", or "BOTTOM").
            all_zones: List of all zones the bbox spans (may be multi-zone).
        """
        zone_names = ["TOP", "MID", "BOTTOM"]
        x, y, w, h = bbox

        y_center = y + h / 2
        y_top = y
        y_bottom = y + h

        # Zone pixel boundaries
        boundary_1 = int(frame_height * zone_boundaries[0])
        boundary_2 = int(frame_height * zone_boundaries[1])

        # Primary zone (based on center)
        if y_center < boundary_1:
            primary_zone = "TOP"
        elif y_center < boundary_2:
            primary_zone = "MID"
        else:
            primary_zone = "BOTTOM"

        # Check if bbox spans multiple zones
        zones_hit = set()
        if y_top < boundary_1:
            zones_hit.add("TOP")
        if y_top < boundary_2 and y_bottom >= boundary_1:
            zones_hit.add("MID")
        if y_bottom >= boundary_2:
            zones_hit.add("BOTTOM")

        return primary_zone, sorted(zones_hit)

    def compute_spray_duration(
        self,
        bbox: Tuple[int, int, int, int],
        alpha: float = 0.025,
        beta: float = 30.0,
        min_ms: int = 50,
        max_ms: int = 500,
    ) -> int:
        """
        Compute spray duration (ms) from bounding box area (VRA).

        Formula: d = clamp(alpha * area_pixels + beta, min_ms, max_ms)

        Args:
            bbox: (x, y, w, h) in pixel coordinates.
            alpha: Scaling coefficient.
            beta: Intercept (minimum base duration).
            min_ms: Minimum spray duration (solenoid inertia).
            max_ms: Maximum spray duration (prevents over-application).

        Returns:
            duration_ms: Spray relay-open duration in milliseconds.
        """
        _, _, w, h = bbox
        area = w * h
        duration = alpha * area + beta
        duration_ms = int(max(min_ms, min(max_ms, duration)))
        return duration_ms

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        heatmap_threshold: float = 0.5,
        frame_height: int = 480,
        zone_boundaries: List[float] = [0.333, 0.667],
        vra_alpha: float = 0.025,
        vra_beta: float = 30.0,
        vra_min_ms: int = 50,
        vra_max_ms: int = 500,
    ) -> Dict:
        """
        Full pipeline: image → heatmap → bbox → zone → spray duration.

        Args:
            input_tensor: Preprocessed image, shape (1, 3, 224, 224).
            target_class: Override predicted class (optional).
            heatmap_threshold: Threshold for bbox extraction.
            frame_height: Original camera frame height (for zone mapping).
            zone_boundaries: Zone Y-fraction boundaries.
            vra_alpha, vra_beta, vra_min_ms, vra_max_ms: VRA parameters.

        Returns:
            dict with keys:
                - class_idx: Predicted class index
                - confidence: Prediction confidence
                - heatmap: Numpy heatmap (H, W)
                - bbox: (x, y, w, h) or None
                - primary_zone: "TOP"/"MID"/"BOTTOM" or None
                - all_zones: List of zones hit, or []
                - spray_duration_ms: int or 0
                - bbox_area: int or 0
        """
        heatmap, pred_class, confidence = self.generate_heatmap(
            input_tensor, target_class
        )

        result = {
            "class_idx": pred_class,
            "confidence": confidence,
            "heatmap": heatmap,
            "bbox": None,
            "primary_zone": None,
            "all_zones": [],
            "spray_duration_ms": 0,
            "bbox_area": 0,
        }

        bbox = self.extract_bbox(heatmap, threshold=heatmap_threshold)
        if bbox is not None:
            result["bbox"] = bbox
            result["bbox_area"] = bbox[2] * bbox[3]

            primary_zone, all_zones = self.compute_zone(
                bbox, frame_height, zone_boundaries
            )
            result["primary_zone"] = primary_zone
            result["all_zones"] = all_zones

            duration = self.compute_spray_duration(
                bbox, vra_alpha, vra_beta, vra_min_ms, vra_max_ms
            )
            result["spray_duration_ms"] = duration

        return result


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay heatmap on original image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Resize heatmap to match original image
    if heatmap_color.shape[:2] != original_image.shape[:2]:
        heatmap_color = cv2.resize(
            heatmap_color,
            (original_image.shape[1], original_image.shape[0]),
        )

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def draw_bbox_and_zone(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]],
    zone: Optional[str],
    duration_ms: int,
    class_name: str,
    confidence: float,
    zone_boundaries: List[float] = [0.333, 0.667],
) -> np.ndarray:
    """Draw bounding box, zone lines, and info text on image."""
    img = image.copy()
    h, w = img.shape[:2]

    # Draw zone boundary lines
    for frac in zone_boundaries:
        y_line = int(h * frac)
        cv2.line(img, (0, y_line), (w, y_line), (255, 255, 0), 1, cv2.LINE_AA)

    # Zone labels
    zone_names = ["TOP", "MID", "BOTTOM"]
    zone_y_positions = [
        int(h * 0.15),
        int(h * 0.5),
        int(h * 0.85),
    ]
    for name, y_pos in zip(zone_names, zone_y_positions):
        cv2.putText(
            img, name, (5, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA,
        )

    # Draw bounding box
    if bbox is not None:
        x, y, bw, bh = bbox
        color = (0, 0, 255) if "blight" in class_name else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)

        # Info label
        label = f"{class_name} ({confidence:.2f}) Z:{zone} {duration_ms}ms"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(
            img, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img


# ---------------------------------------------------------------------------
# CLI for single-image demo
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and preprocess a single image for inference."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load with OpenCV (BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    tensor = transform(img_rgb).unsqueeze(0)  # (1, 3, 224, 224)

    # Also prepare display-size image (center-cropped to match model input)
    pil_img = transforms.ToPILImage()(transforms.ToTensor()(img_rgb))
    display_img = transforms.CenterCrop(size)(transforms.Resize(256)(pil_img))
    display_np = np.array(display_img)  # RGB
    display_bgr = cv2.cvtColor(display_np, cv2.COLOR_RGB2BGR)

    return tensor, display_bgr


def build_model_for_gradcam(
    model_name: str, num_classes: int, checkpoint_path: str, device: torch.device
) -> nn.Module:
    """Build model and load checkpoint for Grad-CAM (requires grad)."""
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    return model, ckpt.get("classes", ["early_blight", "healthy", "late_blight"])


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for tomato blight model"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="gradcam_output.png", help="Output image path")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--target_layer", type=str, default="layer4")
    parser.add_argument("--threshold", type=float, default=0.5, help="Heatmap threshold for bbox")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Note: Grad-CAM requires CPU for backward hooks on MPS in some PyTorch versions
    # Fall back to CPU if MPS
    if device.type == "mps":
        print("Note: Using CPU for Grad-CAM (MPS backward hooks may not be supported).")
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", len(ckpt.get("classes", [])))
    class_names = ckpt.get("classes", ["early_blight", "healthy", "late_blight"])
    model, _ = build_model_for_gradcam(args.model, num_classes, args.checkpoint, device)

    # Create GradCAM
    cam = GradCAM(model, target_layer_name=args.target_layer)

    # Preprocess image
    input_tensor, display_img = preprocess_image(args.image)
    input_tensor = input_tensor.to(device)

    # Run full pipeline
    result = cam(input_tensor, heatmap_threshold=args.threshold)

    # Print results
    class_name = class_names[result["class_idx"]]
    print(f"\n{'='*50}")
    print(f"Grad-CAM Results")
    print(f"{'='*50}")
    print(f"  Class:            {class_name}")
    print(f"  Confidence:       {result['confidence']:.4f}")
    print(f"  Bounding Box:     {result['bbox']}")
    print(f"  BBox Area:        {result['bbox_area']} px²")
    print(f"  Primary Zone:     {result['primary_zone']}")
    print(f"  All Zones:        {result['all_zones']}")
    print(f"  Spray Duration:   {result['spray_duration_ms']} ms")
    print(f"{'='*50}")

    # Generate visualization
    heatmap_overlay = overlay_heatmap(display_img, result["heatmap"], alpha=0.4)
    annotated = draw_bbox_and_zone(
        heatmap_overlay,
        result["bbox"],
        result["primary_zone"],
        result["spray_duration_ms"],
        class_name,
        result["confidence"],
    )

    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, annotated)
    print(f"\n  Output saved: {args.output}")

    # Also save raw heatmap
    heatmap_path = os.path.splitext(args.output)[0] + "_heatmap.png"
    heatmap_vis = np.uint8(255 * result["heatmap"])
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, heatmap_color)
    print(f"  Heatmap saved:  {heatmap_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick pipeline smoke test — run from tomato_model/ directory."""
import json
import os
import sys

# Must be run from tomato_model/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from inference_engine import InferenceEngine

# ── ONNX-only test ────────────────────────────────────────────────────────────
print("=== ONNX Classification Test ===")
eng = InferenceEngine("config.json")
eng.load_onnx_model(eng.config["model"]["onnx_path"])

test_images = {
    "early_blight": "data/test/early_blight/00075aa8-d4a3-4b82-a53e-2b4e09cd52b1___RS_Early.B 7226.JPG",
    "healthy":      "data/test/healthy/0a5b2a94-4a98-4b93-bb5e-61ede3da7db9___RS_HL 2080.JPG",
    "late_blight":  "data/test/late_blight/0fca6d09-1931-4767-a195-1dd2b7cd10b7___GHLB Leaf 8 Day 12.JPG",
}

ok = 0
for label, path in test_images.items():
    if not os.path.exists(path):
        # pick any image from the class dir
        cls_dir = f"data/test/{label}"
        if os.path.isdir(cls_dir):
            files = sorted(os.listdir(cls_dir))
            path = os.path.join(cls_dir, files[0])
        else:
            print(f"  SKIP {label}: no test data found")
            continue

    r = eng.infer_from_file(path, run_gradcam=False)
    correct = r["class_name"] == label
    ok += int(correct)
    mark = "✓" if correct else "✗"
    print(f"  {mark} {label:15s}  pred={r['class_name']:15s}  conf={r['confidence']:.3f}  "
          f"lat={r['total_latency_ms']:.1f}ms  spray={r['should_spray']}")

print(f"\n  {ok}/{len(test_images)} correct with ONNX\n")

# ── Full pipeline test (ONNX + GradCAM) ─────────────────────────────────────
print("=== Full Pipeline Test (ONNX + GradCAM) ===")
eng2 = InferenceEngine("config.json")
eng2.load_onnx_model(eng2.config["model"]["onnx_path"])
eng2.load_pytorch_model(eng2.config["model"]["checkpoint_path"])

# Test one blight image with GradCAM
for label in ("early_blight", "late_blight"):
    cls_dir = f"data/test/{label}"
    if not os.path.isdir(cls_dir):
        continue
    img_path = os.path.join(cls_dir, sorted(os.listdir(cls_dir))[0])
    r = eng2.infer_from_file(img_path, run_gradcam=True)
    bbox = r.get("bbox")
    print(f"\n  {label}")
    print(f"    Class      : {r['class_name']} (conf={r['confidence']:.3f})")
    print(f"    Should spray: {r['should_spray']}")
    print(f"    BBox        : {bbox}")
    print(f"    Y center    : {r.get('y_center_px')} px")
    print(f"    Y target    : {r.get('y_target_mm')} mm")
    print(f"    Spray ms    : {r['spray_duration_ms']}")
    print(f"    Total lat   : {r['total_latency_ms']:.1f} ms")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/test_{label}_gradcam.jpg"
    import cv2
    vis = eng2.visualize_result(cv2.imread(img_path), r, save_path=out_path)
    print(f"    Saved viz   : {out_path}")

print("\n=== Smoke test complete ===")

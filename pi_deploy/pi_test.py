#!/usr/bin/env python3
"""
pi_test.py — Standalone benchmark & accuracy tester for Raspberry Pi 4.

Measures:
  - Hardware inference latency (mean, P50, P95, P99, FPS)
  - CPU temperature (reads /sys/class/thermal)
  - Model accuracy on labelled test images (if --test_dir supplied)
  - Per-class precision, recall, F1

Dependencies (all installable via pip on Raspberry Pi OS):
  onnxruntime  numpy  Pillow  psutil

Usage:
  # Latency only (500 runs):
  python3 pi_test.py --model resnet18_tomato_int8.onnx

  # Accuracy + latency:
  python3 pi_test.py --model resnet18_tomato_int8.onnx --test_dir test/ --samples 300

  # Compare INT8 vs FP32:
  python3 pi_test.py --model resnet18_tomato_int8.onnx \
                     --fp32   resnet18_tomato_fp32.onnx \
                     --test_dir test/ --samples 300
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found.  Run: pip3 install onnxruntime")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Constants ────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["early_blight", "healthy", "late_blight"]


# ── Image preprocessing (matches training validation pipeline) ───────────────
def preprocess(image_path: str) -> np.ndarray:
    """Resize shortest-side→256, centre-crop 224×224, ImageNet normalise → NCHW."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    w, h = img.size
    left = (w - 224) // 2
    top  = (h - 224) // 2
    img  = img.crop((left, top, left + 224, top + 224))
    arr  = np.array(img, dtype=np.float32) / 255.0
    arr  = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]   # NCHW


def rand_input() -> np.ndarray:
    """Random normalised tensor for warm-up / latency-only runs."""
    a = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return (a - MEAN.reshape(1, 3, 1, 1)) / STD.reshape(1, 3, 1, 1)


# ── System info ───────────────────────────────────────────────────────────────
def cpu_temp() -> Optional[float]:
    """Return CPU temperature in °C (Raspberry Pi and Linux)."""
    p = Path("/sys/class/thermal/thermal_zone0/temp")
    if p.exists():
        try:
            return int(p.read_text().strip()) / 1000.0
        except Exception:
            pass
    if HAS_PSUTIL:
        try:
            for vals in psutil.sensors_temperatures().values():
                if vals:
                    return vals[0].current
        except Exception:
            pass
    return None


def cpu_freq() -> Optional[float]:
    """Return current CPU frequency in MHz (Linux)."""
    p = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if p.exists():
        try:
            return int(p.read_text().strip()) / 1000.0
        except Exception:
            pass
    return None


def mem_mb() -> Optional[float]:
    if HAS_PSUTIL:
        try:
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            pass
    return None


# ── ONNX session ──────────────────────────────────────────────────────────────
def make_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path, sess_options=opts,
        providers=["CPUExecutionProvider"]
    )


# ── Latency benchmark ─────────────────────────────────────────────────────────
def bench_latency(sess: ort.InferenceSession,
                  warmup: int = 20,
                  runs: int = 500) -> Dict:
    name = sess.get_inputs()[0].name
    print(f"  Warm-up ({warmup} runs)...", end="", flush=True)
    for _ in range(warmup):
        sess.run(None, {name: rand_input()})
    print(" done")

    print(f"  Benchmarking ({runs} runs)...", end="", flush=True)
    t_wall = time.perf_counter()
    lats: List[float] = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {name: rand_input()})
        lats.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - t_wall
    print(" done")

    l = np.array(lats)
    return dict(
        n_runs   = runs,
        mean_ms  = float(np.mean(l)),
        median_ms= float(np.median(l)),
        std_ms   = float(np.std(l)),
        min_ms   = float(l.min()),
        max_ms   = float(l.max()),
        p50_ms   = float(np.percentile(l, 50)),
        p95_ms   = float(np.percentile(l, 95)),
        p99_ms   = float(np.percentile(l, 99)),
        fps      = float(runs / total),
        total_s  = float(total),
    )


# ── Accuracy evaluation ───────────────────────────────────────────────────────
def test_accuracy(sess: ort.InferenceSession,
                  test_dir: str,
                  n_samples: int) -> Dict:
    """
    Evaluate accuracy on labelled images.

    test_dir must contain subdirectories named after CLASS_NAMES:
       test_dir/early_blight/
       test_dir/healthy/
       test_dir/late_blight/
    """
    name = sess.get_inputs()[0].name

    # Collect all images per class
    pool: List[Tuple[str, int]] = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = Path(test_dir) / cls_name
        if not cls_dir.is_dir():
            print(f"  [WARN] directory not found: {cls_dir}")
            continue
        exts = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
        imgs = [f for ext in exts for f in cls_dir.glob(ext)]
        pool.extend((str(p), cls_idx) for p in imgs)

    if not pool:
        return {"error": "No images found in test_dir"}

    # Stratified subsample
    random.seed(42)
    per_class = max(1, n_samples // len(CLASS_NAMES))
    selected: List[Tuple[str, int]] = []
    for ci in range(len(CLASS_NAMES)):
        cls_pool = [(p, l) for p, l in pool if l == ci]
        random.shuffle(cls_pool)
        selected.extend(cls_pool[:per_class])
    random.shuffle(selected)
    N = len(selected)
    print(f"  Evaluating {N} images ({per_class}/class)...")

    correct = 0
    pc_correct = [0] * len(CLASS_NAMES)
    pc_total   = [0] * len(CLASS_NAMES)
    # For precision/recall
    tp = [0] * len(CLASS_NAMES)
    fp = [0] * len(CLASS_NAMES)
    fn = [0] * len(CLASS_NAMES)
    infer_times: List[float] = []

    for idx, (img_path, true_label) in enumerate(selected):
        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{N}  acc so far: {correct/(idx+1)*100:.1f}%",
                  flush=True)
        try:
            inp = preprocess(img_path)
        except Exception as e:
            print(f"    [SKIP] {Path(img_path).name}: {e}")
            continue

        t = time.perf_counter()
        logits = sess.run(None, {name: inp})[0][0]
        infer_times.append((time.perf_counter() - t) * 1000)

        exp_l = np.exp(logits - logits.max())
        probs = exp_l / exp_l.sum()
        pred  = int(probs.argmax())
        conf  = float(probs[pred])

        pc_total[true_label] += 1
        if pred == true_label:
            correct += 1
            pc_correct[true_label] += 1
            tp[true_label] += 1
        else:
            fp[pred] += 1
            fn[true_label] += 1

    t_arr = np.array(infer_times)
    per_class_metrics = {}
    for i, cname in enumerate(CLASS_NAMES):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class_metrics[cname] = dict(
            correct   = pc_correct[i],
            total     = pc_total[i],
            accuracy  = float(pc_correct[i] / pc_total[i] * 100) if pc_total[i] else 0.0,
            precision = float(prec * 100),
            recall    = float(rec  * 100),
            f1        = float(f1   * 100),
        )

    return dict(
        n_images     = N,
        correct      = correct,
        accuracy_pct = float(correct / N * 100),
        mean_ms      = float(np.mean(t_arr))          if len(t_arr) else 0.0,
        p95_ms       = float(np.percentile(t_arr, 95)) if len(t_arr) else 0.0,
        per_class    = per_class_metrics,
    )


# ── Display helpers ───────────────────────────────────────────────────────────
def hdr(title: str):
    print(f"\n{'='*56}\n  {title}\n{'='*56}")

def sep():
    print("─" * 56)

def show_latency(label: str, d: Dict):
    print(f"\n  ── {label} ──")
    print(f"  Mean latency : {d['mean_ms']:>8.2f} ms")
    print(f"  Median (P50) : {d['p50_ms']:>8.2f} ms")
    print(f"  P95 latency  : {d['p95_ms']:>8.2f} ms")
    print(f"  P99 latency  : {d['p99_ms']:>8.2f} ms")
    print(f"  Min / Max    : {d['min_ms']:.2f} / {d['max_ms']:.2f} ms")
    print(f"  Std dev      : {d['std_ms']:>8.2f} ms")
    print(f"  Throughput   : {d['fps']:>8.1f} FPS")
    print(f"  Total time   : {d['total_s']:.2f} s  ({d['n_runs']} runs)")


def show_accuracy(label: str, d: Dict):
    if "error" in d:
        print(f"  Accuracy [{label}]: {d['error']}")
        return
    print(f"\n  ── Accuracy: {label} ──")
    print(f"  Overall : {d['accuracy_pct']:.2f}%  ({d['correct']}/{d['n_images']})")
    print(f"  Mean inference : {d['mean_ms']:.2f} ms   P95: {d['p95_ms']:.2f} ms")
    print()
    print(f"  {'Class':<15} {'Acc%':>6}  {'Prec%':>6}  {'Recall%':>7}  {'F1%':>5}  {'n':>5}")
    sep()
    for c, v in d["per_class"].items():
        bar = "█" * int(v["accuracy"] / 5)
        print(f"  {c:<15} {v['accuracy']:>6.1f}  {v['precision']:>6.1f}  "
              f"{v['recall']:>7.1f}  {v['f1']:>5.1f}  {v['total']:>5}  {bar}")


def show_readiness(r: Dict):
    print(f"\n  ── Pi 4 Readiness Checklist ──")
    print(f"  Model size : {r['model_mb']:.2f} MB")
    icons = {True: "✓ PASS", False: "✗ FAIL"}
    for k, v in r["checks"].items():
        print(f"  [{icons[v]}]  {k}")
    all_pass = all(r["checks"].values())
    print(f"\n  Result: {'ALL CHECKS PASSED ✓' if all_pass else str(sum(r['checks'].values())) + '/' + str(len(r['checks'])) + ' passed'}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Pi 4 latency & accuracy benchmark")
    ap.add_argument("--model",    required=True,       help="Path to INT8 .onnx model")
    ap.add_argument("--fp32",     default=None,        help="Path to FP32 .onnx (optional)")
    ap.add_argument("--test_dir", default=None,        help="Test images directory (optional)")
    ap.add_argument("--samples",  type=int, default=300, help="Images to sample for accuracy (default 300)")
    ap.add_argument("--warmup",   type=int, default=20,  help="Warm-up runs (default 20)")
    ap.add_argument("--runs",     type=int, default=500, help="Latency benchmark runs (default 500)")
    ap.add_argument("--out",      default="pi_results.json", help="Output JSON path")
    args = ap.parse_args()

    hdr(f"Raspberry Pi 4  —  Model Benchmark")
    print(f"  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model (INT8) : {args.model}")
    print(f"  ONNX Runtime : {ort.__version__}")
    t_start = cpu_temp()
    f_start = cpu_freq()
    ram_start = mem_mb()
    if t_start:  print(f"  CPU temp     : {t_start:.1f} °C")
    if f_start:  print(f"  CPU freq     : {f_start:.0f} MHz")
    if ram_start:print(f"  RAM (RSS)    : {ram_start:.1f} MB")

    results: Dict = {
        "timestamp"      : datetime.now().isoformat(),
        "model_int8"     : args.model,
        "ort_version"    : ort.__version__,
        "cpu_temp_start" : t_start,
        "cpu_freq_mhz"   : f_start,
    }

    # ── INT8 latency ──────────────────────────────────────────────────────────
    hdr("INT8 Latency Benchmark")
    sess_int8 = make_session(args.model)
    lat_int8  = bench_latency(sess_int8, args.warmup, args.runs)
    show_latency("INT8 Model", lat_int8)
    results["int8_latency"] = lat_int8

    # ── INT8 accuracy (optional) ──────────────────────────────────────────────
    acc_int8 = None
    if args.test_dir:
        hdr("INT8 Accuracy Evaluation")
        acc_int8 = test_accuracy(sess_int8, args.test_dir, args.samples)
        show_accuracy("INT8", acc_int8)
        results["int8_accuracy"] = acc_int8

    # ── FP32 comparison (optional) ────────────────────────────────────────────
    if args.fp32:
        hdr("FP32 Latency Benchmark")
        sess_fp32 = make_session(args.fp32)
        lat_fp32  = bench_latency(sess_fp32, args.warmup, args.runs)
        show_latency("FP32 Model", lat_fp32)
        results["fp32_latency"] = lat_fp32

        if args.test_dir:
            hdr("FP32 Accuracy Evaluation")
            acc_fp32 = test_accuracy(sess_fp32, args.test_dir, args.samples)
            show_accuracy("FP32", acc_fp32)
            results["fp32_accuracy"] = acc_fp32

        speedup = lat_fp32["mean_ms"] / lat_int8["mean_ms"]
        size_fp32 = os.path.getsize(args.fp32) / 1e6
        size_int8 = os.path.getsize(args.model) / 1e6
        print(f"\n  ── FP32 vs INT8 Summary ──")
        print(f"  Size   : {size_fp32:.2f} MB  →  {size_int8:.2f} MB  "
              f"({size_fp32/size_int8:.1f}× smaller)")
        print(f"  Latency: {lat_fp32['mean_ms']:.2f} ms  →  {lat_int8['mean_ms']:.2f} ms  "
              f"({speedup:.2f}× faster)")
        if acc_int8 and "accuracy_pct" in acc_int8 and "fp32_accuracy" in results:
            acc_diff = acc_int8["accuracy_pct"] - results["fp32_accuracy"].get("accuracy_pct", 0)
            sign = "+" if acc_diff >= 0 else ""
            print(f"  Accuracy delta: {sign}{acc_diff:.2f} pp  (INT8 - FP32)")
        results["speedup_x"] = speedup

    # ── Readiness checklist ───────────────────────────────────────────────────
    model_mb = os.path.getsize(args.model) / 1e6
    checks = {
        "Model size ≤ 15 MB"       : model_mb <= 15.0,
        "Mean latency ≤ 400 ms"    : lat_int8["mean_ms"] <= 400.0,
        "Throughput ≥ 2 FPS"       : lat_int8["fps"] >= 2.0,
        "P95 latency ≤ 500 ms"     : lat_int8["p95_ms"] <= 500.0,
    }
    if acc_int8 and "accuracy_pct" in acc_int8:
        checks["Accuracy ≥ 90%"] = acc_int8["accuracy_pct"] >= 90.0

    hdr("Pi 4 Deployment Readiness")
    readiness = {"checks": checks, "model_mb": model_mb,
                 "passed": sum(checks.values()), "total": len(checks)}
    show_readiness(readiness)
    results["readiness"] = readiness

    # ── Final system info ──────────────────────────────────────────────────────
    t_end = cpu_temp()
    if t_end:
        print(f"\n  CPU temp (end)  : {t_end:.1f} °C")
        results["cpu_temp_end"] = t_end

    # ── Save results ───────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {args.out}")
    print()


if __name__ == "__main__":
    main()

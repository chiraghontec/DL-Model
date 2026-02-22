#!/usr/bin/env python3
"""
benchmark_pi.py

Comprehensive benchmark script for measuring model performance.
Works on both development machines (macOS/Linux) and Raspberry Pi 4.

Metrics collected:
  - Inference latency (mean, median, P50, P95, P99, min, max)
  - Throughput (FPS)
  - Memory usage (RSS, peak)
  - CPU temperature (Raspberry Pi only)
  - Model file size
  - Accuracy comparison (FP32 vs INT8, if test dataset available)

Outputs:
  - benchmark_results.json (machine-readable)
  - benchmark_report.txt (human-readable summary)
  - latency_distribution.png (histogram)

Usage:
  # Quick benchmark (random inputs):
  python benchmark_pi.py --onnx models/resnet18_tomato_int8.onnx

  # Full benchmark with accuracy check:
  python benchmark_pi.py --onnx models/resnet18_tomato_int8.onnx --onnx_fp32 models/resnet18_tomato_fp32.onnx --test_dir data/test

  # Compare FP32 vs INT8:
  python benchmark_pi.py --onnx models/resnet18_tomato_int8.onnx --onnx_fp32 models/resnet18_tomato_fp32.onnx --compare
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring disabled.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# System Information
# ---------------------------------------------------------------------------

def get_system_info() -> Dict[str, Any]:
    """Collect system information."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
    }

    # Detect Raspberry Pi
    info["is_raspberry_pi"] = False
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_str = f.read().strip()
            info["device_model"] = model_str
            if "raspberry" in model_str.lower():
                info["is_raspberry_pi"] = True
    except FileNotFoundError:
        info["device_model"] = platform.node()

    # CPU info
    if HAS_PSUTIL:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 2)
        info["ram_available_gb"] = round(mem.available / (1024**3), 2)

    # CPU frequency
    try:
        if HAS_PSUTIL:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = round(freq.current, 0)
    except Exception:
        pass

    return info


def get_cpu_temperature() -> Optional[float]:
    """Read CPU temperature (Raspberry Pi / Linux)."""
    # Raspberry Pi / Linux thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_milli = int(f.read().strip())
            return temp_milli / 1000.0
    except FileNotFoundError:
        pass

    # macOS (approximate via powermetrics — requires sudo, skip)
    return None


def get_memory_usage() -> Dict[str, float]:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return {}

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    return {
        "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
        "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Benchmarking Functions
# ---------------------------------------------------------------------------

def benchmark_latency(
    onnx_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    n_warmup: int = 20,
    n_runs: int = 200,
    label: str = "Model",
) -> Dict[str, Any]:
    """
    Benchmark inference latency.

    Args:
        onnx_path: Path to ONNX model.
        input_shape: Input tensor shape.
        n_warmup: Number of warmup iterations.
        n_runs: Number of timed iterations.
        label: Label for this benchmark.

    Returns:
        Dict with latency statistics.
    """
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    print(f"  [{label}] Warming up ({n_warmup} iterations)...", end="", flush=True)
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})
    print(" done")

    # Record memory before
    mem_before = get_memory_usage()

    # Timed runs
    print(f"  [{label}] Benchmarking ({n_runs} iterations)...", end="", flush=True)
    latencies = []
    temp_start = get_cpu_temperature()

    for i in range(n_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    print(" done")

    temp_end = get_cpu_temperature()
    mem_after = get_memory_usage()

    latencies = np.array(latencies)

    result = {
        "label": label,
        "model_path": onnx_path,
        "model_size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
        "n_runs": n_runs,
        "latency_ms": {
            "mean": round(float(latencies.mean()), 3),
            "median": round(float(np.median(latencies)), 3),
            "std": round(float(latencies.std()), 3),
            "min": round(float(latencies.min()), 3),
            "max": round(float(latencies.max()), 3),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p90": round(float(np.percentile(latencies, 90)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
        },
        "throughput_fps": round(1000.0 / float(latencies.mean()), 1),
        "memory": {
            "before": mem_before,
            "after": mem_after,
        },
        "raw_latencies": latencies.tolist(),
    }

    if temp_start is not None:
        result["cpu_temp_start_c"] = temp_start
    if temp_end is not None:
        result["cpu_temp_end_c"] = temp_end

    return result


def benchmark_accuracy(
    onnx_path: str,
    test_dir: str,
    class_names: List[str],
    input_size: int = 224,
    label: str = "Model",
) -> Dict[str, Any]:
    """
    Benchmark model accuracy on test dataset.

    Args:
        onnx_path: Path to ONNX model.
        test_dir: Path to test directory with class subfolders.
        class_names: Sorted class names.
        input_size: Input image size.
        label: Label for this benchmark.

    Returns:
        Dict with accuracy metrics.
    """
    from PIL import Image
    from torchvision import transforms

    print(f"  [{label}] Running accuracy benchmark on {test_dir}...", end="", flush=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    per_class_correct = {cls: 0 for cls in class_names}
    per_class_total = {cls: 0 for cls in class_names}
    confidences = []

    for cls_idx, cls_name in enumerate(sorted(class_names)):
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(cls_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                tensor = transform(img).unsqueeze(0).numpy()
                output = session.run(None, {input_name: tensor})[0]

                # Softmax
                exp_out = np.exp(output - np.max(output))
                probs = exp_out / exp_out.sum()
                pred = int(np.argmax(probs))
                conf = float(probs[0, pred])

                per_class_total[cls_name] += 1
                total += 1
                confidences.append(conf)

                if pred == cls_idx:
                    correct += 1
                    per_class_correct[cls_name] += 1

            except Exception:
                pass

    print(" done")

    accuracy = correct / total if total > 0 else 0
    per_class_acc = {}
    for cls in class_names:
        if per_class_total[cls] > 0:
            per_class_acc[cls] = round(per_class_correct[cls] / per_class_total[cls], 4)
        else:
            per_class_acc[cls] = None

    confidences = np.array(confidences)

    return {
        "label": label,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_acc,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
        "confidence_mean": round(float(confidences.mean()), 4) if len(confidences) > 0 else 0,
        "confidence_median": round(float(np.median(confidences)), 4) if len(confidences) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_latency_distribution(
    results: List[Dict],
    output_path: str = "latency_distribution.png",
):
    """Plot latency distribution histogram."""
    if not HAS_MATPLOTLIB:
        print("  Skipping plot (matplotlib not available)")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for ax, result, color in zip(axes, results, colors):
        latencies = np.array(result["raw_latencies"])
        label = result["label"]

        ax.hist(latencies, bins=50, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(latencies.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {latencies.mean():.1f}ms")
        ax.axvline(np.percentile(latencies, 95), color="orange", linestyle=":", linewidth=1.5, label=f"P95: {np.percentile(latencies, 95):.1f}ms")

        # Target line (400ms for Pi)
        if latencies.mean() > 100:
            ax.axvline(400, color="green", linestyle="-.", linewidth=1.5, label="Target: 400ms")

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}\n{result['model_size_mb']} MB | {result['throughput_fps']} FPS")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Latency plot saved: {output_path}")


def plot_comparison(
    results: List[Dict],
    output_path: str = "benchmark_comparison.png",
):
    """Plot FP32 vs INT8 comparison bar chart."""
    if not HAS_MATPLOTLIB or len(results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = [r["label"] for r in results]
    colors = ["#2196F3", "#FF5722"]

    # Latency comparison
    means = [r["latency_ms"]["mean"] for r in results]
    p95s = [r["latency_ms"]["p95"] for r in results]
    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width/2, means, width, label="Mean", color=colors)
    axes[0].bar(x + width/2, p95s, width, label="P95", color=[c + "80" for c in colors], alpha=0.7)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend()

    # Size comparison
    sizes = [r["model_size_mb"] for r in results]
    axes[1].bar(labels, sizes, color=colors)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Model Size")
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    # FPS comparison
    fps_vals = [r["throughput_fps"] for r in results]
    axes[2].bar(labels, fps_vals, color=colors)
    axes[2].set_ylabel("FPS")
    axes[2].set_title("Throughput")
    axes[2].axhline(y=2, color="green", linestyle="--", label="Target: 2 FPS")
    axes[2].legend()
    for i, v in enumerate(fps_vals):
        axes[2].text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    sys_info: Dict,
    latency_results: List[Dict],
    accuracy_results: Optional[List[Dict]] = None,
    output_path: str = "benchmark_report.txt",
):
    """Generate human-readable benchmark report."""
    lines = []
    lines.append("=" * 65)
    lines.append("  BENCHMARK REPORT — Tomato Blight Classifier")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 65)

    # System info
    lines.append("\n── System Information ──")
    lines.append(f"  Platform:     {sys_info.get('platform')}")
    lines.append(f"  Device:       {sys_info.get('device_model')}")
    lines.append(f"  CPU:          {sys_info.get('processor')}")
    lines.append(f"  CPU Cores:    {sys_info.get('cpu_count_physical', '?')} physical / {sys_info.get('cpu_count_logical', '?')} logical")
    lines.append(f"  RAM:          {sys_info.get('ram_total_gb', '?')} GB total, {sys_info.get('ram_available_gb', '?')} GB available")
    lines.append(f"  Raspberry Pi: {'Yes' if sys_info.get('is_raspberry_pi') else 'No'}")
    lines.append(f"  ONNX Runtime: {sys_info.get('onnxruntime_version')}")
    lines.append(f"  Python:       {sys_info.get('python_version')}")

    # Latency results
    lines.append("\n── Inference Latency ──")
    for result in latency_results:
        lat = result["latency_ms"]
        lines.append(f"\n  {result['label']} ({result['model_size_mb']} MB):")
        lines.append(f"    Mean:       {lat['mean']:.2f} ms")
        lines.append(f"    Median:     {lat['median']:.2f} ms")
        lines.append(f"    Std:        {lat['std']:.2f} ms")
        lines.append(f"    Min:        {lat['min']:.2f} ms")
        lines.append(f"    Max:        {lat['max']:.2f} ms")
        lines.append(f"    P90:        {lat['p90']:.2f} ms")
        lines.append(f"    P95:        {lat['p95']:.2f} ms")
        lines.append(f"    P99:        {lat['p99']:.2f} ms")
        lines.append(f"    Throughput: {result['throughput_fps']:.1f} FPS")

        if "cpu_temp_start_c" in result and "cpu_temp_end_c" in result:
            lines.append(f"    CPU Temp:   {result['cpu_temp_start_c']:.1f}°C → {result['cpu_temp_end_c']:.1f}°C")

        mem = result.get("memory", {})
        if mem.get("after"):
            lines.append(f"    Memory:     {mem['after'].get('rss_mb', '?')} MB RSS")

    # Comparison
    if len(latency_results) >= 2:
        fp32 = latency_results[0]
        int8 = latency_results[1]
        speedup = fp32["latency_ms"]["mean"] / int8["latency_ms"]["mean"] if int8["latency_ms"]["mean"] > 0 else 0
        compression = fp32["model_size_mb"] / int8["model_size_mb"] if int8["model_size_mb"] > 0 else 0

        lines.append("\n── FP32 vs INT8 Comparison ──")
        lines.append(f"  Speedup:     {speedup:.2f}×")
        lines.append(f"  Compression: {compression:.1f}× ({fp32['model_size_mb']} → {int8['model_size_mb']} MB)")

    # Accuracy results
    if accuracy_results:
        lines.append("\n── Accuracy ──")
        for result in accuracy_results:
            lines.append(f"\n  {result['label']}:")
            lines.append(f"    Overall:    {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
            lines.append(f"    Confidence: {result['confidence_mean']:.4f} (mean)")
            lines.append(f"    Per-class:")
            for cls, acc in result["per_class_accuracy"].items():
                n = result["per_class_total"][cls]
                c = result["per_class_correct"][cls]
                lines.append(f"      {cls:15s}: {acc:.4f} ({c}/{n})" if acc is not None else f"      {cls:15s}: N/A")

        if len(accuracy_results) >= 2:
            drop = accuracy_results[0]["accuracy"] - accuracy_results[1]["accuracy"]
            lines.append(f"\n  Accuracy Drop (INT8 vs FP32): {drop:.4f} ({drop*100:.2f}%)")
            status = "✓ PASS" if abs(drop) <= 0.02 else "✗ FAIL"
            lines.append(f"  Tolerance Check (≤2%): {status}")

    # Deployment readiness
    lines.append("\n── Deployment Readiness (Raspberry Pi 4) ──")
    int8_result = latency_results[-1]  # Last result, typically INT8
    checks = {
        "Model size ≤ 15 MB": int8_result["model_size_mb"] <= 15,
        "Mean latency ≤ 400 ms": int8_result["latency_ms"]["mean"] <= 400,
        "Throughput ≥ 2 FPS": int8_result["throughput_fps"] >= 2,
        "P95 latency ≤ 500 ms": int8_result["latency_ms"]["p95"] <= 500,
    }

    if accuracy_results:
        acc = accuracy_results[-1]["accuracy"]
        checks["Accuracy ≥ 90%"] = acc >= 0.90

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        lines.append(f"  {status} {check}")

    all_pass = all(checks.values())
    lines.append(f"\n  {'✓ ALL CHECKS PASSED — Ready for deployment' if all_pass else '✗ SOME CHECKS FAILED — See above'}")

    lines.append(f"\n{'='*65}")

    report_text = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Report saved: {output_path}")

    return report_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX model performance")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model (INT8 or FP32)")
    parser.add_argument("--onnx_fp32", type=str, default=None, help="Path to FP32 ONNX model (for comparison)")
    parser.add_argument("--test_dir", type=str, default=None, help="Test dataset directory for accuracy benchmark")
    parser.add_argument("--classes", nargs="+", default=["early_blight", "healthy", "late_blight"])
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--compare", action="store_true", help="Force comparison mode (requires --onnx_fp32)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Benchmark — Tomato Blight Classifier")
    print("=" * 60)

    # Collect system info
    print("\n[1] Collecting system information...")
    sys_info = get_system_info()

    # Determine models to benchmark
    models_to_benchmark = []
    if args.onnx_fp32 and os.path.exists(args.onnx_fp32):
        models_to_benchmark.append(("FP32", args.onnx_fp32))
    models_to_benchmark.append(("INT8" if args.onnx_fp32 else "Model", args.onnx))

    # Latency benchmarks
    print(f"\n[2] Running latency benchmarks ({args.n_runs} runs each)...")
    latency_results = []
    for label, path in models_to_benchmark:
        result = benchmark_latency(path, n_warmup=args.n_warmup, n_runs=args.n_runs, label=label)
        latency_results.append(result)

    # Accuracy benchmarks
    accuracy_results = None
    if args.test_dir and os.path.isdir(args.test_dir):
        print(f"\n[3] Running accuracy benchmarks...")
        accuracy_results = []
        for label, path in models_to_benchmark:
            result = benchmark_accuracy(path, args.test_dir, args.classes, label=label)
            accuracy_results.append(result)

    # Visualizations
    print(f"\n[4] Generating visualizations...")
    plot_latency_distribution(
        latency_results,
        os.path.join(args.output_dir, "latency_distribution.png"),
    )
    if len(latency_results) >= 2:
        plot_comparison(
            latency_results,
            os.path.join(args.output_dir, "benchmark_comparison.png"),
        )

    # Generate report
    print(f"\n[5] Generating report...")

    # Clean latency results for JSON (remove raw_latencies for JSON file)
    json_results = []
    for r in latency_results:
        r_clean = {k: v for k, v in r.items() if k != "raw_latencies"}
        json_results.append(r_clean)

    # Save JSON
    json_output = {
        "timestamp": datetime.now().isoformat(),
        "system_info": sys_info,
        "latency_benchmarks": json_results,
    }
    if accuracy_results:
        json_output["accuracy_benchmarks"] = accuracy_results

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"  JSON results: {json_path}")

    # Generate text report
    generate_report(
        sys_info,
        latency_results,
        accuracy_results,
        os.path.join(args.output_dir, "benchmark_report.txt"),
    )


if __name__ == "__main__":
    main()

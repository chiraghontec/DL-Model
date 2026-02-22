#!/usr/bin/env python3
"""
quantize_onnx.py

Apply INT8 dynamic quantization to the FP32 ONNX model to reduce size and
improve inference speed on Raspberry Pi 4.

Pipeline:
  1. Load FP32 ONNX model
  2. Apply ONNX Runtime dynamic quantization (QUInt8)
  3. Validate: compare INT8 vs FP32 outputs, measure accuracy drop
  4. Benchmark latency comparison
  5. Optional: test-set accuracy validation (if dataset available)

Expected Results:
  FP32: ~44 MB → INT8: ~11 MB (4× compression)
  Accuracy drop: ≤2%

Usage:
  python quantize_onnx.py --input models/resnet18_tomato_fp32.onnx --output models/resnet18_tomato_int8.onnx
  python quantize_onnx.py --input models/resnet18_tomato_fp32.onnx --validate_dataset data/test
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(input_path: str, output_path: str, quant_type: str = "QUInt8"):
    """
    Apply dynamic quantization to ONNX model.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path to save INT8 quantized model.
        quant_type: Quantization type ("QUInt8" or "QInt8").
    """
    print(f"\n[1/4] Applying dynamic quantization ({quant_type})...")
    start = time.time()

    qt = QuantType.QUInt8 if quant_type == "QUInt8" else QuantType.QInt8

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=qt,
        per_channel=True,
        reduce_range=False,
    )

    elapsed = time.time() - start
    fp32_size = os.path.getsize(input_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    compression_ratio = fp32_size / int8_size if int8_size > 0 else 0

    print(f"  Quantization completed in {elapsed:.1f}s")
    print(f"  FP32 size:          {fp32_size:.2f} MB")
    print(f"  INT8 size:          {int8_size:.2f} MB")
    print(f"  Compression ratio:  {compression_ratio:.1f}×")

    return {
        "fp32_size_mb": round(fp32_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "compression_ratio": round(compression_ratio, 1),
    }


def validate_structure(onnx_path: str):
    """Validate quantized ONNX model structure."""
    print(f"\n[2/4] Validating quantized model structure...")
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"  ✓ Quantized model passes structural validation")

    # Count quantized ops
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    quant_ops = {k: v for k, v in op_types.items() if "Quant" in k or "Integer" in k or "QLinear" in k}
    print(f"  Total ops:     {len(model.graph.node)}")
    print(f"  Quantized ops: {quant_ops if quant_ops else 'dynamic quant (weights only)'}")

    return op_types


def compare_fp32_vs_int8(
    fp32_path: str,
    int8_path: str,
    n_samples: int = 50,
    input_shape: tuple = (1, 3, 224, 224),
):
    """Compare FP32 and INT8 model outputs on random inputs."""
    print(f"\n[3/4] Comparing FP32 vs INT8 outputs ({n_samples} random samples)...")

    fp32_session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    input_name = fp32_session.get_inputs()[0].name

    max_diffs = []
    mean_diffs = []
    class_agreements = 0

    for i in range(n_samples):
        dummy = np.random.randn(*input_shape).astype(np.float32)

        fp32_out = fp32_session.run(None, {input_name: dummy})[0]
        int8_out = int8_session.run(None, {input_name: dummy})[0]

        max_diffs.append(np.max(np.abs(fp32_out - int8_out)))
        mean_diffs.append(np.mean(np.abs(fp32_out - int8_out)))

        if np.argmax(fp32_out) == np.argmax(int8_out):
            class_agreements += 1

    max_diffs = np.array(max_diffs)
    mean_diffs = np.array(mean_diffs)
    agreement_rate = class_agreements / n_samples

    print(f"  Max absolute diff (avg):  {max_diffs.mean():.6f}")
    print(f"  Max absolute diff (max):  {max_diffs.max():.6f}")
    print(f"  Mean absolute diff (avg): {mean_diffs.mean():.6f}")
    print(f"  Class agreement rate:     {agreement_rate:.2%} ({class_agreements}/{n_samples})")

    return {
        "max_diff_avg": float(max_diffs.mean()),
        "max_diff_max": float(max_diffs.max()),
        "mean_diff_avg": float(mean_diffs.mean()),
        "class_agreement_rate": float(agreement_rate),
    }


def benchmark_latency(
    fp32_path: str,
    int8_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    n_runs: int = 100,
):
    """Compare latency between FP32 and INT8 models."""
    print(f"\n[4/4] Benchmarking FP32 vs INT8 latency ({n_runs} runs each)...")

    results = {}
    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy})

        # Timed runs
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            session.run(None, {input_name: dummy})
            latencies.append((time.time() - start) * 1000)

        latencies = np.array(latencies)
        results[label] = {
            "mean_ms": float(latencies.mean()),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "std_ms": float(latencies.std()),
        }

        print(f"\n  {label}:")
        print(f"    Mean:   {results[label]['mean_ms']:.2f} ms")
        print(f"    Median: {results[label]['median_ms']:.2f} ms")
        print(f"    P95:    {results[label]['p95_ms']:.2f} ms")

    speedup = results["FP32"]["mean_ms"] / results["INT8"]["mean_ms"] if results["INT8"]["mean_ms"] > 0 else 0
    print(f"\n  Speedup (INT8 vs FP32): {speedup:.2f}×")
    results["speedup"] = round(speedup, 2)

    return results


def validate_on_dataset(
    fp32_path: str,
    int8_path: str,
    dataset_dir: str,
    class_names: list,
    input_size: int = 224,
):
    """
    Validate accuracy of FP32 and INT8 on actual test dataset.

    Args:
        dataset_dir: Path to test data directory (class subfolders).
        class_names: List of class names matching folder names.
    """
    from PIL import Image
    from torchvision import transforms

    print(f"\n[Bonus] Validating accuracy on test dataset: {dataset_dir}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test images
    images = []
    labels = []
    for cls_idx, cls_name in enumerate(sorted(class_names)):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"  Warning: class directory not found: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(fpath)
                labels.append(cls_idx)

    if not images:
        print("  No images found. Skipping dataset validation.")
        return None

    print(f"  Found {len(images)} test images across {len(set(labels))} classes")

    # Run inference with both models
    results = {}
    for label, path in [("FP32", fp32_path), ("INT8", int8_path)]:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        correct = 0
        total = 0
        for img_path, true_label in zip(images, labels):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).numpy()
                output = session.run(None, {input_name: tensor})[0]
                pred = np.argmax(output, axis=1)[0]
                if pred == true_label:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"  Warning: Failed to process {img_path}: {e}")

        accuracy = correct / total if total > 0 else 0
        results[label] = {"accuracy": accuracy, "correct": correct, "total": total}
        print(f"  {label} Accuracy: {accuracy:.4f} ({correct}/{total})")

    if "FP32" in results and "INT8" in results:
        drop = results["FP32"]["accuracy"] - results["INT8"]["accuracy"]
        print(f"  Accuracy drop (INT8 vs FP32): {drop:.4f} ({drop*100:.2f}%)")
        results["accuracy_drop"] = float(drop)

        if abs(drop) <= 0.02:
            print(f"  ✓ Accuracy drop within 2% tolerance")
        else:
            print(f"  ✗ WARNING: Accuracy drop exceeds 2% tolerance")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument(
        "--input", required=True, help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output INT8 ONNX path (default: <input_dir>/resnet18_tomato_int8.onnx)",
    )
    parser.add_argument(
        "--quant_type", choices=["QUInt8", "QInt8"], default="QUInt8",
        help="Quantization type (default: QUInt8)",
    )
    parser.add_argument(
        "--validate_dataset", type=str, default=None,
        help="Optional: path to test dataset directory for accuracy validation",
    )
    parser.add_argument(
        "--classes", nargs="+", default=["early_blight", "healthy", "late_blight"],
        help="Class names in sorted order",
    )
    parser.add_argument("--benchmark_runs", type=int, default=100)
    args = parser.parse_args()

    # Output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        # Replace fp32 with int8 in name
        if "fp32" in base:
            args.output = base.replace("fp32", "int8") + ".onnx"
        else:
            args.output = base + "_int8.onnx"

    print("=" * 60)
    print("ONNX INT8 Quantization — Tomato Blight Classifier")
    print("=" * 60)
    print(f"  Input (FP32):   {args.input}")
    print(f"  Output (INT8):  {args.output}")
    print(f"  Quant Type:     {args.quant_type}")

    # Step 1: Quantize
    size_info = quantize_model(args.input, args.output, args.quant_type)

    # Step 2: Validate structure
    op_types = validate_structure(args.output)

    # Step 3: Compare outputs
    compare_info = compare_fp32_vs_int8(args.input, args.output)

    # Step 4: Benchmark latency
    latency_info = benchmark_latency(args.input, args.output, n_runs=args.benchmark_runs)

    # Optional: Dataset validation
    dataset_info = None
    if args.validate_dataset and os.path.isdir(args.validate_dataset):
        dataset_info = validate_on_dataset(
            args.input, args.output, args.validate_dataset, args.classes
        )

    # Save report
    report = {
        "fp32_model": args.input,
        "int8_model": args.output,
        "quant_type": args.quant_type,
        "size": size_info,
        "output_comparison": compare_info,
        "latency": latency_info,
    }
    if dataset_info:
        report["dataset_validation"] = dataset_info

    report_path = os.path.splitext(args.output)[0] + "_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    print(f"\n{'='*60}")
    print(f"  Quantization complete!")
    print(f"  FP32: {size_info['fp32_size_mb']:.2f} MB → INT8: {size_info['int8_size_mb']:.2f} MB ({size_info['compression_ratio']}× compression)")
    print(f"  Class agreement: {compare_info['class_agreement_rate']:.2%}")
    print(f"  Speedup: {latency_info.get('speedup', 'N/A')}×")
    if dataset_info and "accuracy_drop" in dataset_info:
        print(f"  Accuracy drop: {dataset_info['accuracy_drop']*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

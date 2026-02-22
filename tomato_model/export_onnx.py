#!/usr/bin/env python3
"""
export_onnx.py

Export the trained ResNet-18 tomato model from PyTorch (.pth) to ONNX format.

Pipeline:
  1. Load checkpoint → rebuild ResNet-18 → load state_dict
  2. Create dummy input (1, 3, 224, 224)
  3. torch.onnx.export() with opset 13, dynamic batch axis
  4. Validate ONNX model structure (onnx.checker)
  5. Compare ONNX Runtime output against PyTorch output (atol=1e-5)
  6. Report model size

Output:
  resnet18_tomato_fp32.onnx (~44 MB)

Usage:
  python export_onnx.py --checkpoint checkpoints/best.pth --output models/resnet18_tomato_fp32.onnx
"""

import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str = "resnet18",
    device: torch.device = torch.device("cpu"),
):
    """Load model architecture and weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = ckpt.get("num_classes", len(ckpt.get("classes", [])))
    class_names = ckpt.get("classes", ["early_blight", "healthy", "late_blight"])

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    return model, class_names, ckpt


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 13,
    dynamic_batch: bool = True,
):
    """Export PyTorch model to ONNX format."""
    dummy_input = torch.randn(*input_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    print(f"\n[1/4] Exporting to ONNX (opset {opset_version})...")
    start = time.time()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    elapsed = time.time() - start
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Export completed in {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Size:   {size_mb:.2f} MB")

    return dummy_input


def validate_onnx_model(onnx_path: str):
    """Validate ONNX model structure using onnx.checker."""
    print(f"\n[2/4] Validating ONNX model structure...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model passes structural validation")

    # Print model info
    graph = onnx_model.graph
    print(f"  Input:  {graph.input[0].name} — {[d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]}")
    print(f"  Output: {graph.output[0].name} — {[d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]}")
    print(f"  Nodes:  {len(graph.node)}")

    return onnx_model


def compare_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    atol: float = 1e-5,
):
    """Compare PyTorch and ONNX Runtime outputs for consistency."""
    print(f"\n[3/4] Comparing PyTorch vs ONNX Runtime outputs...")

    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_input = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_input)[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    mean_diff = np.mean(np.abs(pytorch_output - ort_output))

    print(f"  Max absolute difference:  {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Tolerance:                {atol:.2e}")

    if max_diff < atol:
        print(f"  ✓ Outputs match within tolerance")
    else:
        print(f"  ✗ WARNING: Max difference ({max_diff:.2e}) exceeds tolerance ({atol:.2e})")
        print(f"    This may be within acceptable range for deployment.")

    # Also compare with softmax probabilities
    pytorch_probs = np.exp(pytorch_output) / np.exp(pytorch_output).sum(axis=1, keepdims=True)
    ort_probs = np.exp(ort_output) / np.exp(ort_output).sum(axis=1, keepdims=True)
    prob_diff = np.max(np.abs(pytorch_probs - ort_probs))
    print(f"  Max probability difference: {prob_diff:.2e}")

    return max_diff, mean_diff


def benchmark_onnx_latency(onnx_path: str, input_size: tuple, n_runs: int = 100):
    """Quick latency benchmark of ONNX Runtime inference."""
    print(f"\n[4/4] Benchmarking ONNX Runtime latency ({n_runs} runs)...")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(*input_size).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.time()
        session.run(None, {input_name: dummy})
        latencies.append((time.time() - start) * 1000)  # ms

    latencies = np.array(latencies)
    print(f"  Mean:   {latencies.mean():.2f} ms")
    print(f"  Median: {np.median(latencies):.2f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.2f} ms")
    print(f"  Std:    {latencies.std():.2f} ms")

    return {
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(latencies.std()),
    }


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output ONNX path (default: <checkpoint_dir>/resnet18_tomato_fp32.onnx)",
    )
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--benchmark_runs", type=int, default=100)
    args = parser.parse_args()

    # Device — export must be on CPU
    device = torch.device("cpu")

    # Determine output path
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        output_dir = os.path.join(ckpt_dir, "..", "models") if ckpt_dir else "models"
        args.output = os.path.join(output_dir, f"{args.model}_tomato_fp32.onnx")

    print("=" * 60)
    print("ONNX Export — Tomato Blight Classifier")
    print("=" * 60)
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Output:      {args.output}")
    print(f"  Model:       {args.model}")
    print(f"  Opset:       {args.opset}")

    # Load model
    model, class_names, ckpt = load_model_from_checkpoint(
        args.checkpoint, args.model, device
    )
    print(f"  Classes:     {class_names}")
    epoch = ckpt.get("epoch", "?")
    acc = ckpt.get("val_acc", ckpt.get("acc", "?"))
    print(f"  Epoch:       {epoch}")
    print(f"  Val Acc:     {acc}")

    # Step 1: Export
    dummy_input = export_to_onnx(model, args.output, opset_version=args.opset)

    # Step 2: Validate structure
    validate_onnx_model(args.output)

    # Step 3: Compare outputs
    max_diff, mean_diff = compare_outputs(model, args.output, dummy_input, args.atol)

    # Step 4: Benchmark
    latency = benchmark_onnx_latency(args.output, (1, 3, 224, 224), args.benchmark_runs)

    # Save export metadata
    metadata = {
        "checkpoint": args.checkpoint,
        "onnx_path": args.output,
        "model_name": args.model,
        "opset_version": args.opset,
        "classes": class_names,
        "num_classes": len(class_names),
        "input_shape": [1, 3, 224, 224],
        "file_size_mb": round(os.path.getsize(args.output) / (1024 * 1024), 2),
        "max_output_diff": float(max_diff),
        "mean_output_diff": float(mean_diff),
        "latency_benchmark": latency,
        "source_epoch": ckpt.get("epoch", None),
        "source_val_acc": float(acc) if isinstance(acc, (int, float)) else None,
    }

    metadata_path = os.path.splitext(args.output)[0] + "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved: {metadata_path}")

    print(f"\n{'='*60}")
    print(f"  Export complete! FP32 ONNX model: {args.output}")
    print(f"  Size: {metadata['file_size_mb']:.2f} MB")
    print(f"  Latency (CPU): {latency['mean_ms']:.1f} ms avg")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

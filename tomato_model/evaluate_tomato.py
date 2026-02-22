#!/usr/bin/env python3
"""
evaluate_tomato.py

Comprehensive evaluation of the tomato blight detection model on the test set.
Produces:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix visualization (saved as PNG)
  - Confidence distribution plot per class
  - Per-image prediction CSV
  - Summary table formatted for IEEE paper

Usage:
  python evaluate_tomato.py --data_root ./data --checkpoint checkpoints/best_epoch_X_acc_Y.pth
  python evaluate_tomato.py --data_root ./data --checkpoint checkpoints/best_epoch_X_acc_Y.pth --output_dir results/
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tomato blight model on test set"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to folder containing train/val/test subfolders",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .pth checkpoint file"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda, mps, or cpu (auto if None)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Filename for per-image predictions CSV",
    )
    return parser.parse_args()


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list, output_path: str, normalize: bool = True
):
    """Save confusion matrix as a high-quality PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (norm, title_suffix) in enumerate(
        [(False, "Counts"), (True, "Normalized")]
    ):
        ax = axes[ax_idx]
        if norm:
            cm_plot = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            fmt = ".2f"
        else:
            cm_plot = cm
            fmt = "d"

        im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix ({title_suffix})", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=10)

        # Add text annotations
        thresh = cm_plot.max() / 2.0
        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                val = cm_plot[i, j]
                txt = f"{val:{fmt}}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "black",
                    fontsize=11,
                )

        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {output_path}")


def plot_confidence_distribution(
    all_confs: list,
    all_labels: list,
    all_preds: list,
    class_names: list,
    output_path: str,
):
    """Plot confidence distribution per class, split by correct/incorrect."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(6 * len(class_names), 5))

    if len(class_names) == 1:
        axes = [axes]

    for idx, cls in enumerate(class_names):
        ax = axes[idx]

        # Correct predictions for this class
        correct_confs = [
            c
            for c, l, p in zip(all_confs, all_labels, all_preds)
            if l == idx and p == idx
        ]
        # Wrong predictions for this class
        wrong_confs = [
            c
            for c, l, p in zip(all_confs, all_labels, all_preds)
            if l == idx and p != idx
        ]

        bins = np.linspace(0, 1, 25)
        if correct_confs:
            ax.hist(
                correct_confs,
                bins=bins,
                alpha=0.7,
                color="green",
                label=f"Correct ({len(correct_confs)})",
                edgecolor="darkgreen",
            )
        if wrong_confs:
            ax.hist(
                wrong_confs,
                bins=bins,
                alpha=0.7,
                color="red",
                label=f"Wrong ({len(wrong_confs)})",
                edgecolor="darkred",
            )

        ax.axvline(x=0.70, color="orange", linestyle="--", linewidth=2, label="Threshold (0.70)")
        ax.set_title(f"{cls}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.suptitle("Confidence Distribution by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Confidence distribution saved: {output_path}")


def plot_training_history(history_path: str, output_path: str):
    """Plot training curves from training_history.json if it exists."""
    if not os.path.isfile(history_path):
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_acc, "b-o", markersize=3, label="Train Accuracy")
    ax1.plot(epochs, val_acc, "r-o", markersize=3, label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training & Validation Accuracy", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_loss, "b-o", markersize=3, label="Train Loss")
    ax2.plot(epochs, val_loss, "r-o", markersize=3, label="Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training & Validation Loss", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
    print(f"Device: {device}")

    # Test dataset
    test_path = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_path):
        raise SystemExit(f"Test folder not found: {test_path}")

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    if classes is None:
        classes = test_dataset.classes
    num_classes = len(classes)

    model_name = ckpt.get("model_name", args.model) if isinstance(ckpt, dict) else args.model

    print(f"Classes ({num_classes}): {classes}")
    print(f"Test samples: {len(test_dataset)}")

    # Build and load model
    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    # Run inference
    all_preds = []
    all_labels = []
    all_confs = []
    all_probs = []

    print("\nRunning inference on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  test"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    accuracy = accuracy_score(all_labels, all_preds)
    precision_w = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:            {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision (weighted): {precision_w:.4f}")
    print(f"  Recall (weighted):    {recall_w:.4f}")
    print(f"  F1-Score (weighted):  {f1_w:.4f}")
    print(f"{'='*60}")

    # Full classification report
    report = classification_report(
        all_labels, all_preds, target_names=classes, digits=4
    )
    print(f"\nClassification Report:\n{report}")

    # Save report to file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision (weighted): {precision_w:.4f}\n")
        f.write(f"Recall (weighted): {recall_w:.4f}\n")
        f.write(f"F1-Score (weighted): {f1_w:.4f}\n")
        f.write(f"\n{report}\n")
    print(f"  Report saved: {report_path}")

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path)

    # -----------------------------------------------------------------------
    # Confidence distribution
    # -----------------------------------------------------------------------
    conf_path = os.path.join(args.output_dir, "confidence_distribution.png")
    plot_confidence_distribution(all_confs, all_labels, all_preds, classes, conf_path)

    # -----------------------------------------------------------------------
    # Training curves (if history available)
    # -----------------------------------------------------------------------
    ckpt_dir = os.path.dirname(args.checkpoint)
    history_path = os.path.join(ckpt_dir, "training_history.json")
    curves_path = os.path.join(args.output_dir, "training_curves.png")
    plot_training_history(history_path, curves_path)

    # -----------------------------------------------------------------------
    # Confidence analysis
    # -----------------------------------------------------------------------
    print(f"\nConfidence Analysis:")
    conf_arr = np.array(all_confs)
    print(f"  Mean confidence: {conf_arr.mean():.4f}")
    print(f"  Median confidence: {np.median(conf_arr):.4f}")
    print(f"  Std confidence: {conf_arr.std():.4f}")
    print(f"  Min confidence: {conf_arr.min():.4f}")
    print(f"  Max confidence: {conf_arr.max():.4f}")

    # Below threshold analysis (PRD FR-02.4: threshold = 0.70)
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    print(f"\n  Samples below confidence thresholds:")
    for thresh in thresholds:
        below = (conf_arr < thresh).sum()
        pct = below / len(conf_arr) * 100
        print(f"    P < {thresh:.2f}: {below} samples ({pct:.2f}%)")

    # -----------------------------------------------------------------------
    # Per-image CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, args.output_csv)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["image_path", "true_class", "pred_class", "confidence", "correct"]
        header.extend([f"prob_{cls}" for cls in classes])
        writer.writerow(header)

        for i, (path, true_label) in enumerate(test_dataset.samples):
            pred_label = all_preds[i]
            conf = all_confs[i]
            correct = int(true_label == pred_label)
            row = [
                path,
                classes[true_label],
                classes[pred_label],
                f"{conf:.4f}",
                correct,
            ]
            row.extend([f"{p:.4f}" for p in all_probs[i]])
            writer.writerow(row)

    print(f"  Per-image predictions CSV: {csv_path}")

    # -----------------------------------------------------------------------
    # IEEE Paper Summary Table
    # -----------------------------------------------------------------------
    per_class_report = classification_report(
        all_labels, all_preds, target_names=classes, digits=4, output_dict=True
    )

    print(f"\n{'='*60}")
    print(f"IEEE PAPER TABLE — Test Set Performance")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'Value':>12}")
    print(f"{'-'*42}")
    print(f"{'Test Accuracy':<30} {accuracy*100:>11.2f}%")
    print(f"{'Precision (weighted)':<30} {precision_w*100:>11.2f}%")
    print(f"{'Recall (weighted)':<30} {recall_w*100:>11.2f}%")
    print(f"{'F1-Score (weighted)':<30} {f1_w*100:>11.2f}%")
    print(f"{'Mean Confidence':<30} {conf_arr.mean()*100:>11.2f}%")
    print(f"{'Samples below 0.70 threshold':<30} {(conf_arr < 0.70).sum():>12d}")
    print(f"{'-'*42}")
    for cls in classes:
        m = per_class_report[cls]
        print(f"{cls + ' F1':<30} {m['f1-score']*100:>11.2f}%")
    print(f"{'='*60}")

    # Save summary as JSON
    summary = {
        "model": model_name,
        "checkpoint": args.checkpoint,
        "num_test_samples": len(test_dataset),
        "classes": classes,
        "accuracy": accuracy,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "mean_confidence": float(conf_arr.mean()),
        "per_class": {cls: per_class_report[cls] for cls in classes},
        "confusion_matrix": cm.tolist(),
    }
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()

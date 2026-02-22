#!/usr/bin/env python3
"""
train_tomato.py

Training script for the 3-class Tomato Blight Detection model.
Classes: early_blight, late_blight, healthy

Key improvements over the original train.py:
  - Field-robust augmentations (ColorJitter, Rotation, GaussianBlur, RandomErasing, Affine)
  - Class-weighted CrossEntropyLoss for imbalanced data
  - Early stopping with configurable patience
  - Per-class metrics logged each epoch (precision, recall, F1)
  - Confidence distribution tracking

Usage:
  python train_tomato.py --data_root ./data --epochs 30 --batch_size 64
  python train_tomato.py --data_root ./data --epochs 30 --pretrained --save_every
  python train_tomato.py --data_root ./data --resume checkpoints/last_epoch_10.pth
"""

import argparse
import json
import os
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3-class tomato blight detector (ResNet-18)"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to folder containing train/val/test subfolders",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use ImageNet pretrained weights"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze feature extractor when using pretrained",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda, or mps (auto-detect if not provided)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (CUDA only)",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--save_every",
        action="store_true",
        help="Save a checkpoint every epoch (last_epoch_<n>.pth)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--seed", type=int, default=42)
    # --- New additions for tomato model ---
    parser.add_argument(
        "--class_weights",
        action="store_true",
        default=True,
        help="Use inverse-frequency class weights in loss (default: True)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class-weighted loss",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop training if val accuracy does not improve for N epochs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json for augmentation parameters",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed & reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(
    name: str, pretrained: bool, num_classes: int, freeze_backbone: bool
) -> nn.Module:
    """Build a torchvision ResNet model with modified classifier head."""
    weights = "IMAGENET1K_V1" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone and pretrained:
        for param_name, param in model.named_parameters():
            if not param_name.startswith("fc"):
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {name} | Total params: {total:,} | Trainable: {trainable:,}")

    return model


# ---------------------------------------------------------------------------
# Augmentation transforms (field-robust)
# ---------------------------------------------------------------------------
def get_train_transforms(config: dict = None) -> transforms.Compose:
    """
    Build training transforms with field-robustness augmentations.

    Additions over the original train.py:
      - ColorJitter (sunlight variation)
      - RandomRotation (leaf angles)
      - GaussianBlur (motion blur from rover)
      - RandomAffine (camera position jitter)
      - RandomErasing (occlusion by neighboring leaves)
    """
    aug = config.get("augmentation", {}) if config else {}

    transform_list = [
        transforms.Resize(aug.get("resize", 256)),
        transforms.RandomResizedCrop(aug.get("crop_size", 224), scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]

    # ColorJitter — simulates varying sunlight conditions (morning/noon/afternoon)
    cj = aug.get("color_jitter", {})
    transform_list.append(
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0.3),
            contrast=cj.get("contrast", 0.3),
            saturation=cj.get("saturation", 0.2),
            hue=cj.get("hue", 0.1),
        )
    )

    # RandomRotation — leaf orientation varies in field
    rot_deg = aug.get("rotation_degrees", 15)
    if rot_deg > 0:
        transform_list.append(transforms.RandomRotation(rot_deg))

    # GaussianBlur — motion blur from rover movement
    gb = aug.get("gaussian_blur", {})
    ks = gb.get("kernel_size", 3)
    sigma = gb.get("sigma", [0.1, 2.0])
    transform_list.append(
        transforms.GaussianBlur(kernel_size=ks, sigma=tuple(sigma))
    )

    # RandomAffine — camera position jitter on moving rover
    affine_translate = aug.get("random_affine_translate", [0.1, 0.1])
    transform_list.append(
        transforms.RandomAffine(degrees=0, translate=tuple(affine_translate))
    )

    # Convert to tensor + normalize
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # RandomErasing — simulates occlusion by neighboring leaves/stems
    erasing_p = aug.get("random_erasing_prob", 0.2)
    if erasing_p > 0:
        transform_list.append(transforms.RandomErasing(p=erasing_p))

    return transforms.Compose(transform_list)


def get_val_transforms(config: dict = None) -> transforms.Compose:
    """Validation/test transforms (deterministic)."""
    aug = config.get("augmentation", {}) if config else {}
    return transforms.Compose(
        [
            transforms.Resize(aug.get("resize", 256)),
            transforms.CenterCrop(aug.get("crop_size", 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Compute class weights
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    class_counts = defaultdict(int)
    for _, label in dataset.samples:
        class_counts[label] += 1

    n_samples = len(dataset)
    n_classes = len(dataset.classes)
    weights = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        # Inverse frequency weight: N / (C * count_i)
        w = n_samples / (n_classes * count)
        weights.append(w)

    weights_tensor = torch.FloatTensor(weights).to(device)
    print(f"Class weights: {dict(zip(dataset.classes, [f'{w:.4f}' for w in weights]))}")
    return weights_tensor


# ---------------------------------------------------------------------------
# Checkpoint utils
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, save_dir: str, filename: str = "best.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"  Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, amp_scaler=None):
    model.train()
    losses = []
    preds_all = []
    labels_all = []
    confs_all = []

    pbar = tqdm(loader, desc="  train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if amp_scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
        confs_all.extend(confs.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{sum(losses)/len(losses):.4f}")

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = sum(losses) / max(1, len(losses))
    return avg_loss, acc, preds_all, labels_all, confs_all


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    labels_all = []
    confs_all = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="  val  ", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            preds_all.extend(preds.detach().cpu().numpy().tolist())
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            confs_all.extend(confs.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{sum(losses)/len(losses):.4f}")

    acc = accuracy_score(labels_all, preds_all)
    avg_loss = sum(losses) / max(1, len(losses))
    return avg_loss, acc, preds_all, labels_all, confs_all


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------
def log_per_class_metrics(
    preds: list, labels: list, class_names: list, phase: str = "val"
) -> dict:
    """Compute and print per-class precision, recall, F1."""
    p = precision_score(labels, preds, average=None, zero_division=0)
    r = recall_score(labels, preds, average=None, zero_division=0)
    f = f1_score(labels, preds, average=None, zero_division=0)

    metrics = {}
    for i, cls in enumerate(class_names):
        metrics[cls] = {"precision": p[i], "recall": r[i], "f1": f[i]}

    # Print compact table
    print(f"  [{phase}] Per-class metrics:")
    print(f"    {'Class':<20} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"    {'-'*48}")
    for cls in class_names:
        m = metrics[cls]
        print(
            f"    {cls:<20} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}"
        )

    # Weighted averages
    p_w = precision_score(labels, preds, average="weighted", zero_division=0)
    r_w = recall_score(labels, preds, average="weighted", zero_division=0)
    f_w = f1_score(labels, preds, average="weighted", zero_division=0)
    print(f"    {'weighted avg':<20} {p_w:>8.4f} {r_w:>8.4f} {f_w:>8.4f}")

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config if provided
    config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
        print(f"Loaded config from: {args.config}")

    # Device selection
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

    # Build transforms
    transform_train = get_train_transforms(config)
    transform_val = get_val_transforms(config)

    print(f"\nTraining augmentation pipeline:")
    for i, t in enumerate(transform_train.transforms):
        print(f"  {i+1}. {t}")

    # Load datasets
    train_path = os.path.join(args.data_root, "train")
    val_path = os.path.join(args.data_root, "val")

    if not os.path.isdir(train_path):
        raise SystemExit(f"Train folder not found: {train_path}")
    if not os.path.isdir(val_path):
        raise SystemExit(f"Val folder not found: {val_path}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform_val)

    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    print(f"\nClasses ({num_classes}): {class_names}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Class distribution
    class_counts = defaultdict(int)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    print("Train class distribution:")
    for i, cls in enumerate(class_names):
        print(f"  {cls}: {class_counts[i]} ({class_counts[i]/len(train_dataset)*100:.1f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Build model
    model = build_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Loss function with optional class weights
    use_weights = args.class_weights and not args.no_class_weights
    if use_weights:
        class_weights = compute_class_weights(train_dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted CrossEntropyLoss")

    # AMP scaler (CUDA only)
    amp_scaler = (
        torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    )

    # TensorBoard
    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("TensorBoard not available.")
        else:
            tb_dir = os.path.join(args.save_dir, "runs")
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

    # Training state
    best_val_acc = 0.0
    best_epoch = -1
    start_epoch = 1
    epochs_no_improve = 0  # For early stopping
    training_history = []

    # Resume from checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Resuming from: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            try:
                model.load_state_dict(ckpt["model_state_dict"])
            except Exception:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    print("Warning: optimizer state could not be loaded")
            best_val_acc = ckpt.get("val_acc", best_val_acc)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(
                f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.4f}"
            )
        else:
            print(f"Resume checkpoint not found: {args.resume}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc, train_preds, train_labels, train_confs = (
            train_one_epoch(model, train_loader, criterion, optimizer, device, amp_scaler)
        )

        # Validate
        val_loss, val_acc, val_preds, val_labels, val_confs = validate(
            model, val_loader, criterion, device
        )

        # Current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} | LR: {current_lr:.6f}"
        )

        # Per-class metrics on validation
        val_metrics = log_per_class_metrics(
            val_preds, val_labels, class_names, phase="val"
        )

        # Compute weighted F1 for richer tracking
        val_f1_weighted = f1_score(
            val_labels, val_preds, average="weighted", zero_division=0
        )

        # Record history
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1_weighted": val_f1_weighted,
            "lr": current_lr,
            "per_class": val_metrics,
        }
        training_history.append(epoch_record)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("val/f1_weighted", val_f1_weighted, epoch)
            writer.add_scalar("lr", current_lr, epoch)
            for cls in class_names:
                writer.add_scalar(
                    f"val_f1/{cls}", val_metrics[cls]["f1"], epoch
                )

        # Scheduler step
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1_weighted": val_f1_weighted,
                "classes": class_names,
                "num_classes": num_classes,
                "model_name": args.model,
                "training_config": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "pretrained": args.pretrained,
                    "class_weighted_loss": use_weights,
                    "seed": args.seed,
                },
            }
            save_checkpoint(
                state,
                args.save_dir,
                filename=f"best_epoch_{epoch}_acc_{val_acc:.4f}.pth",
            )
        else:
            epochs_no_improve += 1

        # Save every epoch
        if args.save_every:
            state_last = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": class_names,
                "num_classes": num_classes,
                "model_name": args.model,
            }
            save_checkpoint(
                state_last, args.save_dir, filename=f"last_epoch_{epoch}.pth"
            )

        # Early stopping
        if epochs_no_improve >= args.early_stopping_patience:
            print(
                f"\n  Early stopping triggered: no improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed in {elapsed/60:.2f} minutes")
    print(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"{'='*60}")

    # Save training history
    history_path = os.path.join(args.save_dir, "training_history.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2, default=str)
    print(f"Training history saved to: {history_path}")

    if writer is not None:
        writer.flush()
        writer.close()

    # Print final summary
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python evaluate_tomato.py --data_root {args.data_root} "
          f"--checkpoint {args.save_dir}/best_epoch_{best_epoch}_acc_{best_val_acc:.4f}.pth")
    print(f"  2. Export:   python export_onnx.py "
          f"--checkpoint {args.save_dir}/best_epoch_{best_epoch}_acc_{best_val_acc:.4f}.pth")


if __name__ == "__main__":
    main()

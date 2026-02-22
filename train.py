"""
train.py

Simple training script for ImageFolder datasets. Uses a ResNet model (default `resnet18`) from torchvision
with optional transfer learning. Saves the best checkpoint by validation accuracy.

Example:
  python3 train.py --data_root "/path/to/YourCropDataset" --epochs 10 --batch_size 32

"""
import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classifier on ImageFolder-formatted data")
    parser.add_argument("--data_root", required=True, help="Path to folder containing train/val/test subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", type=str, default="resnet18", help="Model name from torchvision.models")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze feature extractor when using pretrained")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (auto if not provided)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision if available")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging to save_dir/runs")
    parser.add_argument("--save_every", action="store_true", help="Save a checkpoint every epoch (last_epoch_<n>.pth)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (will load model and optimizer states)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, pretrained: bool, num_classes: int, freeze_backbone: bool):
    # Only a few model names supported here; can be extended.
    if name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone and pretrained:
        for name_param, param in model.named_parameters():
            if not name_param.startswith("fc"):
                param.requires_grad = False

    return model


def save_checkpoint(state: dict, save_dir: str, filename: str = "best.pth"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def train_one_epoch(model, loader, criterion, optimizer, device, amp_scaler=None):
    model.train()
    losses = []
    preds_all = []
    labels_all = []

    pbar = tqdm(loader, desc="train", leave=False)
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
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        preds_all.extend(preds.tolist())
        labels_all.extend(labels_np.tolist())
        pbar.set_postfix(loss=sum(losses)/len(losses))

    acc = accuracy_score(labels_all, preds_all) if len(preds_all) > 0 else 0.0
    return sum(losses) / max(1, len(losses)), acc


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    labels_all = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            preds_all.extend(preds.tolist())
            labels_all.extend(labels_np.tolist())
            pbar.set_postfix(loss=sum(losses)/len(losses))

    acc = accuracy_score(labels_all, preds_all) if len(preds_all) > 0 else 0.0
    return sum(losses) / max(1, len(losses)), acc


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(args.data_root, "train")
    val_path = os.path.join(args.data_root, "val")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform_val)

    num_classes = len(train_dataset.classes)
    print(f"Found classes: {train_dataset.classes}")
    print(f"Num train samples: {len(train_dataset)} | Num val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, pretrained=args.pretrained, num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    amp_scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    # TensorBoard writer
    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print('TensorBoard not available in this environment (torch.utils.tensorboard not found).')
        else:
            tb_dir = os.path.join(args.save_dir, 'runs')
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

    best_val_acc = 0.0
    best_epoch = -1
    start_epoch = 1

    # Resume from checkpoint if provided
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint for resume: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                model.load_state_dict(ckpt['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    print('Warning: optimizer state could not be fully loaded')
            best_val_acc = ckpt.get('val_acc', best_val_acc)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch}. Best val acc so far: {best_val_acc}")
        else:
            print(f"Resume checkpoint not found: {args.resume}")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, amp_scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            # log learning rate (first param group)
            try:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('lr', lr, epoch)
            except Exception:
                pass

        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
            }
            save_checkpoint(state, args.save_dir, filename=f"best_epoch_{epoch}_acc_{val_acc:.4f}.pth")

        # Optionally save every epoch so we can resume exactly here later
        if args.save_every:
            state_last = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': train_dataset.classes,
            }
            save_checkpoint(state_last, args.save_dir, filename=f"last_epoch_{epoch}.pth")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.2f} minutes. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

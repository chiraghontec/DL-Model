"""
evaluate.py

Loads a checkpoint saved by `train.py` and evaluates on the `test/` split of an ImageFolder dataset.
Outputs a CSV of per-image predictions and prints a classification report and confusion matrix summary.

Example:
  python3 evaluate.py --data_root "/path/to/YourCropDataset" --checkpoint "./checkpoints/best.pth" --batch_size 32
"""
import argparse
import os
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--data_root", required=True, help="Path to folder containing train/val/test subfolders")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file saved by train.py")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture used when training")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if not provided)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="CSV path for per-image predictions")
    return parser.parse_args()


def build_model(name: str, num_classes: int, pretrained: bool = False):
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
    return model


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    test_path = os.path.join(args.data_root, "test")
    if not os.path.isdir(test_path):
        raise SystemExit(f"Test folder not found: {test_path}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Try to get classes from checkpoint, fall back to dataset
    classes = ckpt.get('classes') if isinstance(ckpt, dict) else None
    if classes is None:
        classes = test_dataset.classes

    num_classes = len(classes)

    model = build_model(args.model, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    filepaths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy().tolist()
            labels_np = labels.cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    # Map numeric labels to class names
    pred_names = [classes[p] for p in all_preds]
    true_names = [classes[t] for t in all_labels]

    # Print report
    print("Classification Report:\n")
    print(classification_report(true_names, pred_names, digits=4))

    print("Confusion Matrix:\n")
    cm = confusion_matrix(true_names, pred_names, labels=classes)
    print(cm)

    # Save per-image CSV (use dataset.samples order)
    csv_path = Path(args.output_csv)
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'true_label', 'pred_label'])

        # ImageFolder stores samples in the same order DataLoader would visit when shuffle=False
        for (filepath, _), true_label_idx, pred_idx in zip(test_dataset.samples, all_labels, all_preds):
            writer.writerow([filepath, classes[true_label_idx], classes[pred_idx]])

    print(f"Saved predictions CSV to: {csv_path}")


if __name__ == "__main__":
    main()

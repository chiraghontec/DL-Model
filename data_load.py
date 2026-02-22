"""
data_load.py

Loads images organized in ImageFolder format and prints dataset sizes and class names.
Usage:
  python3 data_load.py --data_root "/path/to/YourCropDataset" --batch_size 32
"""
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Create PyTorch DataLoaders using ImageFolder")
    parser.add_argument("--data_root", required=True, help="Path to folder containing train/val/test subfolders")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    DATA_ROOT = args.data_root

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = f"{DATA_ROOT}/train"
    val_path = f"{DATA_ROOT}/val"
    test_path = f"{DATA_ROOT}/test"

    if not (torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print(f"Using device: {device}")

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = None
    try:
        test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    except Exception:
        # test set is optional for this loader script
        pass

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(val_dataset))
    if test_dataset is not None:
        print('Number of test samples:', len(test_dataset))
    print('Class Names:', train_dataset.classes)

    # Show a single batch shape example
    batch = next(iter(train_loader))
    images, labels = batch
    print('Example batch - images shape:', images.shape)
    print('Example batch - labels shape:', labels.shape)


if __name__ == "__main__":
    main()

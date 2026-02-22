#!/usr/bin/env python3
"""
download_dataset.py

Downloads the PlantVillage tomato subset from Kaggle and creates
a stratified train/val/test split for 3-class tomato blight detection.

Classes extracted:
  - Tomato___Early_blight  → early_blight
  - Tomato___Late_blight   → late_blight
  - Tomato___healthy       → healthy

Requirements:
  pip install kagglehub

Setup:
  1. Create a Kaggle account at https://www.kaggle.com
  2. Go to https://www.kaggle.com/settings → "Create New Token"
  3. Save downloaded kaggle.json to ~/.kaggle/kaggle.json
  4. chmod 600 ~/.kaggle/kaggle.json

Usage:
  python download_dataset.py [--output_dir ./data] [--ratio 0.7 0.1 0.2] [--seed 42] [--verify]
"""

import argparse
import hashlib
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# Mapping from PlantVillage folder names → our class names
PLANTVILLAGE_CLASS_MAP = {
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___healthy": "healthy",
}

# Also handle alternate naming conventions in different PlantVillage mirrors
ALTERNATE_CLASS_MAP = {
    "Tomato_Early_blight": "early_blight",
    "Tomato_Late_blight": "late_blight",
    "Tomato_healthy": "healthy",
    "Tomato__Early_blight": "early_blight",
    "Tomato__Late_blight": "late_blight",
    "Tomato__healthy": "healthy",
    "Early_blight": "early_blight",
    "Late_blight": "late_blight",
    "healthy": "healthy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download PlantVillage tomato subset and create train/val/test split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for the split dataset (default: ./data)",
    )
    parser.add_argument(
        "--ratio",
        nargs=3,
        type=float,
        default=[0.7, 0.1, 0.2],
        help="Train/Val/Test split ratios (default: 0.7 0.1 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run MD5 duplicate check and image integrity verification",
    )
    parser.add_argument(
        "--kaggle_dataset",
        type=str,
        default="abdallahalidev/plantvillage-dataset",
        help="Kaggle dataset identifier",
    )
    return parser.parse_args()


def download_from_kaggle(dataset_id: str) -> str:
    """Download dataset using kagglehub and return the local path."""
    try:
        import kagglehub

        print(f"Downloading dataset '{dataset_id}' via kagglehub...")
        path = kagglehub.dataset_download(dataset_id)
        print(f"Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("ERROR: kagglehub is not installed.")
        print("Install it with: pip install kagglehub")
        print(
            "\nAlternatively, manually download from:"
        )
        print(f"  https://www.kaggle.com/datasets/{dataset_id}")
        print(
            "  Extract it and pass the path via environment variable PLANTVILLAGE_PATH"
        )
        raise
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print(
            "\nMake sure you have a valid Kaggle API key at ~/.kaggle/kaggle.json"
        )
        print("Get one from: https://www.kaggle.com/settings → 'Create New Token'")
        raise


def find_tomato_classes(dataset_root: str) -> dict:
    """
    Search the downloaded dataset for tomato-related class folders.
    Returns dict mapping: our_class_name → list of source image paths.
    """
    dataset_root = Path(dataset_root)
    class_images = defaultdict(list)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG"}

    # Build combined class map
    full_map = {**PLANTVILLAGE_CLASS_MAP, **ALTERNATE_CLASS_MAP}

    print(f"Scanning dataset at: {dataset_root}")

    # Recursively find all directories
    all_dirs = [d for d in dataset_root.rglob("*") if d.is_dir()]

    matched_dirs = []
    for dir_path in all_dirs:
        dir_name = dir_path.name
        if dir_name in full_map:
            our_class = full_map[dir_name]
            # Collect all image files in this directory
            for f in dir_path.iterdir():
                if f.is_file() and f.suffix in image_extensions:
                    class_images[our_class].append(str(f))
            matched_dirs.append((dir_name, our_class, len(class_images[our_class])))

    if not class_images:
        # Try a broader search — look for directories containing "blight" or "healthy"
        print("Standard folder names not found. Trying fuzzy match...")
        for dir_path in all_dirs:
            dir_name_lower = dir_path.name.lower()
            our_class = None
            if "early" in dir_name_lower and "blight" in dir_name_lower:
                our_class = "early_blight"
            elif "late" in dir_name_lower and "blight" in dir_name_lower:
                our_class = "late_blight"
            elif "healthy" in dir_name_lower and "tomato" in str(dir_path).lower():
                our_class = "healthy"

            if our_class:
                for f in dir_path.iterdir():
                    if f.is_file() and f.suffix in image_extensions:
                        class_images[our_class].append(str(f))
                matched_dirs.append(
                    (dir_path.name, our_class, len(class_images[our_class]))
                )

    # Deduplicate file paths per class
    for cls in class_images:
        class_images[cls] = sorted(set(class_images[cls]))

    print(f"\nFound {len(matched_dirs)} matching directories:")
    for orig_name, our_name, count in matched_dirs:
        print(f"  {orig_name} → {our_name}")
    print(f"\nClass image counts:")
    for cls in sorted(class_images.keys()):
        print(f"  {cls}: {len(class_images[cls])} images")

    return dict(class_images)


def verify_images(class_images: dict) -> dict:
    """Verify image integrity and remove duplicates via MD5 hashing."""
    from PIL import Image

    print("\nVerifying images and checking for duplicates...")
    clean_images = defaultdict(list)
    seen_hashes = set()
    corrupted = 0
    duplicates = 0

    for cls, paths in class_images.items():
        for path in tqdm(paths, desc=f"Verifying {cls}"):
            try:
                # Verify image can be opened and converted
                with Image.open(path) as img:
                    img.verify()
                # Re-open for full load verification
                with Image.open(path) as img:
                    img = img.convert("RGB")

                # Check for duplicates via MD5
                with open(path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in seen_hashes:
                    duplicates += 1
                    continue

                seen_hashes.add(file_hash)
                clean_images[cls].append(path)

            except Exception as e:
                corrupted += 1
                continue

    print(f"\nVerification complete:")
    print(f"  Corrupted images removed: {corrupted}")
    print(f"  Duplicate images removed: {duplicates}")
    for cls in sorted(clean_images.keys()):
        print(f"  {cls}: {len(clean_images[cls])} unique valid images")

    return dict(clean_images)


def create_split(
    class_images: dict,
    output_dir: str,
    ratios: list,
    seed: int,
):
    """Create stratified train/val/test split."""
    output_dir = Path(output_dir)
    random.seed(seed)

    splits = ["train", "val", "test"]
    split_counts = defaultdict(lambda: defaultdict(int))

    # Create output directories
    for split in splits:
        for cls in class_images:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"\nCreating stratified split (ratio: {ratios}) with seed={seed}...")

    for cls, paths in class_images.items():
        # Shuffle deterministically
        shuffled = paths.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # Rest goes to test
        splits_data = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }

        for split, split_paths in splits_data.items():
            for src_path in tqdm(
                split_paths, desc=f"Copying {cls}/{split}", leave=False
            ):
                src = Path(src_path)
                dst = output_dir / split / cls / src.name

                # Handle filename collisions
                if dst.exists():
                    stem = dst.stem
                    suffix = dst.suffix
                    counter = 1
                    while dst.exists():
                        dst = output_dir / split / cls / f"{stem}_{counter}{suffix}"
                        counter += 1

                shutil.copy2(str(src), str(dst))
                split_counts[split][cls] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"{'Class':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    total_train, total_val, total_test = 0, 0, 0
    for cls in sorted(class_images.keys()):
        t = split_counts["train"][cls]
        v = split_counts["val"][cls]
        te = split_counts["test"][cls]
        total_train += t
        total_val += v
        total_test += te
        print(f"{cls:<20} {t:>8} {v:>8} {te:>8} {t+v+te:>8}")
    print("-" * 60)
    total = total_train + total_val + total_test
    print(
        f"{'TOTAL':<20} {total_train:>8} {total_val:>8} {total_test:>8} {total:>8}"
    )
    print("=" * 60)
    print(f"\nDataset saved to: {output_dir.resolve()}")

    return dict(split_counts)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Check if dataset already exists
    if (output_dir / "train").exists() and (output_dir / "val").exists():
        existing_classes = [
            d.name for d in (output_dir / "train").iterdir() if d.is_dir()
        ]
        if len(existing_classes) >= 3:
            print(f"Dataset already exists at {output_dir} with classes: {existing_classes}")
            print("Delete the directory or choose a different --output_dir to re-download.")
            return

    # Step 1: Download from Kaggle (or use manual path)
    manual_path = os.environ.get("PLANTVILLAGE_PATH")
    if manual_path and os.path.isdir(manual_path):
        print(f"Using manual dataset path: {manual_path}")
        dataset_path = manual_path
    else:
        dataset_path = download_from_kaggle(args.kaggle_dataset)

    # Step 2: Find tomato class folders
    class_images = find_tomato_classes(dataset_path)

    if len(class_images) < 3:
        print(
            f"\nERROR: Expected 3 classes but found {len(class_images)}: {list(class_images.keys())}"
        )
        print("The dataset structure may differ. Check the downloaded folder structure:")
        print(f"  {dataset_path}")
        return

    if len(class_images) > 3:
        print(
            f"\nWARNING: Found {len(class_images)} classes, expected 3. Keeping only target classes."
        )
        target_classes = {"early_blight", "late_blight", "healthy"}
        class_images = {k: v for k, v in class_images.items() if k in target_classes}

    # Step 3: Verify images (optional but recommended)
    if args.verify:
        class_images = verify_images(class_images)

    # Step 4: Create stratified split
    create_split(class_images, str(output_dir), args.ratio, args.seed)

    print("\nDone! Next steps:")
    print(
        "  1. Visually inspect ~20 random images per class to confirm label quality"
    )
    print("  2. Run training: python train_tomato.py --data_root ./data --epochs 30")


if __name__ == "__main__":
    main()

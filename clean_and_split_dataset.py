#!/usr/bin/env python3
"""
Clean dataset by removing duplicates, normalizing class names, and creating
a proper stratified split without data leakage.

Steps:
1. Scan original dataset and compute hashes for all images
2. Remove duplicates (keep first occurrence only)
3. Normalize class names (remove numeric suffixes like "3102", "1714", etc.)
4. Create stratified train/val/test split (70/10/20)
5. Copy cleaned images to new dataset structure
"""
import os
import re
import hashlib
import shutil
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def get_file_hash(filepath):
    """Compute MD5 hash of a file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None


def normalize_class_name(class_name):
    """Remove numeric suffixes from class names.
    
    Examples:
        anthracnose3102 -> anthracnose
        bacterial blight3241 -> bacterial blight
        healthy5877 -> healthy
    """
    # Remove trailing numbers (e.g., "3102", "1714")
    normalized = re.sub(r'\d+$', '', class_name).strip()
    return normalized if normalized else class_name


def scan_dataset(dataset_root):
    """Scan dataset and return dict of: normalized_class -> [(filepath, hash), ...]"""
    dataset_root = Path(dataset_root)
    class_files = defaultdict(list)
    hash_to_files = defaultdict(list)
    
    print("Scanning original dataset...")
    
    # Find all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    all_files = []
    
    for ext in image_extensions:
        all_files.extend(dataset_root.rglob(f'*{ext}'))
        all_files.extend(dataset_root.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(all_files)} image files. Computing hashes...")
    
    for filepath in tqdm(all_files, desc="Hashing"):
        # Extract class name from parent directory structure
        # Assume structure: .../ClassName/image.jpg or .../train/ClassName/image.jpg
        parts = filepath.parts
        
        # Find the class folder (parent of the image file)
        class_name = filepath.parent.name
        
        # Normalize the class name
        normalized_class = normalize_class_name(class_name)
        
        # Compute hash
        file_hash = get_file_hash(str(filepath))
        if file_hash is None:
            continue
        
        # Track this file
        class_files[normalized_class].append((str(filepath), file_hash))
        hash_to_files[file_hash].append((str(filepath), normalized_class))
    
    return class_files, hash_to_files


def remove_duplicates(class_files, hash_to_files):
    """Remove duplicate files, keeping only the first occurrence of each hash."""
    print("\nRemoving duplicates...")
    
    unique_files = defaultdict(list)
    seen_hashes = set()
    duplicates_removed = 0
    
    for class_name, file_list in class_files.items():
        for filepath, file_hash in file_list:
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files[class_name].append(filepath)
            else:
                duplicates_removed += 1
    
    print(f"Removed {duplicates_removed} duplicate files")
    print(f"Unique images remaining: {sum(len(v) for v in unique_files.values())}")
    
    return unique_files


def stratified_split(class_files, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=42):
    """Create stratified train/val/test split."""
    random.seed(seed)
    
    splits = {'train': defaultdict(list), 'val': defaultdict(list), 'test': defaultdict(list)}
    
    print("\nCreating stratified split...")
    for class_name, file_list in class_files.items():
        # Shuffle files for this class
        shuffled = file_list.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits['train'][class_name] = shuffled[:train_end]
        splits['val'][class_name] = shuffled[train_end:val_end]
        splits['test'][class_name] = shuffled[val_end:]
        
        print(f"  {class_name:30s}: train={len(splits['train'][class_name]):5d}, "
              f"val={len(splits['val'][class_name]):5d}, test={len(splits['test'][class_name]):5d}")
    
    return splits


def copy_split_to_disk(splits, output_root, dry_run=False):
    """Copy split files to output directory structure."""
    output_root = Path(output_root)
    
    if dry_run:
        print("\n[DRY RUN] Would create the following structure:")
    else:
        print(f"\nCopying files to {output_root}...")
    
    total_copied = 0
    
    for split_name, class_dict in splits.items():
        for class_name, file_list in class_dict.items():
            split_dir = output_root / split_name / class_name
            
            if not dry_run:
                split_dir.mkdir(parents=True, exist_ok=True)
            
            for src_path in tqdm(file_list, desc=f"{split_name}/{class_name}", leave=False):
                src = Path(src_path)
                dst = split_dir / src.name
                
                if not dry_run:
                    try:
                        shutil.copy2(src, dst)
                        total_copied += 1
                    except Exception as e:
                        print(f"Error copying {src} -> {dst}: {e}")
    
    if not dry_run:
        print(f"Copied {total_copied} files successfully")
    
    # Print summary
    print("\n" + "="*80)
    print("SPLIT SUMMARY")
    print("="*80)
    for split_name in ['train', 'val', 'test']:
        total = sum(len(files) for files in splits[split_name].values())
        print(f"{split_name.capitalize():5s}: {total:6,d} images across {len(splits[split_name])} classes")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Clean dataset and create proper split")
    parser.add_argument('--input', required=True, help='Path to original dataset root')
    parser.add_argument('--output', required=True, help='Path to output cleaned dataset')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Train split ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.10, help='Validation split ratio (default: 0.10)')
    parser.add_argument('--test_ratio', type=float, default=0.20, help='Test split ratio (default: 0.20)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry_run', action='store_true', help='Dry run without copying files')
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("ERROR: Split ratios must sum to 1.0")
        return
    
    print("="*80)
    print("DATASET CLEANING AND SPLITTING")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Split:  train={args.train_ratio:.0%} val={args.val_ratio:.0%} test={args.test_ratio:.0%}")
    print(f"Seed:   {args.seed}")
    print("="*80)
    
    # Step 1: Scan dataset
    class_files, hash_to_files = scan_dataset(args.input)
    
    print(f"\nOriginal dataset:")
    print(f"  Classes found: {len(class_files)}")
    print(f"  Total images: {sum(len(v) for v in class_files.values())}")
    
    # Step 2: Remove duplicates
    unique_files = remove_duplicates(class_files, hash_to_files)
    
    print(f"\nAfter removing duplicates:")
    print(f"  Classes: {len(unique_files)}")
    print(f"  Total unique images: {sum(len(v) for v in unique_files.values())}")
    
    # Step 3: Create stratified split
    splits = stratified_split(unique_files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    # Step 4: Copy to disk
    copy_split_to_disk(splits, args.output, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Re-run without --dry_run to actually copy files.")
    else:
        print(f"\n✅ Dataset cleaning and splitting complete!")
        print(f"   Output: {args.output}")


if __name__ == '__main__':
    main()

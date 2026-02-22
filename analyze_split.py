#!/usr/bin/env python3
"""
Analyze class distribution and investigate splitting issues.

Checks:
1. Class distribution in each split (train/val/test)
2. Duplicate images across splits (same file in multiple splits)
3. Class representation percentages
4. Imbalance ratios
"""
import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json


def get_file_hash(filepath):
    """Compute MD5 hash of a file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def analyze_split(split_root):
    """Analyze a single split (train/val/test)."""
    split_root = Path(split_root)
    results = {
        'split': split_root.name,
        'classes': {},
        'total_images': 0,
        'file_hashes': {}  # hash -> list of file paths
    }
    
    if not split_root.exists():
        print(f"Split path not found: {split_root}")
        return results
    
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        image_count = 0
        
        for img_file in class_dir.glob('*'):
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                image_count += 1
                # Compute hash for duplicate detection
                file_hash = get_file_hash(str(img_file))
                if file_hash:
                    if file_hash not in results['file_hashes']:
                        results['file_hashes'][file_hash] = []
                    results['file_hashes'][file_hash].append(str(img_file))
        
        if image_count > 0:
            results['classes'][class_name] = image_count
            results['total_images'] += image_count
    
    return results


def main():
    data_root = Path("/Users/vinayakprasad/Documents/Major Project/YourCropDataset")
    splits = ['train', 'val', 'test']
    
    print("=" * 80)
    print("DATASET SPLIT AND CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    split_analyses = {}
    all_hashes = defaultdict(list)  # hash -> [(split, class, file), ...]
    
    # Analyze each split
    for split_name in splits:
        split_path = data_root / split_name
        analysis = analyze_split(split_path)
        split_analyses[split_name] = analysis
        
        print(f"\n{split_name.upper()} SET")
        print("-" * 80)
        print(f"Total images: {analysis['total_images']}")
        print(f"Number of classes: {len(analysis['classes'])}")
        
        # Class distribution
        sorted_classes = sorted(analysis['classes'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            pct = 100.0 * count / analysis['total_images'] if analysis['total_images'] > 0 else 0
            print(f"  {class_name:30s}: {count:6d} ({pct:5.2f}%)")
        
        # Track hashes for duplicate detection
        for file_hash, file_list in analysis['file_hashes'].items():
            for file_path in file_list:
                class_name = Path(file_path).parent.name
                all_hashes[file_hash].append((split_name, class_name, file_path))
    
    # Check for duplicates across splits
    print("\n" + "=" * 80)
    print("DUPLICATE DETECTION (same file in multiple splits)")
    print("=" * 80)
    duplicates_found = 0
    for file_hash, occurrences in all_hashes.items():
        if len(occurrences) > 1:
            duplicates_found += 1
            splits_involved = set(occ[0] for occ in occurrences)
            print(f"\nDuplicate #{duplicates_found} (hash: {file_hash[:8]}...)")
            print(f"  Appears in {len(occurrences)} locations across {len(splits_involved)} split(s):")
            for split_name, class_name, file_path in occurrences:
                print(f"    [{split_name:5s}] {class_name:30s} - {Path(file_path).name}")
    
    if duplicates_found == 0:
        print("No duplicates found across splits (GOOD).")
    else:
        print(f"\nWARNING: Found {duplicates_found} duplicate files across splits!")
    
    # Check for class distribution balance
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Get all classes
    all_classes = set()
    for analysis in split_analyses.values():
        all_classes.update(analysis['classes'].keys())
    
    all_classes = sorted(all_classes)
    print(f"Total unique classes: {len(all_classes)}\n")
    
    print(f"{'Class Name':30s} | {'Train':>6s} | {'Val':>6s} | {'Test':>6s} | {'Total':>6s} | {'Balance':<30s}")
    print("-" * 110)
    
    for class_name in all_classes:
        train_count = split_analyses['train']['classes'].get(class_name, 0)
        val_count = split_analyses['val']['classes'].get(class_name, 0)
        test_count = split_analyses['test']['classes'].get(class_name, 0)
        total = train_count + val_count + test_count
        
        # Calculate split percentages
        train_pct = 100.0 * train_count / total if total > 0 else 0
        val_pct = 100.0 * val_count / total if total > 0 else 0
        test_pct = 100.0 * test_count / total if total > 0 else 0
        
        balance_str = f"T:{train_pct:5.1f}% V:{val_pct:5.1f}% Te:{test_pct:5.1f}%"
        
        print(f"{class_name:30s} | {train_count:6d} | {val_count:6d} | {test_count:6d} | {total:6d} | {balance_str}")
    
    # Overall split statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    train_total = split_analyses['train']['total_images']
    val_total = split_analyses['val']['total_images']
    test_total = split_analyses['test']['total_images']
    grand_total = train_total + val_total + test_total
    
    print(f"Train: {train_total:,d} ({100.0*train_total/grand_total:.1f}%)")
    print(f"Val:   {val_total:,d} ({100.0*val_total/grand_total:.1f}%)")
    print(f"Test:  {test_total:,d} ({100.0*test_total/grand_total:.1f}%)")
    print(f"Total: {grand_total:,d}")
    
    # Check for missing classes
    print("\n" + "=" * 80)
    print("MISSING CLASSES IN SPLITS")
    print("=" * 80)
    issues = []
    for class_name in all_classes:
        train_has = class_name in split_analyses['train']['classes']
        val_has = class_name in split_analyses['val']['classes']
        test_has = class_name in split_analyses['test']['classes']
        
        if not (train_has and val_has and test_has):
            issues.append((class_name, train_has, val_has, test_has))
    
    if issues:
        print(f"WARNING: {len(issues)} classes not present in all splits!\n")
        print(f"{'Class':30s} | Train | Val | Test")
        print("-" * 50)
        for class_name, train_has, val_has, test_has in issues:
            print(f"{class_name:30s} | {'✓' if train_has else '✗':5s} | {'✓' if val_has else '✗':3s} | {'✓' if test_has else '✗':4s}")
    else:
        print("All classes present in all splits (GOOD).")
    
    # Imbalance ratio
    print("\n" + "=" * 80)
    print("CLASS IMBALANCE RATIO")
    print("=" * 80)
    train_counts = list(split_analyses['train']['classes'].values())
    if train_counts:
        max_count = max(train_counts)
        min_count = min(train_counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else 0
        print(f"Train set imbalance ratio (max/min): {imbalance_ratio:.2f}x")
        print(f"  Max class size: {max_count:,d}")
        print(f"  Min class size: {min_count:,d}")
        if imbalance_ratio > 10:
            print(f"  WARNING: Severe class imbalance detected (> 10x)!")
        elif imbalance_ratio > 5:
            print(f"  WARNING: Moderate class imbalance detected (> 5x)!")


if __name__ == '__main__':
    main()

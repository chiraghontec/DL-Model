"""
split_data.py

Usage:
  python3 split_data.py --input "/path/to/OriginalData" --output "/path/to/YourCropDataset" --ratio 0.8 0.1 0.1

This script expects the input folder to contain one subfolder per class, e.g.
  /OriginalData/Cashew_Healthy/*
  /OriginalData/Tomato_Blight/*

It will create `train/`, `val/`, `test/` inside the output folder.
"""
import argparse
import os
import splitfolders


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test using splitfolders")
    parser.add_argument("--input", required=True, help="Path to input folder containing class subfolders")
    parser.add_argument("--output", required=True, help="Path to output folder to create train/val/test")
    parser.add_argument(
        "--ratio",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Three floats for train/val/test ratios (must sum to 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        raise SystemExit(f"Input folder does not exist: {args.input}")

    # Ensure sum to 1 (within tolerance)
    if abs(sum(args.ratio) - 1.0) > 1e-6:
        raise SystemExit("The provided ratios must sum to 1.0")

    print(f"Splitting data from: {args.input}")
    print(f"Output will be: {args.output}")
    print(f"Using ratios: {args.ratio}")

    splitfolders.ratio(args.input, output=args.output, seed=args.seed, ratio=tuple(args.ratio))

    print("Done. Output structure:")
    print(os.path.join(args.output, "train"))
    print(os.path.join(args.output, "val"))
    print(os.path.join(args.output, "test"))


if __name__ == "__main__":
    main()

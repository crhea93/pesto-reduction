import os
import argparse
from pesto_reduction.mosaic import mosaic_images


def parse_args():
    parser = argparse.ArgumentParser(description="Create mosaic from stacked images.")
    parser.add_argument("--filter", required=True, help="Filter name (e.g. i, Ha)")
    parser.add_argument("--name", required=True, help="Target name (e.g. Arp94)")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory with stacked images and where mosaic will be saved",
    )
    parser.add_argument(
        "--n_pos", type=int, default=6, help="Number of positions (default: 6)"
    )
    parser.add_argument(
        "--combine",
        type=str,
        default="median",
        choices=["median", "mean"],
        help="Combination method (default: median)",
    )
    return parser.parse_args()


def main():
    """
    Example Usage:
    > python scripts/mosaic.py     --filter i     --name Arp94     --output_dir /home/carterrhea/Documents/OMM-Cleaned/Arp94     --n_pos 6     --combine median
    """
    args = parse_args()

    image_paths = []
    for i in range(args.n_pos):
        pos = f"{args.name}_pos{i + 1}"
        path = os.path.join(args.output_dir, f"{pos}_{args.filter}.fits")
        if os.path.exists(path):
            image_paths.append(path)
        else:
            print(f"[WARNING] Missing: {path}")

    if not image_paths:
        print("[ERROR] No valid image paths found. Aborting.")
        return

    output_path = os.path.join(args.output_dir, f"{args.name}_{args.filter}.fits")
    print(f"[INFO] Building mosaic: {output_path}")
    mosaic_images(image_paths, output_path, combine=args.combine)


if __name__ == "__main__":
    main()

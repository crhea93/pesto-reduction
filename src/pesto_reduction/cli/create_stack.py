import argparse
import os

from pesto_reduction.create_stack import create_stack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create median-stacked image per position."
    )
    parser.add_argument("--filter", required=True, help="Filter name (e.g. i, Ha)")
    parser.add_argument("--name", required=True, help="Target name (e.g. Arp94)")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where output stacks and flat are stored",
    )
    parser.add_argument(
        "--data_root", required=True, help="Root directory of the raw light frames"
    )
    parser.add_argument(
        "--n_pos", type=int, default=6, help="Number of positions (default: 6)"
    )
    parser.add_argument(
        "--bias-level",
        type=float,
        default=300.0,
        help="Bias level to subtract from light frames (default: 300.0). Must match the bias level used when creating the master flat.",
    )
    return parser.parse_args()


def main():
    """
    example:
    ```bash
    python scripts/create_stack.py     --filter Ha     --name Arp94     --output_dir /home/carterrhea/Documents/OMM-Cleaned/Arp94     --data_root /home/carterrhea/Documents/OMM-Data/210413     --n_pos 6
    ```
    """
    args = parse_args()

    flat_path = os.path.join(args.output_dir, "master_flat.fits")

    if not os.path.exists(flat_path):
        print(f"[ERROR] Master flat not found at {flat_path}")
        return

    for i in range(args.n_pos):
        i += 1
        pos = f"{args.name}_pos{i}"
        data_dir = os.path.join(args.data_root, "Target", pos, "redux")
        if not os.path.isdir(data_dir):
            print(f"[WARNING] Skipping {pos}: directory not found.")
            continue

        print(f"[INFO] Stacking {pos} with filter {args.filter}")
        create_stack(
            filter_=args.filter,
            pos=pos,
            data_dir=data_dir,
            output_dir=args.output_dir,
            flat_path=flat_path,
            target_name=args.name,
            bias_level=args.bias_level,
        )


if __name__ == "__main__":
    main()

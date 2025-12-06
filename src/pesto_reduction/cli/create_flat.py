import argparse

from pesto_reduction.flats import create_master_flat


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a master flat from a directory of FITS files."
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing flat-field FITS files"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the output master flat FITS file",
    )
    parser.add_argument(
        "--bias_level",
        type=float,
        default=300,
        help="Bias level to subtract from each flat (default: 300)",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Disable histogram plotting"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=500,
        help="Half-size of center crop for normalization (default: 500)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    create_master_flat(
        data_dir=args.data_dir,
        output_path=args.output_path,
        bias_level=args.bias_level,
        center_crop_halfsize=args.crop_size,
        plot_hist=not args.no_plot,
    )


if __name__ == "__main__":
    main()

import click

from pesto_reduction.flats import optimize_bias_level


@click.command()
@click.option(
    "--light-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a representative light frame FITS file",
)
@click.option(
    "--flat-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing flat field FITS files",
)
@click.option(
    "--bias-min",
    type=float,
    default=100.0,
    help="Minimum bias value to search (default: 100)",
)
@click.option(
    "--bias-max",
    type=float,
    default=600.0,
    help="Maximum bias value to search (default: 600)",
)
@click.option(
    "--center-crop-halfsize",
    type=int,
    default=500,
    help="Half-size of central region for flat normalization (default: 500)",
)
def main(light_path, flat_dir, bias_min, bias_max, center_crop_halfsize):
    """
    Optimize bias level by minimizing correlation between flat-corrected
    image and the flat field pattern.

    This tool helps find the optimal bias level when you don't have
    proper bias/dark frames. The optimal bias should result in a
    flat-corrected image with minimal flat field artifacts.
    """
    optimal_bias = optimize_bias_level(
        light_path=light_path,
        flat_path=flat_dir,
        bias_range=(bias_min, bias_max),
        center_crop_halfsize=center_crop_halfsize,
    )

    click.echo(f"\nOptimal bias level: {optimal_bias:.2f}")
    click.echo(f"\nUse this value in your create_flat and create_stack commands.")


if __name__ == "__main__":
    main()

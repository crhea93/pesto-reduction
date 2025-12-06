import glob
import os
import warnings
from datetime import datetime

import numpy as np
from dfreproject import calculate_reprojection

from pesto_reduction.logger import make_time_named_logger
from pesto_reduction.utils import (
    get_coords,
    remove_cosmic,
    solve_field,
    subtract_background,
)

warnings.filterwarnings(
    "ignore",
    message="cdelt will be ignored since cd is present",
    category=RuntimeWarning,
    module="dfreproject.reproject",
)
logger = make_time_named_logger()


from astropy.io import fits
from astropy.wcs import WCS
from joblib import Parallel, delayed


def build_global_wcs(ra, dec, pixel_scale=1.0, field_size_arcmin=30.0):
    """
    Build a WCS covering all input images.

    Parameters
    ----------
    ra : float
        Right ascension in degrees
    dec : float
        Declination in degrees
    pixel_scale : float
        Desired pixel scale in arcsec/pixel.
    field_size_arcmin : float
        Field of view size in arcminutes (default: 10.0)

    Returns
    -------
    output_wcs : astropy.wcs.WCS
    WCS covering all images.
    output_shape : tuple of int
    (ny, nx) shape of the output grid.
    """

    scale_deg = pixel_scale / 3600.0  # arcsec to degrees
    field_size_deg = field_size_arcmin / 60.0  # arcmin to degrees

    # Calculate number of pixels needed
    nx = int(np.ceil(field_size_deg / scale_deg))
    ny = int(np.ceil(field_size_deg / scale_deg))

    output_wcs = WCS(naxis=2)
    output_wcs.wcs.crpix = [nx / 2, ny / 2]
    output_wcs.wcs.crval = [ra, dec]
    output_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # Use CD matrix instead of CDELT to avoid warning
    output_wcs.wcs.cd = np.array([[-scale_deg, 0], [0, scale_deg]])

    return output_wcs, (ny, nx)


def process_light(light, master_flat, output_wcs, output_shape, ra, dec, bias_level):
    try:
        # light_header = fits.getheader(light)
        light_vals = fits.getdata(light).astype("float32")[:-1, :]
        light_vals -= bias_level  # Subtract bias
        light_basename = os.path.basename(light)

        # Apply flat field correction
        light_vals = light_vals / master_flat.astype("float32")

        # light_vals = remove_cosmic(light_vals)
        # light_vals = subtract_background(light_vals)
        # # wcs_solved_path = solve_field(
        # #     light, scale_low=0.2, scale_high=1.0, ra=ra, dec=dec, radius=10
        # # )
        # light_vals = calculate_reprojection(
        #     source_hdus=(light_vals, fits.getheader(light)),
        #     target_wcs=output_wcs,
        #     shape_out=output_shape,
        #     order="bilinear",
        # )
        #
        fits.writeto(
            f"/home/carterrhea/Downloads/{light_basename}_processed.fits",
            light_vals,
            overwrite=True,
        )
        return light_vals

    except FileNotFoundError:
        logger.warning(f"Skipping {light} due to plate-solve error.")
        return None
    except Exception as e:
        logger.warning(f"Error processing {light}: {e}")
        return None


def create_stack(
    filter_, pos, data_dir, output_dir, flat_path, target_name, bias_level=300.0
):
    """
    Create a median-stacked, background-subtracted, WCS-aligned image for a given position and filter.

    This function loads all light frames from a given directory, selects those matching the desired filter,
    performs preprocessing including bias subtraction, flat-field correction, cosmic ray removal, and
    background subtraction, solves for WCS alignment, reprojects images to a common grid, and writes
    out the final stacked FITS file.

    Parameters
    ----------
    filter_ : str
    The filter name to select light frames (e.g., 'Ha', 'OIII').
    pos : str
    A name or identifier for the image position (e.g., 'Arp94_pos3').
    data_dir : str
    Directory containing raw light frames in FITS format.
    output_dir : str
    Directory where the final stacked FITS file will be saved.
    flat_path : str
    Path to the master flat FITS file used for flat-field correction.
    target_name : str
    Name of the astronomical target, used to look up sky coordinates for astrometric solving.
    bias_level : float, optional
    Constant bias value to subtract from each light frame, by default 300.0.
    Must match the bias level used when creating the master flat.

    Returns
    -------
    None
    The result is written to a FITS file at `output_dir/{pos}.fits`.
    """
    logger.info(f"Reducing data for {pos} ({filter_})")

    os.makedirs(output_dir, exist_ok=True)

    lights = glob.glob(f"{data_dir}/*.fits")
    master_flat = fits.getdata(flat_path).astype("float32")
    valid_light_paths = [
        path
        for path in lights
        if filter_.lower() in fits.getheader(path).get("FILTER", "").lower()
    ]
    if len(valid_light_paths) > 0:
        ra, dec = get_coords(target_name=target_name)
        output_wcs, output_shape = build_global_wcs(
            ra, dec, pixel_scale=0.495
        )  # arcsec/pixel
        light_data = Parallel(n_jobs=1, backend="multiprocessing")(
            delayed(process_light)(
                light, master_flat, output_wcs, output_shape, ra, dec, bias_level
            )
            for light in valid_light_paths
        )

        # Filter out any None results
        light_data = [img for img in light_data if img is not None]

        logger.info(f"Found {len(light_data)} valid frames for filter {filter_}")

        if len(light_data) > 0:
            stacked = np.stack(light_data, axis=0)
            median = np.nanmedian(stacked, axis=0)

            # Create new header with correct WCS from the target
            final_wcs = output_wcs
            final_header = fits.Header()
            final_header.update(final_wcs.to_header())

            # Add stacking metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            final_header["COMMENT"] = "MEDIAN STACKED COADD"
            final_header["COMMENT"] = f"NUMBER OF FRAMES IN STACK: {len(light_data)}"
            final_header["COMMENT"] = f"DATE CREATED {timestamp}"
            final_header["FILTER"] = filter_

            output_path = os.path.join(output_dir, f"{pos}_{filter_}.fits")
            fits.writeto(output_path, median, header=final_header, overwrite=True)
            logger.info(f"Saved final stacked image to {output_path}")
    else:
        logger.warning("No valid light frames found. Skipping stack creation.")

    return None

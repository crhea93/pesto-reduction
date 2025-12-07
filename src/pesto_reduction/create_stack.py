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


def process_light(
    light,
    master_flat,
    output_wcs,
    output_shape,
    ra,
    dec,
    bias_level,
    optimize_bias=True,
    bias_range=(100, 600),
):
    try:
        # light_header = fits.getheader(light)
        light_vals_raw = fits.getdata(light).astype("float32")[:-1, :]
        light_basename = os.path.basename(light)

        # Optimize bias for this specific light frame if requested
        if optimize_bias:
            from scipy.ndimage import median_filter
            from scipy.optimize import minimize_scalar

            # Pre-compute source mask using a rough bias estimate
            # This is independent of the exact bias value and only needs to be computed once
            rough_bias = np.mean(bias_range)
            temp_corrected = (light_vals_raw - rough_bias) / master_flat

            # Detect sources on downsampled image for speed (4x downsampling)
            # from skimage.transform import downscale_local_mean, resize

            # temp_small = downscale_local_mean(temp_corrected, (1, 1))

            # Use smaller filter for speed (downsampled, so use size=6 ~= 25/4)
            background_small = median_filter(temp_corrected, size=50)
            residual_small = temp_corrected - background_small

            # Compute robust statistics (median absolute deviation)
            median_res = np.nanmedian(residual_small)
            mad = np.nanmedian(np.abs(residual_small - median_res))
            sigma = 1.4826 * mad  # Convert MAD to std deviation

            # Mask pixels more than 3-sigma above background (sources)
            source_mask = residual_small > 3 * sigma

            # # Upscale mask back to full resolution
            # source_mask = (
            #     resize(
            #         source_mask_small.astype(float),
            #         light_vals_raw.shape,
            #         order=0,
            #         preserve_range=True,
            #         anti_aliasing=False,
            #     )
            #     > 0.5
            # )
            source_mask = source_mask | ~np.isfinite(master_flat)
            background_pixels = ~source_mask

            if np.sum(background_pixels) < 1000:
                logger.warning(
                    f"{light_basename}: Not enough background pixels, using simple optimization"
                )
                # Fall back to simple method without source masking
                background_pixels = np.isfinite(master_flat)

            def objective(bias):
                """Find bias that minimizes correlation with flat, using pre-computed mask."""
                light_minus_bias = light_vals_raw - bias
                corrected = light_minus_bias / master_flat

                # Use pre-computed background pixel mask
                valid = background_pixels & np.isfinite(corrected)

                if np.sum(valid) < 1000:
                    return 1.0

                # Compute correlation between corrected background and flat
                corr = np.corrcoef(
                    corrected[valid].flatten(),
                    master_flat[valid].flatten(),
                )[0, 1]

                return abs(corr)

            result = minimize_scalar(objective, bounds=bias_range, method="bounded")
            optimal_bias = result.x
            min_corr = result.fun
            logger.info(
                f"{light_basename}: optimal bias = {optimal_bias:.2f}, correlation = {min_corr:.6f}"
            )
            light_vals = light_vals_raw - optimal_bias
        else:
            light_vals = light_vals_raw - bias_level

        # Apply flat field correction
        light_vals = light_vals / master_flat.astype("float32")

        light_vals = remove_cosmic(light_vals)
        # light_vals = subtract_background(light_vals)
        # # wcs_solved_path = solve_field(
        # #     light, scale_low=0.2, scale_high=1.0, ra=ra, dec=dec, radius=10
        # # )
        light_vals = calculate_reprojection(
            source_hdus=(light_vals, fits.getheader(light)),
            target_wcs=output_wcs,
            shape_out=output_shape,
            order="bilinear",
        )
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
    filter_,
    pos,
    data_dir,
    output_dir,
    flat_path,
    target_name,
    bias_level=300.0,
    optimize_bias=False,
    bias_range=(290, 310),
    n_jobs=4,
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
    Only used if optimize_bias is False.
    optimize_bias : bool, optional
    If True, optimize the bias level independently for each light frame
    by minimizing correlation with the flat pattern, by default False.
    bias_range : tuple of float, optional
    (min, max) range to search for optimal bias per frame, by default (100, 600).
    Only used if optimize_bias is True.
    n_jobs : int, optional
    Number of parallel jobs for processing light frames, by default -1 (use all CPUs).

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
        light_data = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(process_light)(
                light,
                master_flat,
                output_wcs,
                output_shape,
                ra,
                dec,
                bias_level,
                optimize_bias,
                bias_range,
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

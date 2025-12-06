import glob
import os

import astroscrappy
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import minimize_scalar

from pesto_reduction.logger import make_time_named_logger

logger = make_time_named_logger()


def load_flats(data_dir, bias_level=300.0):
    """
    Load FITS files from a directory and subtract a constant bias level.

    Parameters
    ----------
    data_dir : str
    Path to the directory containing flat-field FITS files.
    bias_level : int, optional
    Constant value to subtract from each frame, by default 300.

    Returns
    -------
    list of ndarray
    List of 2D numpy arrays with bias-subtracted flat-field data.
    """
    flats = glob.glob(os.path.join(data_dir, "*.fits"))
    flat_data = [fits.getdata(f).astype("float32")[:-1, :] - bias_level for f in flats]
    return flat_data


def plot_mean_histogram(flat_data):
    """
    Plot a histogram of the mean pixel values for each flat frame.

    Parameters
    ----------
    flat_data : list of ndarray
    List of 2D numpy arrays representing flat-field frames.
    """
    means = [np.mean(arr) for arr in flat_data]
    plt.hist(means, bins=10, edgecolor="black")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Mean Values")
    plt.show()


def remove_cosmic(input_array):
    """
    Remove cosmic rays from a 2D array using the astroscrappy algorithm.

    Parameters
    ----------
    input_array : ndarray
    2D numpy array representing a single image.

    Returns
    -------
    ndarray
    Cleaned image with cosmic rays removed.
    """
    _, clean_array = astroscrappy.detect_cosmics(input_array, sepmed=False)
    return clean_array


def optimize_bias_level(
    light_path, flat_path, bias_range=(100, 600), center_crop_halfsize=500
):
    """
    Find optimal bias level by minimizing correlation between
    flat-corrected light frame and the flat field pattern.

    The correct bias level should result in a flat-corrected image
    that has minimal correlation with the flat field itself.

    Parameters
    ----------
    light_path : str
        Path to a representative light frame FITS file
    flat_path : str
        Path to directory containing flat field FITS files
    bias_range : tuple of float, optional
        (min, max) range to search for optimal bias, by default (100, 600)
    center_crop_halfsize : int, optional
        Half-size of central region for flat normalization, by default 500

    Returns
    -------
    optimal_bias : float
        The bias level that minimizes correlation
    """
    # Load raw data
    light_data = fits.getdata(light_path).astype("float32")[:-1, :]

    # Load and combine flats without bias subtraction initially
    flat_files = glob.glob(os.path.join(flat_path, "*.fits"))
    flat_frames = [fits.getdata(f).astype("float32")[:-1, :] for f in flat_files]
    flat_data = np.median(np.stack(flat_frames, axis=0), axis=0)

    def objective(bias):
        """Compute absolute correlation between corrected image and flat."""
        # Apply bias to both
        light_minus_bias = light_data - bias
        flat_minus_bias = flat_data - bias

        # Normalize flat to center region
        h, w = flat_minus_bias.shape
        center_crop = flat_minus_bias[
            h // 2 - center_crop_halfsize : h // 2 + center_crop_halfsize,
            w // 2 - center_crop_halfsize : w // 2 + center_crop_halfsize,
        ]
        flat_norm = flat_minus_bias / np.median(center_crop)

        # Apply flat correction
        corrected = light_minus_bias / flat_norm

        # Calculate correlation - we want this close to zero
        # Use valid pixels only (avoid NaN, inf)
        mask = np.isfinite(corrected) & np.isfinite(flat_norm)
        if np.sum(mask) < 100:
            return 1.0  # Bad bias value

        corr = np.corrcoef(corrected[mask].flatten(), flat_norm[mask].flatten())[0, 1]

        return abs(corr)

    # Optimize
    result = minimize_scalar(objective, bounds=bias_range, method="bounded")

    optimal_bias = result.x
    min_correlation = result.fun

    logger.info(f"Optimal bias level: {optimal_bias:.2f}")
    logger.info(f"Minimum correlation: {min_correlation:.6f}")

    return optimal_bias


def create_master_flat(
    data_dir: str,
    output_path: str,
    plot_hist: bool = True,
    bias_level: int = 300,
    center_crop_halfsize: int = 500,
    z_thresh: float = 3.0,
) -> None:
    """
    Create a master flat by median combining bias-subtracted frames,
    normalizing to the central region, and cleaning cosmic rays.

    Parameters
    ----------
    data_dir : str
    Path to the directory containing flat-field FITS files.
    output_path : str
    Path to save the resulting master flat FITS file.
    plot_hist : bool, optional
    Whether to display a histogram of the mean values of input flats,
    by default True.
    bias_level : int, optional
    Constant bias value to subtract from each frame, by default 300.
    center_crop_halfsize : int, optional
    Half-size of the square region in the center used for normalization,
    by default 500.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    flat_data = load_flats(data_dir, bias_level=bias_level)
    logger.info(f"Found {len(flat_data)} flats")
    # Compute per-image median and reject outliers
    medians = np.array([np.median(arr) for arr in flat_data])
    z_scores = (medians - np.mean(medians)) / np.std(medians)
    keep_mask = np.abs(z_scores) < z_thresh

    num_removed = len(flat_data) - np.sum(keep_mask)
    if num_removed > 0:
        logger.warning(f"Removed {num_removed} outlier flat(s) based on median counts.")

    flat_data = [img for img, keep in zip(flat_data, keep_mask) if keep]

    if len(flat_data) == 0:
        logger.error("No valid flats remaining after outlier rejection.")
        return

    if plot_hist:
        plot_mean_histogram(flat_data)

    stacked = np.stack(flat_data, axis=0)
    median_array = np.median(stacked, axis=0)

    h, w = median_array.shape
    center_crop = median_array[
        h // 2 - center_crop_halfsize : h // 2 + center_crop_halfsize,
        w // 2 - center_crop_halfsize : w // 2 + center_crop_halfsize,
    ]

    median_array /= np.median(center_crop)
    # cleaned_flat = remove_cosmic(median_array)
    print(f"Median value : {np.median(median_array)}")
    print(f"Max value: {np.max(median_array)}")
    print(f"Min value: {np.min(median_array)}")
    fits.writeto(f"{output_path}/master_flat.fits", median_array, overwrite=True)
    logger.info(f"Master flat saved to {output_path}/master_flat.fits")
    return None

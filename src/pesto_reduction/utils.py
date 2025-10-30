from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
import astroscrappy
import subprocess
from pathlib import Path
from astropy.coordinates import SkyCoord
import numpy as np
from scipy import ndimage


def get_coords(target_name):
    """
    Resolve an astronomical target name to (RA, Dec) coordinates using `astropy`.

    Parameters
    ----------
    target_name : str
        Name of the target object (e.g., 'M51', 'Arp94').

    Returns
    -------
    ra_deg : float or None
        Right Ascension in decimal degrees, or None if resolution fails.
    dec_deg : float or None
        Declination in decimal degrees, or None if resolution fails.
    """
    try:
        coord = SkyCoord.from_name(target_name)
        ra_deg = coord.ra.degree
        dec_deg = coord.dec.degree
        return ra_deg, dec_deg
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def remove_cosmic(input_array):
    """
    Detect and remove cosmic rays from a 2D image using L.A.Cosmic (astroscrappy).

    Parameters
    ----------
    input_array : ndarray
        2D numpy array representing the image.

    Returns
    -------
    ndarray
        Cleaned image with cosmic rays removed.
    """
    _, clean_array = astroscrappy.detect_cosmics(input_array, sigclip=2.0)
    return clean_array


def subtract_background(light):
    """
    Estimate and subtract the background from an astronomical image using 2D median filtering.

    Parameters
    ----------
    light : ndarray
        2D numpy array representing the image.

    Returns
    -------
    ndarray
        Background-subtracted image.
    """
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        light,
        (64, 64),
        filter_size=(5, 5),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator,
    )
    background = bkg.background
    subtracted = light - background
    return subtracted


def fill_nan_interpolation(data):
    """Fill NaN values using interpolation."""
    from scipy.interpolate import griddata

    mask = np.isfinite(data)
    if not np.any(mask):
        return np.zeros_like(data)

    if np.all(mask):
        return data

    # Get coordinates
    y, x = np.mgrid[: data.shape[0], : data.shape[1]]

    # Points with valid data
    points = np.column_stack((y[mask].ravel(), x[mask].ravel()))
    values = data[mask].ravel()

    # Points to interpolate
    xi = np.column_stack((y.ravel(), x.ravel()))

    # Interpolate
    try:
        interpolated = griddata(
            points, values, xi, method="linear", fill_value=np.median(values)
        )
        return interpolated.reshape(data.shape)
    except:
        # Fallback
        return np.full_like(data, np.nanmedian(data))


def robust_background_2d(data, box_size=64, filter_size=3, exclude_percentile=90):
    """
    More robust 2D background estimation that better handles bright sources.
    """
    try:
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        # Create mask to exclude bright sources
        threshold = np.nanpercentile(data[np.isfinite(data)], exclude_percentile)
        source_mask = data > threshold

        # Dilate mask to exclude source halos
        if np.any(source_mask):
            source_mask = ndimage.binary_dilation(source_mask, structure=disk(5))

        sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
        bkg_estimator = MedianBackground()

        bkg = Background2D(
            data,
            box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=~np.isfinite(data) | source_mask,
        )

        return bkg.background
    except:
        # Fallback: simple median filter
        from scipy.ndimage import median_filter

        # Mask bright sources
        finite_data = data[np.isfinite(data)]
        if len(finite_data) > 0:
            threshold = np.percentile(finite_data, exclude_percentile)
            masked_data = np.where(data > threshold, np.nan, data)

            # Apply median filter
            background = median_filter(masked_data, size=box_size // 2, mode="nearest")

            # Fill NaN values by interpolation
            background = fill_nan_interpolation(background)

            return background
        else:
            return np.zeros_like(data)


def solve_field(
    fits_path,
    scale_low=0.3,
    scale_high=2.5,
    ra=None,
    dec=None,
    radius=None,
    scale_units="arcsecperpix",
    downsample=4,
    overwrite=True,
    no_plots=True,
    additional_args=None,
):
    """
    Run astrometry.net's solve-field on a FITS image with no WCS.

    Parameters
    ----------
    fits_path : str or Path
        Path to the FITS file to solve.
    scale_low : float
        Minimum pixel scale (arcsec/pix).
    scale_high : float
        Maximum pixel scale (arcsec/pix).
    scale_units : str
        Units for scale (usually "arcsecperpix").
        ra, dec : float, optional
        Center RA and Dec in degrees.
    radius : float, optional
        Search radius in degrees.
    downsample : int
        Downsampling factor.
    overwrite : bool
        Overwrite existing files.
    no_plots : bool
        Disable plot generation.
    additional_args : list of str, optional
        Any additional arguments for solve-field.

    Returns
    -------
    Path
        Path to the WCS-added FITS file (usually `<name>.new`)
    """
    fits_path = Path(fits_path)
    if not fits_path.exists():
        raise FileNotFoundError(f"File not found: {fits_path}")

    cmd = [
        "solve-field",
        str(fits_path),
        "--scale-units",
        "arcsecperpix",
        "--scale-low",
        str(scale_low),
        "--scale-high",
        str(scale_high),
        "--downsample",
        str(downsample),
        "--tweak-order",
        "1",
        "--sigma",
        "1",
    ]

    if overwrite:
        cmd.append("--overwrite")
    if no_plots:
        cmd.append("--no-plots")
    if additional_args:
        cmd.extend(additional_args)
    if ra is not None and dec is not None:
        cmd.extend(["--ra", str(ra), "--dec", str(dec)])
        if radius:
            cmd.extend(["--radius", str(radius)])
    # try:
    print(" ".join(cmd))
    subprocess.run(
        " ".join(cmd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
        shell=True,
    )
    # except subprocess.CalledProcessError as e:
    #     print(f"solve-field failed: {e}")
    #     return None

    return fits_path.with_suffix(".new")

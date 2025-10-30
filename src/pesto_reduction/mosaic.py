from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from dfreproject import calculate_reprojection
from scipy import ndimage
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.convolution import Gaussian2DKernel, convolve
from skimage.filters import gaussian
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def estimate_background_2d(data, box_size=64, filter_size=3):
    """
    Estimate 2D background using a mesh-based approach.
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground

    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    bkg_estimator = MedianBackground()

    try:
        bkg = Background2D(
            data,
            box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=~np.isfinite(data),
        )
        return bkg.background
    except:
        # Fallback to simple background estimation
        return np.full_like(data, np.nanmedian(data))


def create_smooth_weight_map(data, edge_buffer=100, smooth_sigma=20):
    """
    Create a smoothly varying weight map with gradual edge transitions.
    """
    weight = np.ones_like(data, dtype=np.float32)

    # Zero weight for invalid pixels
    invalid_mask = ~np.isfinite(data) | (data == 0)
    weight[invalid_mask] = 0.0

    # Create distance transform from invalid regions and edges
    valid_mask = weight > 0

    if not np.any(valid_mask):
        return weight

    # Distance from invalid pixels (including image edges)
    dist = ndimage.distance_transform_edt(valid_mask)

    # Create edge tapering
    edge_weight = np.minimum(dist / edge_buffer, 1.0)

    # Apply smooth falloff using error function-like curve
    edge_weight = 0.5 * (1.0 + np.tanh((edge_weight - 0.5) * 6))
    weight *= edge_weight

    # Smooth the weight map to avoid sharp transitions
    if smooth_sigma > 0:
        weight = gaussian(weight, sigma=smooth_sigma, preserve_range=True)

    return weight


def match_images_photometrically(images, weights, method="overlap"):
    """
    Match images photometrically using overlap regions or scaling factors.
    """
    n_images = len(images)
    scale_factors = np.ones(n_images)
    background_corrections = np.zeros(n_images)

    # Reference is first image
    ref_idx = 0
    ref_image = images[ref_idx]
    ref_weight = weights[ref_idx]

    # Calculate 2D background for reference
    ref_bg_2d = estimate_background_2d(ref_image)
    ref_corrected = ref_image - ref_bg_2d

    for i in range(1, n_images):
        img = images[i]
        weight = weights[i]

        # Calculate 2D background for current image
        img_bg_2d = estimate_background_2d(img)
        img_corrected = img - img_bg_2d

        # Find overlap region with reference
        overlap_mask = (ref_weight > 0.3) & (weight > 0.3)

        if np.sum(overlap_mask) > 1000:  # Need substantial overlap
            # Get overlap data
            ref_overlap = ref_corrected[overlap_mask]
            img_overlap = img_corrected[overlap_mask]

            # Remove outliers using sigma clipping
            valid = (np.abs(ref_overlap) < 5 * mad_std(ref_overlap)) & (
                np.abs(img_overlap) < 5 * mad_std(img_overlap)
            )

            if np.sum(valid) > 100:
                ref_valid = ref_overlap[valid]
                img_valid = img_overlap[valid]

                # Robust scaling using median ratio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ratios = ref_valid / img_valid
                    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]

                    if len(ratios) > 0:
                        scale_factors[i] = np.median(ratios)

                    # Background difference after scaling
                    scaled_img_overlap = img_valid * scale_factors[i]
                    background_corrections[i] = np.median(
                        ref_valid - scaled_img_overlap
                    )

        print(
            f"[INFO] Image {i + 1}: scale={scale_factors[i]:.3f}, bg_corr={background_corrections[i]:.2f}"
        )

    return scale_factors, background_corrections


def plot_footprints(images, output_path):
    """
    Create a visualization showing the footprint of non-NaN data in each image.

    Parameters
    ----------
    images : list of ndarray
        List of images to visualize
    output_path : str
        Path to save the output figure
    """
    n_images = len(images)

    # Create figure with grid layout
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Define colors for each position
    colors = plt.cm.tab10(np.linspace(0, 1, n_images))

    for i, (img, ax) in enumerate(zip(images, axes)):
        # Create binary mask of non-NaN pixels
        mask = np.isfinite(img)

        # Display the mask
        ax.imshow(mask, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(
            f"Position {i + 1}\n{np.sum(mask):,} valid pixels ({100 * np.sum(mask) / mask.size:.1f}%)",
            fontsize=10,
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.grid(True, alpha=0.3)

    # Hide any extra subplots if n_images < 9
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.suptitle(
        f"Data Footprints for {n_images} Images", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO]   Footprint visualization saved to {output_path}")

    # Also create an overlay plot showing all footprints together
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create combined footprint showing overlap
    combined_mask = np.zeros(images[0].shape, dtype=int)
    for img in images:
        combined_mask += np.isfinite(img).astype(int)

    im = ax.imshow(combined_mask, cmap="hot", origin="lower", interpolation="nearest")
    ax.set_title(
        f"Combined Coverage Map\n(shows number of images covering each pixel)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of images", rotation=270, labelpad=20)

    plt.tight_layout()
    overlay_path = output_path.replace("_footprint.png", "_coverage.png")
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO]   Coverage map saved to {overlay_path}")


def combine_images_seamlessly(
    images, weights, backgrounds, scales, combine="weighted_mean"
):
    """
    Combine images with seamless blending.
    """
    output_shape = images[0].shape

    if combine == "weighted_mean":
        # Simple unweighted mean
        numerator = np.zeros(output_shape, dtype=np.float32)
        denominator = np.zeros(output_shape, dtype=np.float32)

        for i, (img, weight, bg, scale) in enumerate(
            zip(images, weights, backgrounds, scales)
        ):
            # No photometric corrections, no weighting
            # corrected_img = (img - bg) * scale
            corrected_img = img

            # Accumulate simple sum (no weighting)
            valid = np.isfinite(corrected_img)
            numerator[valid] += corrected_img[valid]
            denominator[valid] += 1.0

        # Final combination
        mosaic = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    elif combine == "median":
        # Stack all images without corrections
        stack = []

        for i, (img, weight, bg, scale) in enumerate(
            zip(images, weights, backgrounds, scales)
        ):
            # No photometric corrections
            # corrected_img = (img - bg) * scale
            stack.append(img)

        stack_array = np.stack(stack, axis=0)

        # Use numpy's nanmedian for vectorized operation (much faster)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mosaic = np.nanmedian(stack_array, axis=0).astype(np.float32)

    return mosaic


def mosaic_images(
    image_paths,
    output_path,
    combine="weighted_mean",
    edge_buffer=150,
    smooth_sigma=30,
    box_size=128,
):
    """
    Create a mosaic by stitching together images with different WCS coordinates.

    Parameters
    ----------
    image_paths : list of str
    Paths to the input FITS images to combine.
    output_path : str
    Path where the mosaic FITS file will be saved.
    combine : str, optional
    Combination method: 'weighted_mean' or 'median'. Default is 'weighted_mean'.
    edge_buffer : int, optional
    Buffer zone in pixels for edge tapering. Default is 150.
    smooth_sigma : float, optional
    Gaussian smoothing sigma for weight maps. Default is 30.
    box_size : int, optional
    Box size for 2D background estimation. Default is 128.

    Returns
    -------
    None
    """
    assert len(image_paths) >= 2, "You need at least 2 images to mosaic"

    print(f"[INFO] Creating mosaic from {len(image_paths)} images...")
    print(image_paths)

    # Step 1: Load all images and their WCS
    print("[INFO] Step 1: Loading images and determining mosaic footprint...")
    hdus = []
    wcs_list = []

    for path in image_paths:
        hdu = fits.open(path)[0]
        hdus.append(hdu)
        wcs_list.append(WCS(hdu.header))

    # Step 2: Find the combined footprint that encompasses all images
    # Get corner coordinates of all images in sky coordinates
    all_corners_ra = []
    all_corners_dec = []

    for i, (hdu, wcs) in enumerate(zip(hdus, wcs_list)):
        ny, nx = hdu.data.shape
        print(f"[INFO]   Image {i + 1} dimensions: {nx} x {ny}")
        # Get the four corners in pixel coordinates
        corners_pix = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1]])
        # Convert to sky coordinates
        corners_sky = wcs.pixel_to_world_values(corners_pix[:, 0], corners_pix[:, 1])
        print(
            f"[INFO]   Image {i + 1} RA range: {np.min(corners_sky[0]):.6f} to {np.max(corners_sky[0]):.6f}"
        )
        print(
            f"[INFO]   Image {i + 1} Dec range: {np.min(corners_sky[1]):.6f} to {np.max(corners_sky[1]):.6f}"
        )
        all_corners_ra.extend(corners_sky[0])
        all_corners_dec.extend(corners_sky[1])

    # Find the bounding box in RA/Dec
    ra_min, ra_max = np.min(all_corners_ra), np.max(all_corners_ra)
    dec_min, dec_max = np.min(all_corners_dec), np.max(all_corners_dec)

    print(f"[INFO]   RA range: {ra_min:.6f} to {ra_max:.6f}")
    print(f"[INFO]   Dec range: {dec_min:.6f} to {dec_max:.6f}")

    # Step 3: Create a new WCS for the output mosaic
    # Use the pixel scale from the first image
    ref_wcs = wcs_list[0]
    # Get pixel scale from pixel_scale_matrix (handles PC matrix properly)
    pixel_scale = np.abs(ref_wcs.pixel_scale_matrix[0, 0])  # degrees per pixel

    # Calculate output size
    ra_center = (ra_min + ra_max) / 2
    dec_center = (dec_min + dec_max) / 2

    # Account for cos(dec) factor in RA size
    ra_size = (ra_max - ra_min) * np.cos(np.radians(dec_center))
    dec_size = dec_max - dec_min

    output_nx = int(np.ceil(ra_size / pixel_scale))
    output_ny = int(np.ceil(dec_size / pixel_scale))

    # Handle case where all images have identical WCS (simple stacking, not mosaicing)
    identical_wcs = False
    if output_nx <= 1 or output_ny <= 1:
        print("[WARNING] All images appear to have identical WCS coordinates!")
        print("[INFO]   Skipping reprojection - will stack images directly...")
        identical_wcs = True
        output_ny, output_nx = hdus[0].data.shape
        output_wcs = ref_wcs.deepcopy()  # Use exact WCS from first image
        output_shape = (output_ny, output_nx)
    else:
        # Create new WCS centered on the mosaic
        output_wcs = WCS(naxis=2)
        output_wcs.wcs.crpix = [output_nx / 2, output_ny / 2]
        output_wcs.wcs.crval = [ra_center, dec_center]
        output_wcs.wcs.cdelt = ref_wcs.wcs.cdelt
        output_wcs.wcs.ctype = ref_wcs.wcs.ctype
        if hasattr(ref_wcs.wcs, "pc"):
            output_wcs.wcs.pc = ref_wcs.wcs.pc
        output_shape = (output_ny, output_nx)

    print(f"[INFO]   Output mosaic size: {output_nx} x {output_ny} pixels")
    print(f"[INFO]   Pixel scale: {pixel_scale * 3600:.3f} arcsec/pixel")

    # Step 4: Reproject all images to the new WCS (or just load if identical WCS)
    if identical_wcs:
        print("[INFO] Step 2: Loading images (no reprojection needed)...")
        images = []
        for i, hdu in enumerate(hdus):
            print(f"[INFO]   Loading image {i + 1}/{len(hdus)}")
            images.append(hdu.data.astype("float32"))
    else:
        print("[INFO] Step 2: Reprojecting images to common grid...")
        images = []
        for i, (hdu, wcs) in enumerate(zip(hdus, wcs_list)):
            print(f"[INFO]   Reprojecting image {i + 1}/{len(hdus)}")

            reprojected = calculate_reprojection(
                source_hdus=(hdu.data.astype("float32"), hdu.header),
                target_wcs=output_wcs,
                shape_out=output_shape,
                order="bilinear",
            )

            images.append(reprojected)

    # No background subtraction, no photometric matching
    backgrounds_2d = [np.zeros_like(img) for img in images]
    scales = np.ones(len(images))
    weights = [np.ones_like(img) for img in images]  # Simple weighting

    print(f"[INFO] Step 3: Combining using {combine} method (no corrections)...")

    # Final combination
    mosaic = combine_images_seamlessly(images, weights, backgrounds_2d, scales, combine)

    print("[INFO] Step 4: Creating footprint visualization...")

    # Create footprint plot
    plot_footprints(images, output_path.replace(".fits", "_footprint.png"))

    print("[INFO] Step 5: Saving result...")

    # Create output HDU with proper WCS
    output_hdu = fits.PrimaryHDU(data=mosaic.astype(np.float32))
    output_hdu.header.update(output_wcs.to_header())
    output_hdu.header["HISTORY"] = f"Mosaic from {len(image_paths)} images"
    output_hdu.header["COMBTYPE"] = (combine, "Combination method")
    output_hdu.header["NIMAGES"] = (len(image_paths), "Number of images")

    # Save mosaic
    output_hdu.writeto(output_path, overwrite=True)

    print(f"[INFO] Mosaic saved to {output_path}")

    # Final statistics
    valid_pixels = np.sum(np.isfinite(mosaic))
    total_pixels = mosaic.size
    coverage = 100.0 * valid_pixels / total_pixels

    print(f"[INFO] Final statistics:")
    print(f"[INFO]   Coverage: {coverage:.1f}% ({valid_pixels}/{total_pixels} pixels)")
    print(f"[INFO]   Mean: {np.nanmean(mosaic):.2f}")
    print(f"[INFO]   Std: {np.nanstd(mosaic):.2f}")
    print(f"[INFO]   Range: [{np.nanmin(mosaic):.2f}, {np.nanmax(mosaic):.2f}]")


# Additional utility function for quality assessment
def assess_mosaic_quality(mosaic_path, plot=True):
    """
    Assess the quality of a mosaic by looking for seams and artifacts.
    """
    hdu = fits.open(mosaic_path)[0]
    data = hdu.data

    # Calculate local standard deviation to highlight seams
    from scipy.ndimage import uniform_filter

    local_std = np.sqrt(
        uniform_filter(data**2, size=20) - uniform_filter(data, size=20) ** 2
    )

    # Statistics
    print(f"[INFO] Mosaic quality assessment for {mosaic_path}:")
    print(
        f"[INFO]   Local std deviation range: [{np.nanmin(local_std):.3f}, {np.nanmax(local_std):.3f}]"
    )
    print(f"[INFO]   90th percentile local std: {np.nanpercentile(local_std, 90):.3f}")

    if plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Original mosaic
        im1 = ax1.imshow(
            data,
            origin="lower",
            cmap="gray",
            vmin=np.nanpercentile(data, 1),
            vmax=np.nanpercentile(data, 99),
        )
        ax1.set_title("Mosaic")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")

        # Local standard deviation (highlights seams)
        im2 = ax2.imshow(
            local_std, origin="lower", cmap="hot", vmax=np.nanpercentile(local_std, 95)
        )
        ax2.set_title("Local Standard Deviation\n(bright = potential seams)")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")

        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    image_list = ["image1.fits", "image2.fits", "image3.fits"]

    # Create seamless mosaic
    mosaic_images(
        image_list,
        "seamless_mosaic.fits",
        combine="weighted_mean",
        edge_buffer=200,  # Larger buffer for smoother edges
        smooth_sigma=50,  # More smoothing for seamless blending
        box_size=64,
    )  # Finer background estimation

    # Assess quality
    # assess_mosaic_quality("seamless_mosaic.fits")

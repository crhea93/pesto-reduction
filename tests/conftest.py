"""
Pytest configuration and shared fixtures for pesto_reduction tests.
"""

import pytest
import numpy as np
from astropy.io import fits


@pytest.fixture
def sample_image():
    """Create a sample astronomical image for testing."""
    # Create a 100x100 image with background
    image = np.full((100, 100), 100.0, dtype=np.float32)
    # Add Gaussian noise
    image += np.random.normal(0, 10, image.shape)
    return image


@pytest.fixture
def sample_image_with_source():
    """Create a sample image with a point source."""
    # Background
    image = np.full((100, 100), 100.0, dtype=np.float32)
    # Add a Gaussian source in the center
    y, x = np.ogrid[:100, :100]
    source = 1000 * np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / (2 * 5**2))
    image += source
    return image


@pytest.fixture
def sample_flat_frames(tmp_path):
    """Create sample flat field frames in a temporary directory."""
    flat_dir = tmp_path / "flats"
    flat_dir.mkdir()

    # Create 5 flat frames with slight variations
    for i in range(5):
        data = np.full((100, 100), 1000.0 + i * 10, dtype=np.float32)
        # Add some variation
        data += np.random.normal(0, 5, data.shape)
        fits.writeto(flat_dir / f"flat_{i:03d}.fits", data, overwrite=True)

    return flat_dir


@pytest.fixture
def sample_master_flat():
    """Create a sample master flat field."""
    # Create a flat field that's roughly uniform with slight vignetting
    y, x = np.ogrid[:100, :100]
    center_y, center_x = 50, 50
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    # Vignetting effect (brightness decreases toward edges)
    flat = 1.0 - 0.2 * (r / 50) ** 2
    flat = np.clip(flat, 0.5, 1.0).astype(np.float32)
    return flat


@pytest.fixture
def sample_fits_file(tmp_path, sample_image):
    """Create a sample FITS file."""
    fits_path = tmp_path / "test_image.fits"
    fits.writeto(fits_path, sample_image, overwrite=True)
    return fits_path


@pytest.fixture
def sample_fits_with_header(tmp_path, sample_image):
    """Create a sample FITS file with header metadata."""
    fits_path = tmp_path / "test_image_with_header.fits"
    header = fits.Header()
    header["FILTER"] = "Ha"
    header["OBJECT"] = "M51"
    header["EXPTIME"] = 300.0
    fits.writeto(fits_path, sample_image, header=header, overwrite=True)
    return fits_path


@pytest.fixture
def cosmic_ray_image():
    """Create an image with artificial cosmic rays."""
    image = np.full((100, 100), 100.0, dtype=np.float32)
    # Add some cosmic rays (hot pixels)
    cosmic_positions = [(25, 25), (50, 50), (75, 75), (30, 70)]
    for y, x in cosmic_positions:
        image[y, x] = 10000.0
        # Add some smaller nearby pixels
        if x > 0:
            image[y, x - 1] = 5000.0
        if y > 0:
            image[y - 1, x] = 5000.0
    return image


@pytest.fixture
def nan_image():
    """Create an image with NaN values for testing interpolation."""
    image = np.ones((50, 50), dtype=np.float32)
    # Add some NaN values
    image[10:15, 10:15] = np.nan
    image[30, 30] = np.nan
    image[0, 0] = np.nan
    return image


@pytest.fixture
def image_with_gradient():
    """Create an image with a background gradient."""
    y, x = np.mgrid[:100, :100]
    # Linear gradient from bottom-left to top-right
    gradient = 50 + 0.5 * x + 0.3 * y
    return gradient.astype(np.float32)

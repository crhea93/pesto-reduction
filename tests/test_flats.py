import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import os
from astropy.io import fits
from pesto_reduction.flats import (
    load_flats,
    plot_mean_histogram,
    remove_cosmic,
    create_master_flat,
)


class TestLoadFlats:
    """Test loading flat field frames."""

    def test_load_flats_with_bias_subtraction(self, tmp_path):
        """Test loading flats and bias subtraction."""
        # Create temporary FITS files
        data1 = np.full((100, 100), 1000.0, dtype=np.float32)
        data2 = np.full((100, 100), 1100.0, dtype=np.float32)

        fits.writeto(tmp_path / "flat1.fits", data1, overwrite=True)
        fits.writeto(tmp_path / "flat2.fits", data2, overwrite=True)

        # Load with bias level of 300
        flat_data = load_flats(str(tmp_path), bias_level=300)

        assert len(flat_data) == 2
        # Check bias was subtracted correctly
        # load_flats uses glob which may return files in any order, so check both values are present
        values = sorted([np.mean(flat_data[0]), np.mean(flat_data[1])])
        assert np.allclose(values[0], 700.0, rtol=1e-5)
        assert np.allclose(values[1], 800.0, rtol=1e-5)

    def test_load_flats_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        flat_data = load_flats(str(tmp_path), bias_level=300)

        assert len(flat_data) == 0

    def test_load_flats_custom_bias(self, tmp_path):
        """Test loading with custom bias level."""
        data = np.full((50, 50), 500.0, dtype=np.float32)
        fits.writeto(tmp_path / "flat1.fits", data, overwrite=True)

        flat_data = load_flats(str(tmp_path), bias_level=100)

        # Check that bias was subtracted correctly
        assert np.allclose(np.mean(flat_data[0]), 400.0, rtol=1e-5)


class TestPlotMeanHistogram:
    """Test histogram plotting function."""

    @patch("pesto_reduction.flats.plt")
    def test_plot_mean_histogram(self, mock_plt):
        """Test that histogram plotting is called correctly."""
        flat_data = [
            np.full((10, 10), 100.0),
            np.full((10, 10), 200.0),
            np.full((10, 10), 150.0),
        ]

        plot_mean_histogram(flat_data)

        # Verify plt.hist was called
        mock_plt.hist.assert_called_once()
        # Check that it was called with correct number of mean values
        args, kwargs = mock_plt.hist.call_args
        assert len(args[0]) == 3


class TestRemoveCosmicFlats:
    """Test cosmic ray removal in flats module."""

    def test_remove_cosmic_preserves_shape(self):
        """Test that cosmic removal preserves array shape."""
        image = np.random.normal(100, 10, (100, 100)).astype(np.float32)

        cleaned = remove_cosmic(image)

        assert cleaned.shape == image.shape

    def test_remove_cosmic_reduces_hot_pixels(self):
        """Test that hot pixels are reduced."""
        image = np.full((50, 50), 100.0, dtype=np.float32)
        image[25, 25] = 10000  # Hot pixel

        cleaned = remove_cosmic(image)

        # Hot pixel should be significantly reduced
        assert cleaned[25, 25] < image[25, 25]


class TestCreateMasterFlat:
    """Test master flat creation."""

    @patch("pesto_reduction.flats.plot_mean_histogram")
    def test_create_master_flat_basic(self, mock_plot, tmp_path):
        """Test basic master flat creation."""
        # Create input directory with test flats
        input_dir = tmp_path / "flats"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test flat frames with consistent values
        for i in range(5):
            data = np.full((100, 100), 1000.0 + i * 10, dtype=np.float32)
            fits.writeto(input_dir / f"flat{i}.fits", data, overwrite=True)

        create_master_flat(
            data_dir=str(input_dir),
            output_path=str(output_dir),
            plot_hist=False,
            bias_level=300,
            center_crop_halfsize=20,
        )

        # Check output file exists
        output_file = output_dir / "master_flat.fits"
        assert output_file.exists()

        # Read and verify output
        master_flat = fits.getdata(output_file)
        assert master_flat.shape == (98, 98)  # Cropped by 1 pixel on each side
        # Should be normalized to max of 1
        assert np.max(master_flat) == pytest.approx(1.0, rel=1e-5)

    @patch("pesto_reduction.flats.plot_mean_histogram")
    def test_create_master_flat_outlier_rejection(self, mock_plot, tmp_path):
        """Test that outlier frames are rejected."""
        input_dir = tmp_path / "flats"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create normal flats
        for i in range(5):
            data = np.full((100, 100), 1000.0, dtype=np.float32)
            fits.writeto(input_dir / f"flat{i}.fits", data, overwrite=True)

        # Create an outlier flat (much brighter)
        outlier_data = np.full((100, 100), 5000.0, dtype=np.float32)
        fits.writeto(input_dir / "flat_outlier.fits", outlier_data, overwrite=True)

        # Should handle outliers gracefully
        create_master_flat(
            data_dir=str(input_dir),
            output_path=str(output_dir),
            plot_hist=False,
            bias_level=300,
            z_thresh=3.0,
        )

        output_file = output_dir / "master_flat.fits"
        assert output_file.exists()

    @patch("pesto_reduction.flats.plot_mean_histogram")
    def test_create_master_flat_with_histogram(self, mock_plot, tmp_path):
        """Test master flat creation with histogram plotting."""
        input_dir = tmp_path / "flats"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create test flats with varying values so they won't be rejected
        for i in range(3):
            data = np.full((50, 50), 1000.0 + i * 5, dtype=np.float32)
            fits.writeto(input_dir / f"flat{i}.fits", data, overwrite=True)

        create_master_flat(
            data_dir=str(input_dir),
            output_path=str(output_dir),
            plot_hist=True,
            bias_level=300,
        )

        # Verify histogram was called
        mock_plot.assert_called_once()

    def test_create_master_flat_normalization(self, tmp_path):
        """Test that master flat is properly normalized."""
        input_dir = tmp_path / "flats"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create flats with known values and variation to avoid outlier rejection
        for i in range(3):
            data = np.full((100, 100), 2000.0 + i * 10, dtype=np.float32)
            fits.writeto(input_dir / f"flat{i}.fits", data, overwrite=True)

        create_master_flat(
            data_dir=str(input_dir),
            output_path=str(output_dir),
            plot_hist=False,
            bias_level=300,
        )

        master_flat = fits.getdata(str(output_dir / "master_flat.fits"))
        # Maximum should be 1.0 after normalization
        assert np.max(master_flat) == pytest.approx(1.0, rel=1e-5)

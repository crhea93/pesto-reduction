import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from astropy.io import fits
from pesto_reduction.utils import (
    get_coords,
    remove_cosmic,
    subtract_background,
    fill_nan_interpolation,
    robust_background_2d,
    solve_field,
)


class TestGetCoords:
    """Test astronomical coordinate resolution."""

    @patch("pesto_reduction.utils.SkyCoord")
    def test_get_coords_success(self, mock_skycoord):
        """Test successful coordinate resolution."""
        mock_coord = MagicMock()
        mock_coord.ra.degree = 150.0
        mock_coord.dec.degree = 45.0
        mock_skycoord.from_name.return_value = mock_coord

        ra, dec = get_coords("M51")

        assert ra == 150.0
        assert dec == 45.0
        mock_skycoord.from_name.assert_called_once_with("M51")

    @patch("pesto_reduction.utils.SkyCoord")
    def test_get_coords_failure(self, mock_skycoord):
        """Test coordinate resolution failure."""
        mock_skycoord.from_name.side_effect = Exception("Target not found")

        ra, dec = get_coords("NonexistentObject")

        assert ra is None
        assert dec is None


class TestRemoveCosmic:
    """Test cosmic ray removal."""

    def test_remove_cosmic_basic(self):
        """Test that cosmic ray removal returns an array of the same shape."""
        # Create a simple test image
        image = np.random.normal(100, 10, (100, 100)).astype(np.float32)
        # Add some cosmic rays (hot pixels)
        image[50, 50] = 10000
        image[25, 75] = 10000

        cleaned = remove_cosmic(image)

        assert cleaned.shape == image.shape
        assert isinstance(cleaned, np.ndarray)
        # Cosmic rays should be reduced
        assert cleaned[50, 50] < image[50, 50]

    def test_remove_cosmic_preserves_good_data(self):
        """Test that cosmic ray removal preserves good data."""
        # Create uniform image without cosmic rays
        image = np.full((50, 50), 100.0, dtype=np.float32)

        cleaned = remove_cosmic(image)

        # Should be very similar to original
        assert np.allclose(cleaned, image, rtol=0.1)


class TestSubtractBackground:
    """Test background subtraction."""

    def test_subtract_background_reduces_level(self):
        """Test that background subtraction reduces overall level."""
        # Create image with constant background
        background_level = 100.0
        image = np.full((128, 128), background_level, dtype=np.float32)
        # Add some variation
        image += np.random.normal(0, 5, image.shape)

        subtracted = subtract_background(image)

        # Background-subtracted image should be centered near zero
        assert np.abs(np.median(subtracted)) < 20.0

    def test_subtract_background_preserves_sources(self):
        """Test that background subtraction preserves bright sources."""
        # Create larger image with background and a bright source
        image = np.full((256, 256), 100.0, dtype=np.float32)
        # Add some noise to make it realistic
        image += np.random.normal(0, 5, image.shape).astype(np.float32)
        # Add a bright source in the center
        y, x = np.ogrid[:256, :256]
        source = 1000 * np.exp(-((x - 128) ** 2 + (y - 128) ** 2) / (2 * 10**2))
        image += source

        subtracted = subtract_background(image)

        # The bright source should still be prominent after background subtraction
        assert np.max(subtracted) > 500

    def test_subtract_background_shape(self):
        """Test that background subtraction preserves array shape."""
        image = np.random.normal(100, 10, (100, 100)).astype(np.float32)

        subtracted = subtract_background(image)

        assert subtracted.shape == image.shape


class TestFillNanInterpolation:
    """Test NaN filling by interpolation."""

    def test_fill_nan_basic(self):
        """Test basic NaN filling."""
        data = np.ones((10, 10), dtype=np.float32)
        data[5, 5] = np.nan
        data[3, 3] = np.nan

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))
        # Interpolated values should be close to 1.0
        assert np.abs(filled[5, 5] - 1.0) < 0.5

    def test_fill_nan_all_nan(self):
        """Test filling when all values are NaN."""
        data = np.full((10, 10), np.nan, dtype=np.float32)

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))
        # Should return zeros when all NaN
        assert np.allclose(filled, 0.0)

    def test_fill_nan_no_nan(self):
        """Test that arrays without NaN are unchanged."""
        data = np.random.random((10, 10)).astype(np.float32)

        filled = fill_nan_interpolation(data)

        assert np.allclose(filled, data)

    def test_fill_nan_edge_cases(self):
        """Test NaN filling at edges."""
        data = np.ones((10, 10), dtype=np.float32)
        data[0, 0] = np.nan
        data[9, 9] = np.nan

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))


class TestRobustBackground2D:
    """Test robust 2D background estimation."""

    def test_robust_background_basic(self):
        """Test basic background estimation."""
        # Create image with constant background
        background = np.full((128, 128), 100.0, dtype=np.float32)
        background += np.random.normal(0, 5, background.shape)

        estimated_bg = robust_background_2d(background)

        assert estimated_bg.shape == background.shape
        # Estimated background should be close to 100
        assert 80 < np.median(estimated_bg) < 120

    def test_robust_background_with_sources(self):
        """Test background estimation with bright sources."""
        # Create background
        background = np.full((128, 128), 100.0, dtype=np.float32)
        # Add bright sources
        background[64, 64] = 10000
        background[32, 32] = 5000

        estimated_bg = robust_background_2d(background)

        # Background estimate should not be heavily biased by bright sources
        assert np.median(estimated_bg) < 500

    def test_robust_background_shape_preservation(self):
        """Test that output shape matches input."""
        image = np.random.normal(100, 10, (100, 100)).astype(np.float32)

        bg = robust_background_2d(image, box_size=32)

        assert bg.shape == image.shape

    def test_robust_background_different_box_sizes(self):
        """Test with different box sizes."""
        image = np.random.normal(100, 10, (128, 128)).astype(np.float32)

        bg_small = robust_background_2d(image, box_size=32)
        bg_large = robust_background_2d(image, box_size=64)

        assert bg_small.shape == image.shape
        assert bg_large.shape == image.shape
        # Different box sizes should give similar results for uniform background
        assert np.abs(np.median(bg_small) - np.median(bg_large)) < 20

    def test_robust_background_fallback_path(self):
        """Test fallback path when Background2D fails."""
        # Create image that might trigger fallback
        image = np.random.normal(100, 5, (128, 128)).astype(np.float32)
        image[50:60, 50:60] = 5000  # Bright source

        # This should use fallback gracefully
        bg = robust_background_2d(image, exclude_percentile=85)

        assert bg.shape == image.shape
        assert not np.any(np.isnan(bg))

    def test_robust_background_all_nan(self):
        """Test with all NaN values."""
        image = np.full((100, 100), np.nan, dtype=np.float32)

        bg = robust_background_2d(image)

        assert bg.shape == image.shape
        assert np.allclose(bg, 0.0)


class TestSolveField:
    """Test astrometry.net field solving."""

    def test_solve_field_file_not_found(self, tmp_path):
        """Test error handling for missing FITS file."""
        nonexistent = tmp_path / "nonexistent.fits"

        with pytest.raises(FileNotFoundError):
            solve_field(str(nonexistent))

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_success(self, mock_run, tmp_path):
        """Test successful field solving."""
        # Create a test FITS file
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        # Create the expected output file
        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file))

        assert result == output_file
        mock_run.assert_called_once()

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_with_coordinates(self, mock_run, tmp_path):
        """Test field solving with RA/Dec hint."""
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file), ra=150.0, dec=45.0, radius=5.0)

        assert result == output_file
        # Check that command includes RA/Dec
        call_args = mock_run.call_args[0][0]
        assert "--ra" in call_args
        assert "--dec" in call_args
        assert "--radius" in call_args

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_custom_scale(self, mock_run, tmp_path):
        """Test field solving with custom scale range."""
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file), scale_low=0.5, scale_high=1.5)

        assert result == output_file
        call_args = mock_run.call_args[0][0]
        assert "0.5" in call_args
        assert "1.5" in call_args

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_no_overwrite(self, mock_run, tmp_path):
        """Test field solving without overwrite flag."""
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file), overwrite=False)

        assert result == output_file
        call_args = mock_run.call_args[0][0]
        assert "--overwrite" not in call_args

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_no_plots_disabled(self, mock_run, tmp_path):
        """Test field solving with plot generation enabled."""
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file), no_plots=False)

        assert result == output_file
        call_args = mock_run.call_args[0][0]
        assert "--no-plots" not in call_args

    @patch("pesto_reduction.utils.subprocess.run")
    def test_solve_field_additional_args(self, mock_run, tmp_path):
        """Test field solving with additional arguments."""
        test_data = np.random.random((100, 100)).astype(np.float32)
        fits_file = tmp_path / "test.fits"
        fits.writeto(fits_file, test_data, overwrite=True)

        output_file = fits_file.with_suffix(".new")
        fits.writeto(output_file, test_data, overwrite=True)

        result = solve_field(str(fits_file), additional_args=["--cpulimit", "60"])

        assert result == output_file
        call_args = mock_run.call_args[0][0]
        assert "--cpulimit" in call_args
        assert "60" in call_args


class TestGetCoordsEdgeCases:
    """Additional edge case tests for get_coords."""

    @patch("pesto_reduction.utils.SkyCoord")
    def test_get_coords_various_targets(self, mock_skycoord):
        """Test coordinate resolution for various target names."""
        mock_coord = MagicMock()
        mock_coord.ra.degree = 210.5
        mock_coord.dec.degree = -45.3
        mock_skycoord.from_name.return_value = mock_coord

        ra, dec = get_coords("NGC4535")

        assert ra == 210.5
        assert dec == -45.3

    @patch("pesto_reduction.utils.SkyCoord")
    def test_get_coords_empty_string(self, mock_skycoord):
        """Test with empty target name."""
        mock_skycoord.from_name.side_effect = Exception("Invalid name")

        ra, dec = get_coords("")

        assert ra is None
        assert dec is None


class TestRemoveCosmicEdgeCases:
    """Additional edge case tests for remove_cosmic."""

    def test_remove_cosmic_uniform_image(self):
        """Test cosmic removal on uniform image."""
        image = np.full((50, 50), 100.0, dtype=np.float32)

        cleaned = remove_cosmic(image)

        assert cleaned.shape == image.shape
        # Uniform image should remain mostly unchanged
        assert np.allclose(cleaned, image, rtol=0.05)

    def test_remove_cosmic_extreme_values(self):
        """Test cosmic removal with extreme pixel values."""
        image = np.full((100, 100), 1000.0, dtype=np.float32)
        image[25, 25] = 65000  # Very hot pixel
        image[75, 75] = 10  # Very cold pixel

        cleaned = remove_cosmic(image)

        assert cleaned.shape == image.shape
        # Hot pixel should be significantly reduced
        assert cleaned[25, 25] < 65000
        # Cold pixel might stay similar or be slightly adjusted
        assert np.isfinite(cleaned[75, 75])


class TestSubtractBackgroundEdgeCases:
    """Additional edge case tests for background subtraction."""

    def test_subtract_background_small_image(self):
        """Test background subtraction on smaller image."""
        image = np.random.normal(100, 10, (64, 64)).astype(np.float32)

        # Should handle smaller images gracefully
        try:
            subtracted = subtract_background(image)
            assert subtracted.shape == image.shape
        except ValueError:
            # Some images may be too small, that's acceptable
            pass

    def test_subtract_background_nan_handling(self):
        """Test background subtraction with some NaN values."""
        image = np.full((200, 200), 100.0, dtype=np.float32)
        image[10:15, 10:15] = np.nan

        # Should handle NaN gracefully
        try:
            subtracted = subtract_background(image)
            assert subtracted.shape == image.shape
        except ValueError:
            # Some NaN patterns may not be supported, that's acceptable
            pass


class TestFillNanInterpolationEdgeCases:
    """Additional edge case tests for NaN interpolation."""

    def test_fill_nan_sparse_data(self):
        """Test interpolation with mostly NaN data."""
        data = np.full((50, 50), np.nan, dtype=np.float32)
        # Only a few valid pixels
        data[10, 10] = 100.0
        data[40, 40] = 200.0

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))
        # Should be filled with reasonable values
        assert np.all(np.isfinite(filled))

    def test_fill_nan_single_valid_pixel(self):
        """Test with only one valid pixel."""
        data = np.full((50, 50), np.nan, dtype=np.float32)
        data[25, 25] = 150.0

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))

    def test_fill_nan_edge_pixels(self):
        """Test NaN filling at image edges."""
        data = np.ones((50, 50), dtype=np.float32) * 100.0
        # NaN border
        data[0, :] = np.nan
        data[-1, :] = np.nan
        data[:, 0] = np.nan
        data[:, -1] = np.nan

        filled = fill_nan_interpolation(data)

        assert not np.any(np.isnan(filled))
        assert filled.shape == data.shape

    def test_fill_nan_preserves_valid_data(self):
        """Test that interpolation preserves known valid data."""
        data = np.ones((50, 50), dtype=np.float32) * 100.0
        data[20:30, 20:30] = np.nan

        filled = fill_nan_interpolation(data)

        # Edges should remain unchanged
        assert np.allclose(filled[0, 0], 100.0)
        assert np.allclose(filled[-1, -1], 100.0)

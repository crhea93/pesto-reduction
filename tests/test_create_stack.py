"""
Tests for create_stack module.
"""

import numpy as np
import pytest
from unittest.mock import patch
from astropy.wcs import WCS
from pesto_reduction.create_stack import (
    build_global_wcs,
    process_light,
)


class TestBuildGlobalWCS:
    """Test WCS grid creation."""

    def test_build_global_wcs_basic(self):
        """Test basic WCS creation."""
        ra, dec = 150.0, 45.0
        pixel_scale = 1.0
        field_size = 10.0

        wcs, shape = build_global_wcs(ra, dec, pixel_scale, field_size)

        assert isinstance(wcs, WCS)
        assert len(shape) == 2
        assert shape[0] > 0 and shape[1] > 0

    def test_build_global_wcs_center(self):
        """Test that WCS is centered correctly."""
        ra, dec = 180.0, 30.0
        wcs, shape = build_global_wcs(ra, dec)

        # Check CRVAL (center coordinate)
        assert wcs.wcs.crval[0] == ra
        assert wcs.wcs.crval[1] == dec

    def test_build_global_wcs_pixel_scale(self):
        """Test pixel scale is set correctly."""
        pixel_scale = 0.5  # arcsec/pixel
        wcs, shape = build_global_wcs(150.0, 45.0, pixel_scale=pixel_scale)

        # CD matrix should reflect the pixel scale
        scale_deg = pixel_scale / 3600.0
        assert np.abs(wcs.wcs.cd[0, 0]) == pytest.approx(scale_deg, rel=1e-6)
        assert np.abs(wcs.wcs.cd[1, 1]) == pytest.approx(scale_deg, rel=1e-6)

    def test_build_global_wcs_field_size(self):
        """Test different field sizes."""
        # Small field
        wcs_small, shape_small = build_global_wcs(
            150.0, 45.0, pixel_scale=1.0, field_size_arcmin=5.0
        )
        # Large field
        wcs_large, shape_large = build_global_wcs(
            150.0, 45.0, pixel_scale=1.0, field_size_arcmin=20.0
        )

        # Larger field should have more pixels
        assert shape_large[0] > shape_small[0]
        assert shape_large[1] > shape_small[1]

    def test_build_global_wcs_shape(self):
        """Test output shape calculation."""
        pixel_scale = 1.0  # arcsec/pixel
        field_size = 10.0  # arcminutes

        wcs, shape = build_global_wcs(150.0, 45.0, pixel_scale, field_size)

        # Expected size: 10 arcmin = 600 arcsec / 1 arcsec/pixel = 600 pixels
        expected_size = int(np.ceil(600))
        assert shape[0] == expected_size
        assert shape[1] == expected_size

    def test_build_global_wcs_ctype(self):
        """Test coordinate type is set correctly."""
        wcs, _ = build_global_wcs(150.0, 45.0)

        assert wcs.wcs.ctype[0] == "RA---TAN"
        assert wcs.wcs.ctype[1] == "DEC--TAN"


class TestProcessLight:
    """Test light frame processing (with mocking)."""

    @patch("pesto_reduction.create_stack.solve_field")
    @patch("pesto_reduction.create_stack.calculate_reprojection")
    @patch("pesto_reduction.create_stack.subtract_background")
    @patch("pesto_reduction.create_stack.remove_cosmic")
    @patch("pesto_reduction.create_stack.fits")
    def test_process_light_success(
        self,
        mock_fits,
        mock_remove_cosmic,
        mock_subtract_bg,
        mock_reproject,
        mock_solve,
        tmp_path,
    ):
        """Test successful light frame processing."""
        # Setup mocks
        test_data = np.full(
            (102, 102), 1000.0, dtype=np.float32
        )  # Larger to account for cropping
        master_flat = np.ones((100, 100), dtype=np.float32)
        output_data = np.full((100, 100), 500.0, dtype=np.float32)

        mock_fits.getdata.return_value = test_data
        mock_fits.getheader.return_value = {}
        mock_remove_cosmic.return_value = output_data
        mock_subtract_bg.return_value = output_data
        mock_reproject.return_value = output_data

        # Create a dummy FITS file for solve_field to return
        from astropy.io import fits as astropy_fits

        solved_file = tmp_path / "test.new"
        astropy_fits.writeto(solved_file, test_data, overwrite=True)
        mock_solve.return_value = solved_file

        wcs = WCS(naxis=2)
        output_shape = (100, 100)

        result = process_light(
            str(tmp_path / "light.fits"),
            master_flat,
            wcs,
            output_shape,
            150.0,
            45.0,
        )

        assert result is not None
        assert result.shape == output_shape

    @patch("pesto_reduction.create_stack.solve_field")
    @patch("pesto_reduction.create_stack.fits")
    def test_process_light_solve_failure(self, mock_fits, mock_solve, tmp_path):
        """Test handling of plate solve failure."""
        mock_fits.getdata.return_value = np.ones((100, 100), dtype=np.float32)
        mock_solve.side_effect = FileNotFoundError("Solve failed")

        master_flat = np.ones((100, 100), dtype=np.float32)
        wcs = WCS(naxis=2)
        output_shape = (100, 100)

        result = process_light(
            str(tmp_path / "light.fits"),
            master_flat,
            wcs,
            output_shape,
            150.0,
            45.0,
        )

        # Should return None on failure
        assert result is None

    @patch("pesto_reduction.create_stack.fits")
    def test_process_light_exception_handling(self, mock_fits, tmp_path):
        """Test exception handling in process_light."""
        mock_fits.getdata.side_effect = Exception("Read error")

        master_flat = np.ones((100, 100), dtype=np.float32)
        wcs = WCS(naxis=2)
        output_shape = (100, 100)

        result = process_light(
            str(tmp_path / "light.fits"),
            master_flat,
            wcs,
            output_shape,
            150.0,
            45.0,
        )

        # Should return None on exception
        assert result is None

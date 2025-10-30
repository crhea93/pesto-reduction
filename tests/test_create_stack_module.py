"""
Tests for the create_stack module (main functionality, not CLI).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

from pesto_reduction.create_stack import build_global_wcs


class TestBuildGlobalWCSComprehensive:
    """Comprehensive tests for WCS grid creation."""

    def test_build_global_wcs_output_size(self):
        """Test that output WCS creates appropriate size."""
        ra, dec = 150.0, 45.0
        pixel_scale = 1.0
        field_size = 10.0

        wcs, shape = build_global_wcs(ra, dec, pixel_scale, field_size)

        # Expected size: 10 arcmin = 600 arcsec / 1 arcsec/pixel ≈ 600
        expected_size = int(np.ceil(600))
        assert shape[0] == expected_size
        assert shape[1] == expected_size

    def test_build_global_wcs_crpix_centered(self):
        """Test that CRPIX is centered in output."""
        ra, dec = 100.0, 0.0
        wcs, (ny, nx) = build_global_wcs(ra, dec)

        # CRPIX should be approximately at center
        assert np.isclose(wcs.wcs.crpix[0], nx / 2, rtol=0.1)
        assert np.isclose(wcs.wcs.crpix[1], ny / 2, rtol=0.1)

    def test_build_global_wcs_cdelt_sign(self):
        """Test that CDELT signs are correct."""
        wcs, _ = build_global_wcs(150.0, 45.0)

        # CD matrix should have correct signs for standard orientation
        assert wcs.wcs.cd[0, 0] < 0  # RA decreases with X
        assert wcs.wcs.cd[1, 1] > 0  # Dec increases with Y

    def test_build_global_wcs_small_field(self):
        """Test with small field size."""
        wcs, shape = build_global_wcs(
            150.0, 45.0, pixel_scale=1.0, field_size_arcmin=1.0
        )

        # 1 arcmin = 60 arcsec / 1 arcsec/pixel = 60
        expected_size = int(np.ceil(60))
        assert shape[0] == expected_size
        assert shape[1] == expected_size

    def test_build_global_wcs_fine_pixel_scale(self):
        """Test with fine pixel scale."""
        wcs_coarse, shape_coarse = build_global_wcs(
            150.0, 45.0, pixel_scale=2.0, field_size_arcmin=10.0
        )
        wcs_fine, shape_fine = build_global_wcs(
            150.0, 45.0, pixel_scale=0.5, field_size_arcmin=10.0
        )

        # Fine pixel scale should produce larger image
        assert shape_fine[0] > shape_coarse[0]
        assert shape_fine[1] > shape_coarse[1]

    def test_build_global_wcs_high_declination(self):
        """Test at high declination where cos(dec) matters."""
        ra, dec = 150.0, 80.0  # High declination
        wcs, shape = build_global_wcs(ra, dec)

        assert shape[0] > 0
        assert shape[1] > 0

    def test_build_global_wcs_negative_declination(self):
        """Test with negative declination."""
        ra, dec = 150.0, -45.0
        wcs, shape = build_global_wcs(ra, dec)

        assert shape[0] > 0
        assert shape[1] > 0
        assert wcs.wcs.crval[1] == -45.0


class TestBuildGlobalWCSEdgeCases:
    """Test edge cases for WCS creation."""

    def test_build_global_wcs_equator(self):
        """Test at equator where cos(dec) = 1."""
        ra, dec = 150.0, 0.0
        wcs, shape = build_global_wcs(ra, dec)

        assert shape[0] > 0
        assert shape[1] > 0

    def test_build_global_wcs_different_cdelt(self):
        """Test with different CD matrix values."""
        wcs1, _ = build_global_wcs(150.0, 45.0, pixel_scale=0.1)
        wcs2, _ = build_global_wcs(150.0, 45.0, pixel_scale=2.0)

        # Different pixel scales should produce different CD matrices
        cd1_scale = np.abs(wcs1.wcs.cd[0, 0])
        cd2_scale = np.abs(wcs2.wcs.cd[0, 0])
        assert cd1_scale < cd2_scale

    def test_build_global_wcs_zero_field_size(self):
        """Test with very small field size."""
        wcs, shape = build_global_wcs(
            150.0, 45.0, pixel_scale=1.0, field_size_arcmin=0.1
        )

        # Should still produce valid output
        assert shape[0] > 0
        assert shape[1] > 0

    def test_build_global_wcs_wide_declination_range(self):
        """Test with various declinations."""
        for dec in [-89, -45, 0, 45, 89]:
            wcs, shape = build_global_wcs(150.0, dec)
            assert shape[0] > 0
            assert shape[1] > 0
            assert wcs.wcs.crval[1] == dec

    def test_build_global_wcs_center_values(self):
        """Test that center coordinates are properly set."""
        ra, dec = 200.0, -30.0
        wcs, shape = build_global_wcs(ra, dec)

        assert wcs.wcs.crval[0] == ra
        assert wcs.wcs.crval[1] == dec

    def test_build_global_wcs_ctype_values(self):
        """Test coordinate type values."""
        wcs, _ = build_global_wcs(150.0, 45.0)

        assert wcs.wcs.ctype[0] == "RA---TAN"
        assert wcs.wcs.ctype[1] == "DEC--TAN"

    def test_build_global_wcs_large_field(self):
        """Test with large field size."""
        wcs, shape = build_global_wcs(
            150.0, 45.0, pixel_scale=1.0, field_size_arcmin=60.0
        )

        # 60 arcmin = 3600 arcsec / 1 arcsec/pixel = 3600
        expected_size = int(np.ceil(3600))
        assert shape[0] == expected_size
        assert shape[1] == expected_size

    def test_build_global_wcs_coarse_pixel_scale(self):
        """Test with coarse pixel scale."""
        wcs, shape = build_global_wcs(
            150.0, 45.0, pixel_scale=10.0, field_size_arcmin=10.0
        )

        # 10 arcmin = 600 arcsec / 10 arcsec/pixel = 60
        expected_size = int(np.ceil(60))
        assert shape[0] == expected_size
        assert shape[1] == expected_size

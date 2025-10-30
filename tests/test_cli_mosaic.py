"""
Tests for the mosaic CLI module.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import os
from pesto_reduction.cli.mosaic import parse_args, main


class TestParseArgsMosaic:
    """Test argument parsing for mosaic CLI."""

    def test_parse_args_required_only(self):
        """Test parsing with only required arguments."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.filter == "Ha"
        assert args.name == "Arp94"
        assert args.output_dir == "/output"
        assert args.n_pos == 6  # default
        assert args.combine == "median"  # default

    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments specified."""
        argv = [
            "mosaic",
            "--filter",
            "i",
            "--name",
            "M51",
            "--output_dir",
            "/output",
            "--n_pos",
            "8",
            "--combine",
            "mean",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.filter == "i"
        assert args.name == "M51"
        assert args.output_dir == "/output"
        assert args.n_pos == 8
        assert args.combine == "mean"

    def test_parse_args_combine_median(self):
        """Test parsing with median combine method."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--combine",
            "median",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.combine == "median"

    def test_parse_args_combine_mean(self):
        """Test parsing with mean combine method."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--combine",
            "mean",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.combine == "mean"

    def test_parse_args_invalid_combine_method(self):
        """Test parsing with invalid combine method."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--combine",
            "invalid",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_missing_filter(self):
        """Test parsing fails without filter."""
        argv = ["mosaic", "--name", "test", "--output_dir", "/output"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_missing_name(self):
        """Test parsing fails without target name."""
        argv = ["mosaic", "--filter", "Ha", "--output_dir", "/output"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_missing_output_dir(self):
        """Test parsing fails without output directory."""
        argv = ["mosaic", "--filter", "Ha", "--name", "test"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_invalid_n_pos(self):
        """Test parsing with invalid n_pos."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "invalid",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_different_filters(self):
        """Test parsing with various filter names."""
        for filter_name in ["Ha", "OIII", "i", "r", "g", "H_alpha"]:
            argv = [
                "mosaic",
                "--filter",
                filter_name,
                "--name",
                "test",
                "--output_dir",
                "/output",
            ]
            with patch("sys.argv", argv):
                args = parse_args()
            assert args.filter == filter_name


class TestMainMosaic:
    """Test main function for mosaic CLI."""

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_all_images_present(self, mock_join, mock_exists, mock_mosaic):
        """Test main when all images are present."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
            "--n_pos",
            "3",
        ]
        with patch("sys.argv", argv):
            main()

        # Should call mosaic_images with all 3 images
        assert mock_mosaic.call_count == 1
        call_args = mock_mosaic.call_args[0]
        assert len(call_args[0]) == 3  # 3 image paths

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    @patch("builtins.print")
    def test_main_some_images_missing(self, mock_print, mock_join, mock_mosaic):
        """Test main when some images are missing."""

        # Use a different approach - check the exists calls
        def exists_side_effect(path):
            # First and third exist, second doesn't
            if "pos1" in path or "pos3" in path:
                return True
            elif "pos2" in path:
                return False
            return True

        with patch("os.path.exists", side_effect=exists_side_effect):
            argv = [
                "mosaic",
                "--filter",
                "Ha",
                "--name",
                "test",
                "--output_dir",
                "/output",
                "--n_pos",
                "3",
            ]
            with patch("sys.argv", argv):
                main()

        # Should call mosaic_images with only 2 images
        call_args = mock_mosaic.call_args[0]
        assert len(call_args[0]) == 2

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=False)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    @patch("builtins.print")
    def test_main_no_images_found(
        self, mock_print, mock_join, mock_exists, mock_mosaic
    ):
        """Test main when no images are found."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "3",
        ]
        with patch("sys.argv", argv):
            main()

        # Should not call mosaic_images
        mock_mosaic.assert_not_called()

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_correct_image_paths(self, mock_join, mock_exists, mock_mosaic):
        """Test main constructs correct image paths."""
        argv = [
            "mosaic",
            "--filter",
            "i",
            "--name",
            "M51",
            "--output_dir",
            "/data",
            "--n_pos",
            "2",
        ]
        with patch("sys.argv", argv):
            main()

        call_args = mock_mosaic.call_args[0]
        paths = call_args[0]

        # Paths should be formatted as name_posN_filter.fits
        assert "M51_pos1_i.fits" in paths[0]
        assert "M51_pos2_i.fits" in paths[1]

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_output_path(self, mock_join, mock_exists, mock_mosaic):
        """Test main sets correct output path."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
            "--n_pos",
            "1",
        ]
        with patch("sys.argv", argv):
            main()

        call_args = mock_mosaic.call_args[0]
        output_path = call_args[1]

        # Output should be name_filter.fits
        assert "Arp94_Ha.fits" in output_path

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_median_combine(self, mock_join, mock_exists, mock_mosaic):
        """Test main with median combine method."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "1",
            "--combine",
            "median",
        ]
        with patch("sys.argv", argv):
            main()

        call_kwargs = mock_mosaic.call_args[1]
        assert call_kwargs["combine"] == "median"

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_mean_combine(self, mock_join, mock_exists, mock_mosaic):
        """Test main with mean combine method."""
        argv = [
            "mosaic",
            "--filter",
            "i",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "1",
            "--combine",
            "mean",
        ]
        with patch("sys.argv", argv):
            main()

        call_kwargs = mock_mosaic.call_args[1]
        assert call_kwargs["combine"] == "mean"

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_zero_positions(self, mock_join, mock_exists, mock_mosaic):
        """Test main with zero positions."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "0",
        ]
        with patch("sys.argv", argv):
            main()

        # Should not call mosaic_images
        mock_mosaic.assert_not_called()

    @patch("pesto_reduction.cli.mosaic.mosaic_images")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def test_main_large_position_count(self, mock_join, mock_exists, mock_mosaic):
        """Test main with large number of positions."""
        argv = [
            "mosaic",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--n_pos",
            "20",
        ]
        with patch("sys.argv", argv):
            main()

        call_args = mock_mosaic.call_args[0]
        # Should have 20 image paths
        assert len(call_args[0]) == 20

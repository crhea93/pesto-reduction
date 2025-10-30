"""
Tests for the create_flat CLI module.
"""

import pytest
from unittest.mock import patch
from pesto_reduction.cli.create_flat import parse_args, main


class TestParseArgsCreateFlat:
    """Test argument parsing for create_flat CLI."""

    def test_parse_args_required_only(self):
        """Test parsing with only required arguments."""
        argv = ["--data_dir", "/path/to/flats", "--output_path", "/path/to/output"]
        args = parse_args(argv)

        assert args.data_dir == "/path/to/flats"
        assert args.output_path == "/path/to/output"
        assert args.bias_level == 300
        assert args.no_plot is False
        assert args.crop_size == 500

    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments specified."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--bias_level",
            "400",
            "--crop_size",
            "600",
            "--no_plot",
        ]
        args = parse_args(argv)

        assert args.data_dir == "/flats"
        assert args.output_path == "/output"
        assert args.bias_level == 400
        assert args.crop_size == 600
        assert args.no_plot is True

    def test_parse_args_custom_bias_level(self):
        """Test parsing with custom bias level."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--bias_level",
            "250",
        ]
        args = parse_args(argv)

        assert args.bias_level == 250

    def test_parse_args_custom_crop_size(self):
        """Test parsing with custom crop size."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--crop_size",
            "300",
        ]
        args = parse_args(argv)

        assert args.crop_size == 300

    def test_parse_args_missing_required(self):
        """Test parsing fails with missing required arguments."""
        with pytest.raises(SystemExit):
            parse_args(["--data_dir", "/flats"])

    def test_parse_args_invalid_bias_level(self):
        """Test parsing with invalid bias level."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--bias_level",
            "invalid",
        ]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_parse_args_invalid_crop_size(self):
        """Test parsing with invalid crop size."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--crop_size",
            "not_a_number",
        ]
        with pytest.raises(SystemExit):
            parse_args(argv)

    def test_parse_args_no_plot_flag(self):
        """Test no_plot flag behavior."""
        # With flag
        args1 = parse_args(
            ["--data_dir", "/flats", "--output_path", "/output", "--no_plot"]
        )
        assert args1.no_plot is True

        # Without flag
        args2 = parse_args(["--data_dir", "/flats", "--output_path", "/output"])
        assert args2.no_plot is False


class TestMainCreateFlat:
    """Test main function for create_flat CLI."""

    @patch("pesto_reduction.cli.create_flat.create_master_flat")
    def test_main_calls_create_master_flat(self, mock_create_flat):
        """Test that main calls create_master_flat with correct arguments."""
        argv = [
            "--data_dir",
            "/flats",
            "--output_path",
            "/output",
            "--bias_level",
            "300",
            "--crop_size",
            "500",
        ]
        main(argv)

        mock_create_flat.assert_called_once_with(
            data_dir="/flats",
            output_path="/output",
            bias_level=300,
            center_crop_halfsize=500,
            plot_hist=True,
        )

    @patch("pesto_reduction.cli.create_flat.create_master_flat")
    def test_main_no_plot_flag(self, mock_create_flat):
        """Test main with no_plot flag disables histogram."""
        argv = ["--data_dir", "/flats", "--output_path", "/output", "--no_plot"]
        main(argv)

        mock_create_flat.assert_called_once()
        call_kwargs = mock_create_flat.call_args[1]
        assert call_kwargs["plot_hist"] is False

    @patch("pesto_reduction.cli.create_flat.create_master_flat")
    def test_main_custom_parameters(self, mock_create_flat):
        """Test main with custom parameters."""
        argv = [
            "--data_dir",
            "/custom/flats",
            "--output_path",
            "/custom/output",
            "--bias_level",
            "400",
            "--crop_size",
            "750",
            "--no_plot",
        ]
        main(argv)

        mock_create_flat.assert_called_once_with(
            data_dir="/custom/flats",
            output_path="/custom/output",
            bias_level=400,
            center_crop_halfsize=750,
            plot_hist=False,
        )

    @patch("pesto_reduction.cli.create_flat.create_master_flat")
    def test_main_exception_handling(self, mock_create_flat):
        """Test main handles exceptions from create_master_flat."""
        mock_create_flat.side_effect = Exception("Processing error")
        argv = ["--data_dir", "/flats", "--output_path", "/output"]

        # Should raise the exception
        with pytest.raises(Exception, match="Processing error"):
            main(argv)

    @patch("pesto_reduction.cli.create_flat.create_master_flat")
    def test_main_default_arguments(self, mock_create_flat):
        """Test main uses default values for optional arguments."""
        argv = ["--data_dir", "/flats", "--output_path", "/output"]
        main(argv)

        mock_create_flat.assert_called_once_with(
            data_dir="/flats",
            output_path="/output",
            bias_level=300,  # default
            center_crop_halfsize=500,  # default
            plot_hist=True,  # default (no_plot=False)
        )

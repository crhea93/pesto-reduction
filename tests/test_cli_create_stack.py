"""
Tests for the create_stack CLI module.
"""

import pytest
from unittest.mock import patch, MagicMock, call
import os
from pesto_reduction.cli.create_stack import parse_args, main


class TestParseArgsCreateStack:
    """Test argument parsing for create_stack CLI."""

    def test_parse_args_required_only(self):
        """Test parsing with only required arguments."""
        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.filter == "Ha"
        assert args.name == "Arp94"
        assert args.output_dir == "/output"
        assert args.data_root == "/data"
        assert args.n_pos == 6  # default

    def test_parse_args_all_arguments(self):
        """Test parsing with all arguments specified."""
        argv = [
            "create_stack",
            "--filter",
            "i",
            "--name",
            "M51",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "8",
        ]
        with patch("sys.argv", argv):
            args = parse_args()

        assert args.filter == "i"
        assert args.name == "M51"
        assert args.output_dir == "/output"
        assert args.data_root == "/data"
        assert args.n_pos == 8

    def test_parse_args_different_filters(self):
        """Test parsing with different filter names."""
        for filter_name in ["Ha", "OIII", "i", "r", "g"]:
            argv = [
                "create_stack",
                "--filter",
                filter_name,
                "--name",
                "test",
                "--output_dir",
                "/output",
                "--data_root",
                "/data",
            ]
            with patch("sys.argv", argv):
                args = parse_args()
            assert args.filter == filter_name

    def test_parse_args_missing_required(self):
        """Test parsing fails with missing required arguments."""
        argv = ["create_stack", "--filter", "Ha"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_invalid_n_pos(self):
        """Test parsing with invalid n_pos."""
        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "invalid",
        ]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_zero_positions(self):
        """Test parsing with zero positions."""
        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "0",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        assert args.n_pos == 0

    def test_parse_args_large_n_pos(self):
        """Test parsing with large n_pos value."""
        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "100",
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        assert args.n_pos == 100


class TestMainCreateStack:
    """Test main function for create_stack CLI."""

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.exists")
    @patch("os.path.isdir")
    def test_main_missing_master_flat(self, mock_isdir, mock_exists, mock_create_stack):
        """Test main exits when master flat is missing."""
        mock_exists.return_value = False

        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "1",
        ]
        with patch("sys.argv", argv):
            main()

        mock_create_stack.assert_not_called()

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    def test_main_single_position(self, mock_isdir, mock_create_stack):
        """Test main with single position."""
        mock_isdir.return_value = True

        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "1",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.exists", return_value=True):
                    main()

        assert mock_create_stack.call_count == 1

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    def test_main_multiple_positions(self, mock_isdir, mock_create_stack):
        """Test main with multiple positions."""
        mock_isdir.return_value = True

        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "3",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.exists", return_value=True):
                    main()

        assert mock_create_stack.call_count == 3

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    @patch("builtins.print")
    def test_main_skips_missing_directories(
        self, mock_print, mock_isdir, mock_create_stack
    ):
        """Test main skips directories that don't exist."""
        # First position doesn't exist, second and third do
        mock_isdir.side_effect = [False, True, True]

        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "3",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.exists", return_value=True):
                    main()

        # Should only process the existing directories
        assert mock_create_stack.call_count == 2

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    def test_main_passes_correct_parameters(self, mock_isdir, mock_create_stack):
        """Test main passes correct parameters to create_stack."""
        mock_isdir.return_value = True

        argv = [
            "create_stack",
            "--filter",
            "i",
            "--name",
            "M51",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "1",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.exists", return_value=True):
                    main()

        # Check the arguments passed to create_stack
        call_kwargs = mock_create_stack.call_args[1]
        assert call_kwargs["filter_"] == "i"
        assert call_kwargs["pos"] == "M51_pos1"
        assert call_kwargs["output_dir"] == "/output"

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    def test_main_position_naming(self, mock_isdir, mock_create_stack):
        """Test that position names are correctly formatted."""
        mock_isdir.return_value = True

        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "Arp94",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "3",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
                with patch("os.path.exists", return_value=True):
                    main()

        # Check position naming (1-indexed)
        calls = mock_create_stack.call_args_list
        assert calls[0][1]["pos"] == "Arp94_pos1"
        assert calls[1][1]["pos"] == "Arp94_pos2"
        assert calls[2][1]["pos"] == "Arp94_pos3"

    @patch("pesto_reduction.cli.create_stack.create_stack")
    @patch("os.path.isdir")
    def test_main_zero_positions(self, mock_isdir, mock_create_stack):
        """Test main with zero positions."""
        argv = [
            "create_stack",
            "--filter",
            "Ha",
            "--name",
            "test",
            "--output_dir",
            "/output",
            "--data_root",
            "/data",
            "--n_pos",
            "0",
        ]
        with patch("sys.argv", argv):
            with patch("os.path.exists", return_value=True):
                main()

        mock_create_stack.assert_not_called()

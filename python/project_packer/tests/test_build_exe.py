"""Tests for build_exe.py module."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module to test
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_exe import (
    build_executable,
    check_pyinstaller,
    clean_build_dirs,
    install_pyinstaller,
    main,
    verify_build,
)


class TestBuildExe:
    """Test cases for build_exe.py module."""

    def test_check_pyinstaller_available(self) -> None:
        """Test PyInstaller availability check when available."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = Mock()
            assert check_pyinstaller() is True

    def test_check_pyinstaller_not_available(self) -> None:
        """Test PyInstaller availability check when not available."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None
            assert check_pyinstaller() is False

    def test_install_pyinstaller_success(self) -> None:
        """Test successful PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert install_pyinstaller() is True
            mock_run.assert_called_once()

    def test_install_pyinstaller_failure(self) -> None:
        """Test failed PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
            assert install_pyinstaller() is False

    def test_clean_build_dirs_existing(self, tmp_path: Path) -> None:
        """Test cleaning existing build directories."""
        # Create actual build and dist directories
        build_dir = tmp_path / "build"
        dist_dir = tmp_path / "dist"
        build_dir.mkdir()
        dist_dir.mkdir()

        with patch("build_exe.Path") as mock_path:
            mock_path.return_value = tmp_path
            with patch("build_exe.shutil.rmtree") as mock_rmtree:
                clean_build_dirs()
                assert mock_rmtree.call_count == 2

    def test_clean_build_dirs_nonexistent(self, tmp_path: Path) -> None:
        """Test cleaning non-existent build directories."""
        # Don't mock Path here, let it use the real implementation
        # but ensure the directories don't exist
        with patch("build_exe.shutil.rmtree") as mock_rmtree:
            clean_build_dirs()
            mock_rmtree.assert_not_called()

    def test_build_executable_script_exists(self) -> None:
        """Test building executable when script exists."""
        with patch("build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("build_exe.clean_build_dirs") as mock_clean:
                with patch("build_exe.subprocess.run") as mock_run:
                    mock_run.return_value = Mock(returncode=0)
                    assert build_executable() is True

    def test_build_executable_script_not_exists(self) -> None:
        """Test building executable when script doesn't exist."""
        with patch("build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            assert build_executable() is False

    def test_build_executable_subprocess_failure(self) -> None:
        """Test building executable when subprocess fails."""
        with patch("build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("build_exe.clean_build_dirs") as mock_clean:
                with patch("build_exe.subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.CalledProcessError(
                        1, "pyinstaller",
                    )
                    assert build_executable() is False

    def test_verify_build_success(self, tmp_path: Path) -> None:
        """Test build verification when executable exists."""
        exe_path = tmp_path / "dist" / "FolderPacker.exe"
        exe_path.parent.mkdir(parents=True)
        exe_path.write_bytes(b"fake executable")

        with patch("build_exe.Path") as mock_path:
            mock_path.return_value = exe_path
            assert verify_build() is True

    def test_verify_build_executable_not_found(self) -> None:
        """Test build verification when executable doesn't exist."""
        with patch("build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            assert verify_build() is False

    def test_main_pyinstaller_available(self) -> None:
        """Test main function when PyInstaller is available."""
        with patch("build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("build_exe.build_executable") as mock_build:
                mock_build.return_value = True
                with patch("build_exe.verify_build") as mock_verify:
                    mock_verify.return_value = True
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_not_called()

    def test_main_pyinstaller_not_available_install_success(self) -> None:
        """Test main function when PyInstaller is not available but installs successfully."""
        with patch("build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = False
            with patch("build_exe.install_pyinstaller") as mock_install:
                mock_install.return_value = True
                with patch("build_exe.build_executable") as mock_build:
                    mock_build.return_value = True
                    with patch("build_exe.verify_build") as mock_verify:
                        mock_verify.return_value = True
                        with patch("sys.exit") as mock_exit:
                            main()
                            mock_exit.assert_not_called()

    def test_main_pyinstaller_not_available_install_failure(self) -> None:
        """Test main function when PyInstaller is not available and install fails."""
        with patch("build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = False
            with patch("build_exe.install_pyinstaller") as mock_install:
                mock_install.return_value = False
                with patch("sys.exit") as mock_exit:
                    main()
                    mock_exit.assert_called_once_with(1)

    def test_main_build_failure(self) -> None:
        """Test main function when build fails."""
        with patch("build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("build_exe.build_executable") as mock_build:
                mock_build.return_value = False
                with patch("sys.exit") as mock_exit:
                    main()
                    mock_exit.assert_called_once_with(1)

    def test_main_verification_failure(self) -> None:
        """Test main function when verification fails."""
        with patch("build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("build_exe.build_executable") as mock_build:
                mock_build.return_value = True
                with patch("build_exe.verify_build") as mock_verify:
                    mock_verify.return_value = False
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__])

# ruff: noqa: E501
"""Tests for project_packer.build_exe.py module."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from project_packer.build_exe import (
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
        with patch("project_packer.build_exe.find_spec") as mock_find_spec:
            mock_find_spec.return_value = Mock()
            assert check_pyinstaller() is True

    def test_check_pyinstaller_not_available(self) -> None:
        """Test PyInstaller availability check when not available."""
        with patch("project_packer.build_exe.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None
            assert check_pyinstaller() is False

    def test_install_pyinstaller_success(self) -> None:
        """Test successful PyInstaller installation."""
        with patch("project_packer.build_exe.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert install_pyinstaller() is True

    def test_install_pyinstaller_failure(self) -> None:
        """Test failed PyInstaller installation."""
        with patch("project_packer.build_exe.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
            assert install_pyinstaller() is False

    def test_clean_build_dirs_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test cleaning existing build directories."""
        build_dir = tmp_path / "build"
        dist_dir = tmp_path / "dist"
        build_dir.mkdir()
        dist_dir.mkdir()
        monkeypatch.setattr("project_packer.build_exe.BUILD_DIR", str(build_dir))
        monkeypatch.setattr("project_packer.build_exe.DIST_DIR", str(dist_dir))

        clean_build_dirs()

        assert not build_dir.exists()
        assert not dist_dir.exists()

    def test_clean_build_dirs_nonexistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test cleaning non-existent build directories."""
        monkeypatch.setattr(
            "project_packer.build_exe.BUILD_DIR", str(tmp_path / "build")
        )
        monkeypatch.setattr("project_packer.build_exe.DIST_DIR", str(tmp_path / "dist"))

        with patch("project_packer.build_exe.shutil.rmtree") as mock_rmtree:
            clean_build_dirs()
            mock_rmtree.assert_not_called()

    def test_build_executable_script_exists(self) -> None:
        """Test building executable when script exists."""
        with patch("project_packer.build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with (
                patch("project_packer.build_exe.clean_build_dirs"),
                patch("project_packer.build_exe.subprocess.run") as mock_run,
            ):
                mock_run.return_value = Mock(returncode=0)
                assert build_executable() is True

    def test_build_executable_script_not_exists(self) -> None:
        """Test building executable when script doesn't exist."""
        with patch("project_packer.build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            assert build_executable() is False

    def test_build_executable_subprocess_failure(self) -> None:
        """Test building executable when subprocess fails."""
        with patch("project_packer.build_exe.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with (
                patch("project_packer.build_exe.clean_build_dirs"),
                patch("project_packer.build_exe.subprocess.run") as mock_run,
            ):
                mock_run.side_effect = subprocess.CalledProcessError(
                    1,
                    "pyinstaller",
                )
                assert build_executable() is False

    def test_verify_build_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test build verification when executable exists."""
        exe_path = tmp_path / "dist" / "FolderPacker.exe"
        exe_path.parent.mkdir(parents=True)
        exe_path.write_bytes(b"fake executable")
        monkeypatch.setattr("project_packer.build_exe.DIST_DIR", str(tmp_path / "dist"))

        assert verify_build() is True

    def test_verify_build_executable_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test build verification when executable doesn't exist."""
        monkeypatch.setattr("project_packer.build_exe.DIST_DIR", str(tmp_path / "dist"))

        assert verify_build() is False

    def test_main_pyinstaller_available(self) -> None:
        """Test main function when PyInstaller is available."""
        with patch("project_packer.build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("project_packer.build_exe.build_executable") as mock_build:
                mock_build.return_value = True
                with patch("project_packer.build_exe.verify_build") as mock_verify:
                    mock_verify.return_value = True
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_not_called()

    def test_main_pyinstaller_not_available_install_success(self) -> None:
        """Test main function when PyInstaller is not available but installs successfully."""
        with patch("project_packer.build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = False
            with patch("project_packer.build_exe.install_pyinstaller") as mock_install:
                mock_install.return_value = True
                with patch("project_packer.build_exe.build_executable") as mock_build:
                    mock_build.return_value = True
                    with patch("project_packer.build_exe.verify_build") as mock_verify:
                        mock_verify.return_value = True
                        with patch("sys.exit") as mock_exit:
                            main()
                            mock_exit.assert_not_called()

    def test_main_pyinstaller_not_available_install_failure(self) -> None:
        """Test main function when PyInstaller is not available and install fails."""
        with patch("project_packer.build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = False
            with patch("project_packer.build_exe.install_pyinstaller") as mock_install:
                mock_install.return_value = False
                with (
                    patch("sys.exit", side_effect=SystemExit(1)) as mock_exit,
                    pytest.raises(SystemExit),
                ):
                    main()
                    mock_exit.assert_called_once_with(1)

    def test_main_build_failure(self) -> None:
        """Test main function when build fails."""
        with patch("project_packer.build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("project_packer.build_exe.build_executable") as mock_build:
                mock_build.return_value = False
                with patch("sys.exit") as mock_exit:
                    main()
                    mock_exit.assert_called_once_with(1)

    def test_main_verification_failure(self) -> None:
        """Test main function when verification fails."""
        with patch("project_packer.build_exe.check_pyinstaller") as mock_check:
            mock_check.return_value = True
            with patch("project_packer.build_exe.build_executable") as mock_build:
                mock_build.return_value = True
                with patch("project_packer.build_exe.verify_build") as mock_verify:
                    mock_verify.return_value = False
                    with patch("sys.exit") as mock_exit:
                        main()
                        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__])

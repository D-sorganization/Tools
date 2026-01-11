"""Tests for build_exe.py - build system testing."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_exe import (
    BUILD_DIR,
    DIST_DIR,
    EXE_NAME,
    SCRIPT_NAME,
    SPEC_FILE,
    build_executable,
    check_pyinstaller,
    clean_build_dirs,
    install_pyinstaller,
    main,
    verify_build,
)


class TestCheckPyInstaller:
    """Test PyInstaller availability checking."""

    def test_pyinstaller_available(self) -> None:
        """Test when PyInstaller is available."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert check_pyinstaller() is True

    def test_pyinstaller_not_available(self) -> None:
        """Test when PyInstaller is not available."""
        with patch("importlib.util.find_spec", return_value=None):
            assert check_pyinstaller() is False


class TestInstallPyInstaller:
    """Test PyInstaller installation."""

    def test_successful_installation(self) -> None:
        """Test successful PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = install_pyinstaller()
            assert result is True
            mock_run.assert_called_once()
            # Verify correct command
            args = mock_run.call_args[0][0]
            assert args[0] == sys.executable
            assert "-m" in args
            assert "pip" in args
            assert "install" in args
            assert "pyinstaller" in args

    def test_failed_installation(self) -> None:
        """Test failed PyInstaller installation."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "pip")
            result = install_pyinstaller()
            assert result is False

    def test_installation_uses_check_true(self) -> None:
        """Test that installation uses check=True for error handling."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            install_pyinstaller()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["check"] is True


class TestCleanBuildDirs:
    """Test build directory cleaning."""

    def test_clean_existing_directories(self, tmp_path: Path) -> None:
        """Test cleaning when directories exist."""
        # Simply test that the function calls rmtree with correct directories
        with patch("shutil.rmtree") as mock_rmtree:
            with patch("pathlib.Path.exists", return_value=True):
                clean_build_dirs()
                # Should call rmtree twice (build and dist)
                assert mock_rmtree.call_count == 2

    def test_clean_nonexistent_directories(self, tmp_path: Path) -> None:
        """Test cleaning when directories don't exist."""
        with patch.object(Path, "exists", return_value=False):
            with patch("shutil.rmtree") as mock_rmtree:
                clean_build_dirs()
                # Should not call rmtree
                mock_rmtree.assert_not_called()


class TestBuildExecutable:
    """Test executable building."""

    def test_successful_build(self, tmp_path: Path) -> None:
        """Test successful executable build."""
        # Create fake script file
        script_file = tmp_path / SCRIPT_NAME
        script_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(Path, "exists", return_value=True):
                with patch("build_exe.clean_build_dirs"):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0)
                        result = build_executable()
                        assert result is True
                        mock_run.assert_called_once()

    def test_build_script_not_found(self, tmp_path: Path) -> None:
        """Test build when script file doesn't exist."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(Path, "exists", return_value=False):
                result = build_executable()
                assert result is False

    def test_build_command_correct_format(self, tmp_path: Path) -> None:
        """Test that build command has correct format."""
        script_file = tmp_path / SCRIPT_NAME
        script_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(Path, "exists", return_value=True):
                with patch("build_exe.clean_build_dirs"):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0)
                        build_executable()

                        # Verify command structure
                        args = mock_run.call_args[0][0]
                        assert args[0] == sys.executable
                        assert "-m" in args
                        assert "PyInstaller" in args
                        assert "--onefile" in args
                        assert "--windowed" in args
                        assert "--name" in args
                        assert EXE_NAME in args

    def test_build_subprocess_failure(self, tmp_path: Path) -> None:
        """Test build when subprocess fails."""
        script_file = tmp_path / SCRIPT_NAME
        script_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(Path, "exists", return_value=True):
                with patch("build_exe.clean_build_dirs"):
                    with patch("subprocess.run") as mock_run:
                        mock_run.side_effect = subprocess.CalledProcessError(1, "pyinstaller")
                        result = build_executable()
                        assert result is False

    def test_build_cleans_before_building(self, tmp_path: Path) -> None:
        """Test that build cleans directories before building."""
        script_file = tmp_path / SCRIPT_NAME
        script_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            with patch.object(Path, "exists", return_value=True):
                with patch("build_exe.clean_build_dirs") as mock_clean:
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0)
                        build_executable()
                        mock_clean.assert_called_once()


class TestVerifyBuild:
    """Test build verification."""

    def test_verify_successful_build(self, tmp_path: Path) -> None:
        """Test verification when executable exists."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=1024 * 1024)
                result = verify_build()
                assert result is True

    def test_verify_missing_executable(self, tmp_path: Path) -> None:
        """Test verification when executable doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            result = verify_build()
            assert result is False

    def test_verify_logs_file_size(self, tmp_path: Path) -> None:
        """Test that verification logs file size."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=5 * 1024 * 1024)  # 5 MB
                with patch("build_exe.logger") as mock_logger:
                    verify_build()
                    # Should log file size
                    assert any(
                        "size" in str(call).lower() for call in mock_logger.info.call_args_list
                    )


class TestMain:
    """Test main build process."""

    def test_main_success_path(self) -> None:
        """Test successful main execution."""
        with patch("build_exe.check_pyinstaller", return_value=True):
            with patch("build_exe.build_executable", return_value=True):
                with patch("build_exe.verify_build", return_value=True):
                    # Should not raise or call sys.exit
                    main()

    def test_main_installs_pyinstaller_if_missing(self) -> None:
        """Test that main installs PyInstaller if not found."""
        with patch("build_exe.check_pyinstaller", return_value=False):
            with patch("build_exe.install_pyinstaller", return_value=True) as mock_install:
                with patch("build_exe.build_executable", return_value=True):
                    with patch("build_exe.verify_build", return_value=True):
                        main()
                        mock_install.assert_called_once()

    def test_main_exits_if_install_fails(self) -> None:
        """Test that main exits if PyInstaller installation fails."""
        with patch("build_exe.check_pyinstaller", return_value=False):
            with patch("build_exe.install_pyinstaller", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_main_exits_if_build_fails(self) -> None:
        """Test that main exits if build fails."""
        with patch("build_exe.check_pyinstaller", return_value=True):
            with patch("build_exe.build_executable", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1

    def test_main_exits_if_verification_fails(self) -> None:
        """Test that main exits if verification fails."""
        with patch("build_exe.check_pyinstaller", return_value=True):
            with patch("build_exe.build_executable", return_value=True):
                with patch("build_exe.verify_build", return_value=False):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1

    def test_main_execution_order(self) -> None:
        """Test that main executes steps in correct order."""
        call_order = []

        def track_check():
            call_order.append("check")
            return True

        def track_build():
            call_order.append("build")
            return True

        def track_verify():
            call_order.append("verify")
            return True

        with patch("build_exe.check_pyinstaller", side_effect=track_check):
            with patch("build_exe.build_executable", side_effect=track_build):
                with patch("build_exe.verify_build", side_effect=track_verify):
                    main()
                    assert call_order == ["check", "build", "verify"]


class TestConstants:
    """Test that constants are properly defined."""

    def test_constants_are_strings(self) -> None:
        """Test that all constants are strings."""
        assert isinstance(SCRIPT_NAME, str)
        assert isinstance(EXE_NAME, str)
        assert isinstance(BUILD_DIR, str)
        assert isinstance(DIST_DIR, str)
        assert isinstance(SPEC_FILE, str)

    def test_constants_not_empty(self) -> None:
        """Test that constants are not empty."""
        assert len(SCRIPT_NAME) > 0
        assert len(EXE_NAME) > 0
        assert len(BUILD_DIR) > 0
        assert len(DIST_DIR) > 0
        assert len(SPEC_FILE) > 0

    def test_script_name_is_python_file(self) -> None:
        """Test that script name ends with .py."""
        assert SCRIPT_NAME.endswith(".py")

    def test_spec_file_is_spec_file(self) -> None:
        """Test that spec file ends with .spec."""
        assert SPEC_FILE.endswith(".spec")

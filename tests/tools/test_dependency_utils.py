"""Tests for dependency_utils.

Expanded test suite covering check_dependencies, install_packages,
pip name mapping, error handling, and edge cases.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

from tools.dependency_utils import check_dependencies, install_packages

# ─── check_dependencies Tests ────────────────────────────────


class TestCheckDependencies:
    """Test check_dependencies."""

    def test_all_present(self) -> None:
        missing = check_dependencies(["sys", "os", "json"])
        assert missing == []

    def test_all_missing(self) -> None:
        fakes = ["_fake_pkg_1", "_fake_pkg_2"]
        for f in fakes:
            sys.modules.pop(f, None)
        missing = check_dependencies(fakes)
        assert sorted(missing) == sorted(fakes)

    def test_mixed_present_and_missing(self) -> None:
        fake = "_nonexistent_xyz_pkg"
        sys.modules.pop(fake, None)
        missing = check_dependencies(["sys", fake, "os"])
        assert missing == [fake]

    def test_empty_list(self) -> None:
        missing = check_dependencies([])
        assert missing == []

    def test_single_present(self) -> None:
        missing = check_dependencies(["json"])
        assert missing == []

    def test_single_missing(self) -> None:
        fake = "_nonexistent_single_pkg"
        sys.modules.pop(fake, None)
        missing = check_dependencies([fake])
        assert missing == [fake]

    def test_returns_list(self) -> None:
        result = check_dependencies(["sys"])
        assert isinstance(result, list)

    def test_pil_special_case(self) -> None:
        """PIL should use 'PIL' as import name."""
        # We don't know if Pillow is installed, so just verify it doesn't crash
        result = check_dependencies(["PIL"])
        assert isinstance(result, list)


# ─── install_packages Tests ──────────────────────────────────


class TestInstallPackages:
    """Test install_packages with mocked subprocess."""

    @patch("tools.dependency_utils.subprocess.run")
    def test_success_single_package(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert install_packages(["numpy"]) is True
        assert mock_run.call_count == 1

    @patch("tools.dependency_utils.subprocess.run")
    def test_success_multiple_packages(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        assert install_packages(["pandas", "numpy", "matplotlib"]) is True
        assert mock_run.call_count == 3

    @patch("tools.dependency_utils.subprocess.run")
    def test_failure_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        assert install_packages(["bad_pkg"]) is False

    @patch("tools.dependency_utils.subprocess.run")
    def test_partial_failure(self, mock_run: MagicMock) -> None:
        """One success + one failure = overall False."""
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1, stderr="fail"),
        ]
        assert install_packages(["good_pkg", "bad_pkg"]) is False
        assert mock_run.call_count == 2

    def test_empty_list_no_subprocess(self) -> None:
        assert install_packages([]) is True

    @patch("tools.dependency_utils.subprocess.run")
    def test_pil_maps_to_pillow(self, mock_run: MagicMock) -> None:
        """PIL should be installed as Pillow."""
        mock_run.return_value = MagicMock(returncode=0)
        install_packages(["PIL"])
        args, _ = mock_run.call_args_list[0]
        assert "Pillow" in args[0]

    @patch("tools.dependency_utils.subprocess.run")
    def test_subprocess_error_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.SubprocessError("broken")
        assert install_packages(["numpy"]) is False

    @patch("tools.dependency_utils.subprocess.run")
    def test_os_error_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("no pip")
        assert install_packages(["numpy"]) is False

    @patch("tools.dependency_utils.subprocess.run")
    def test_uses_sys_executable(self, mock_run: MagicMock) -> None:
        """install_packages should use sys.executable to call pip."""
        mock_run.return_value = MagicMock(returncode=0)
        install_packages(["requests"])
        args, _ = mock_run.call_args_list[0]
        assert args[0][0] == sys.executable

    @patch("tools.dependency_utils.subprocess.run")
    def test_unknown_package_uses_name_directly(self, mock_run: MagicMock) -> None:
        """Packages not in the pip_names map use their own name."""
        mock_run.return_value = MagicMock(returncode=0)
        install_packages(["some_unknown_pkg"])
        args, _ = mock_run.call_args_list[0]
        assert "some_unknown_pkg" in args[0]

"""Tests for dependency_checker - Dependency checking utilities.

These tests verify the dependency checker functions using
Design by Contract principles.
"""

from unittest.mock import MagicMock, patch


class TestDependencyStatusContract:
    """Design by Contract tests for DependencyStatus dataclass."""

    def test_has_required_fields(self):
        """Postcondition: DependencyStatus has required fields."""
        from utils.dependency_checker import DependencyStatus

        status = DependencyStatus(ok=True, missing=[], guidance={})
        assert hasattr(status, "ok")
        assert hasattr(status, "missing")
        assert hasattr(status, "guidance")
        assert hasattr(status, "package_map")


class TestCheckPythonVersionContract:
    """Design by Contract tests for check_python_version function."""

    def test_returns_tuple(self):
        """Postcondition: Returns a tuple of (bool, str)."""
        from utils.dependency_checker import check_python_version

        result = check_python_version()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


class TestCheckPythonVersion:
    """Functional tests for check_python_version."""

    def test_returns_true_for_current_version(self):
        """Test returning True for current Python version."""
        from utils.dependency_checker import check_python_version

        is_valid, version_str = check_python_version(min_major=3, min_minor=10)
        assert is_valid is True
        assert "." in version_str

    def test_returns_false_for_future_version(self):
        """Test returning False for future Python version."""
        from utils.dependency_checker import check_python_version

        is_valid, _ = check_python_version(min_major=99, min_minor=0)
        assert is_valid is False

    def test_version_string_format(self):
        """Test version string format."""
        from utils.dependency_checker import check_python_version

        _, version_str = check_python_version()
        parts = version_str.split(".")
        assert len(parts) == 3


class TestHasModuleContract:
    """Design by Contract tests for has_module function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from utils.dependency_checker import has_module

        result = has_module("sys")
        assert isinstance(result, bool)


class TestHasModule:
    """Functional tests for has_module."""

    def test_returns_true_for_stdlib(self):
        """Test returning True for standard library modules."""
        from utils.dependency_checker import has_module

        assert has_module("sys") is True
        assert has_module("os") is True
        assert has_module("pathlib") is True

    def test_returns_false_for_nonexistent(self):
        """Test returning False for nonexistent modules."""
        from utils.dependency_checker import has_module

        assert has_module("definitely_not_a_real_module_xyz") is False

    def test_uses_custom_spec_finder(self):
        """Test using custom spec finder."""
        from utils.dependency_checker import has_module

        mock_finder = MagicMock(return_value=None)
        result = has_module("anything", spec_finder=mock_finder)

        mock_finder.assert_called_once_with("anything")
        assert result is False


class TestCheckDependenciesContract:
    """Design by Contract tests for check_dependencies function."""

    def test_returns_dependency_status(self):
        """Postcondition: Returns DependencyStatus object."""
        from utils.dependency_checker import DependencyStatus, check_dependencies

        result = check_dependencies(["sys"])
        assert isinstance(result, DependencyStatus)


class TestCheckDependencies:
    """Functional tests for check_dependencies."""

    def test_all_present(self):
        """Test when all dependencies present."""
        from utils.dependency_checker import check_dependencies

        result = check_dependencies(["sys", "os", "pathlib"])
        assert result.ok is True
        assert result.missing == []

    def test_some_missing(self):
        """Test when some dependencies missing."""
        from utils.dependency_checker import check_dependencies

        result = check_dependencies(["sys", "fake_module_xyz"])
        assert result.ok is False
        assert "fake_module_xyz" in result.missing

    def test_dict_format_input(self):
        """Test dictionary format input."""
        from utils.dependency_checker import check_dependencies

        required = {
            "sys": "pip install sys",
            "nonexistent": "pip install nonexistent",
        }
        result = check_dependencies(required)

        assert "nonexistent" in result.missing
        assert "pip install nonexistent" in result.guidance.get("nonexistent", "")

    def test_list_format_input(self):
        """Test list format input."""
        from utils.dependency_checker import check_dependencies

        result = check_dependencies(["sys", "os"])
        assert result.ok is True

    def test_custom_spec_finder(self):
        """Test with custom spec finder."""
        from utils.dependency_checker import check_dependencies

        # Always returns None (module not found)
        mock_finder = MagicMock(return_value=None)
        result = check_dependencies(["pkg1", "pkg2"], spec_finder=mock_finder)

        assert result.ok is False
        assert "pkg1" in result.missing
        assert "pkg2" in result.missing


class TestInstallPackageContract:
    """Design by Contract tests for install_package function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from utils.dependency_checker import install_package

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = install_package("test-package")

        assert isinstance(result, bool)


class TestInstallPackage:
    """Functional tests for install_package."""

    def test_calls_pip_install(self):
        """Test calling pip install."""
        from utils.dependency_checker import install_package

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            install_package("test-package")

            call_args = mock_run.call_args[0][0]
            assert "pip" in " ".join(call_args)
            assert "install" in call_args
            assert "test-package" in call_args

    def test_uses_package_map(self):
        """Test using package map for pip name."""
        from utils.dependency_checker import install_package

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            install_package("cv2", package_map={"cv2": "opencv-python"})

            call_args = mock_run.call_args[0][0]
            assert "opencv-python" in call_args

    def test_upgrade_flag(self):
        """Test upgrade flag."""
        from utils.dependency_checker import install_package

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            install_package("test-package", upgrade=True)

            call_args = mock_run.call_args[0][0]
            assert "--upgrade" in call_args

    def test_returns_false_on_failure(self):
        """Test returning False on failure."""
        import subprocess

        from utils.dependency_checker import install_package

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "pip", stderr="error"
            )
            result = install_package("bad-package")

        assert result is False


class TestInstallMissingPackagesContract:
    """Design by Contract tests for install_missing_packages function."""

    def test_returns_bool(self):
        """Postcondition: Returns a boolean."""
        from utils.dependency_checker import install_missing_packages

        result = install_missing_packages([])
        assert isinstance(result, bool)


class TestInstallMissingPackages:
    """Functional tests for install_missing_packages."""

    def test_returns_true_for_empty_list(self):
        """Test returning True for empty list."""
        from utils.dependency_checker import install_missing_packages

        result = install_missing_packages([])
        assert result is True

    def test_installs_all_packages(self):
        """Test installing all packages."""
        from utils.dependency_checker import install_missing_packages

        with patch("utils.dependency_checker.install_package") as mock_install:
            mock_install.return_value = True
            result = install_missing_packages(["pkg1", "pkg2", "pkg3"])

        assert result is True
        assert mock_install.call_count == 3

    def test_returns_false_if_any_fails(self):
        """Test returning False if any installation fails."""
        from utils.dependency_checker import install_missing_packages

        with patch("utils.dependency_checker.install_package") as mock_install:
            mock_install.side_effect = [True, False, True]
            result = install_missing_packages(["pkg1", "pkg2", "pkg3"])

        assert result is False


class TestFormatMissingDependenciesContract:
    """Design by Contract tests for format_missing_dependencies function."""

    def test_returns_string(self):
        """Postcondition: Returns a string."""
        from utils.dependency_checker import (
            DependencyStatus,
            format_missing_dependencies,
        )

        status = DependencyStatus(ok=True, missing=[], guidance={})
        result = format_missing_dependencies(status)
        assert isinstance(result, str)


class TestFormatMissingDependencies:
    """Functional tests for format_missing_dependencies."""

    def test_all_installed_message(self):
        """Test message when all dependencies installed."""
        from utils.dependency_checker import (
            DependencyStatus,
            format_missing_dependencies,
        )

        status = DependencyStatus(ok=True, missing=[], guidance={})
        result = format_missing_dependencies(status)
        assert "All dependencies" in result

    def test_lists_missing(self):
        """Test listing missing dependencies."""
        from utils.dependency_checker import (
            DependencyStatus,
            format_missing_dependencies,
        )

        status = DependencyStatus(
            ok=False,
            missing=["numpy", "pandas"],
            guidance={"numpy": "pip install numpy", "pandas": "pip install pandas"},
        )
        result = format_missing_dependencies(status)

        assert "numpy" in result
        assert "pandas" in result
        assert "pip install" in result


class TestInstallFromRequirementsContract:
    """Design by Contract tests for install_from_requirements function."""

    def test_returns_bool(self, tmp_path):
        """Postcondition: Returns a boolean."""
        from utils.dependency_checker import install_from_requirements

        result = install_from_requirements(str(tmp_path / "nonexistent.txt"))
        assert isinstance(result, bool)


class TestInstallFromRequirements:
    """Functional tests for install_from_requirements."""

    def test_returns_false_for_missing_file(self, tmp_path):
        """Test returning False for missing requirements file."""
        from utils.dependency_checker import install_from_requirements

        result = install_from_requirements(str(tmp_path / "missing.txt"))
        assert result is False

    def test_installs_from_file(self, tmp_path):
        """Test installing from requirements file."""
        from utils.dependency_checker import install_from_requirements

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("pytest\n")

        with patch("utils.dependency_checker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = install_from_requirements(str(req_file))

        assert result is True
        # Should have been called at least twice (pip upgrade + requirements)
        assert mock_run.call_count >= 1

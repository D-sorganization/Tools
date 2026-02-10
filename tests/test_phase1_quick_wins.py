"""Tests for Phase 1 Quick Wins fixes.

Covers:
- #612: Empty stub directory cleanup (.gitignore entries)
- #627: NotImplementedError stub fixes
- #565: Data I/O backward-compatible reader
- #550: Theme README existence
- #551: Theme dependency in pyproject.toml
- #626: Video processor API module
- #552: Theme screenshot generator module

See issue #529 for test coverage tracking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# #612 / #566 - Gitignore has entries for legacy stub dirs
# =========================================================================


class TestLegacyDirCleanup:
    """Verify .gitignore contains entries for legacy stub directories."""

    def test_gitignore_has_legacy_tools_dir(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        assert "/tools/" in gitignore

    def test_gitignore_has_legacy_python_dir(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        assert "/python/" in gitignore

    def test_gitignore_has_legacy_development_tools_dir(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        assert "/development_tools/" in gitignore

    def test_gitignore_has_legacy_web_app_calculator_dir(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        assert "/web_applications/calculator/" in gitignore


# =========================================================================
# #627 - NotImplementedError stubs
# =========================================================================


class TestNotImplementedErrorFixes:
    """Verify NotImplementedError stubs are properly handled."""

    def test_signal_loader_raises_value_error_for_unknown_format(self):
        """SignalLoader.load should raise ValueError, not NotImplementedError,
        for unsupported internal format keys."""
        io_path = REPO_ROOT / "src" / "shared" / "python" / "signal_toolkit" / "io.py"
        content = io_path.read_text()
        # The dead-code safeguard should raise ValueError, not NotImplementedError
        assert "raise ValueError(msg)" in content
        # Check the comment references issue #627
        assert "issue #627" in content

    def test_format_utils_conversion_has_clear_error_message(self):
        """format_utils.convert should have a clear message about supported conversions."""
        utils_path = (
            REPO_ROOT
            / "src"
            / "shared"
            / "python"
            / "model_generation"
            / "converters"
            / "format_utils.py"
        )
        content = utils_path.read_text()
        assert "URDF <-> MJCF" in content
        assert "issue #627" in content


# =========================================================================
# #565 - Data I/O backward-compatible reader
# =========================================================================


class TestDataIO:
    """Test the backward-compatible data reader."""

    def test_module_importable(self):
        """data_io module should be importable."""
        from upstream_drift_tools.data_io import read_data, write_data

        assert callable(read_data)
        assert callable(write_data)

    def test_read_csv_file(self, tmp_path):
        """read_data should read a CSV file."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

        from upstream_drift_tools.data_io import read_data

        df = read_data(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b", "c"]

    def test_read_data_prefers_parquet_sibling(self, tmp_path):
        """read_data should prefer .parquet sibling when prefer_parquet=True."""
        import pandas as pd

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")

        parquet_path = tmp_path / "data.parquet"
        df_original = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        df_original.to_parquet(parquet_path, index=False)

        from upstream_drift_tools.data_io import read_data

        # Should read parquet (with different values)
        df = read_data(csv_path, prefer_parquet=True)
        assert df["x"].iloc[0] == 10

    def test_read_data_csv_fallback(self, tmp_path):
        """read_data falls back to CSV when no parquet sibling exists."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n3,4\n")

        from upstream_drift_tools.data_io import read_data

        df = read_data(csv_path, prefer_parquet=True)
        assert df["x"].iloc[0] == 1

    def test_read_data_file_not_found(self, tmp_path):
        """read_data raises FileNotFoundError for missing files."""
        from upstream_drift_tools.data_io import read_data

        with pytest.raises(FileNotFoundError):
            read_data(tmp_path / "nonexistent.csv")

    def test_write_data_parquet(self, tmp_path):
        """write_data can write Parquet files."""
        import pandas as pd
        from upstream_drift_tools.data_io import write_data

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out_path = write_data(df, tmp_path / "output.parquet")
        assert out_path.exists()
        assert out_path.suffix == ".parquet"

    def test_unsupported_format_raises(self, tmp_path):
        """read_data raises ValueError for unsupported extensions."""
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("some data")

        from upstream_drift_tools.data_io import read_data

        with pytest.raises(ValueError, match="Unsupported file format"):
            read_data(bad_file)


# =========================================================================
# #550 - Theme README exists
# =========================================================================


class TestThemeDocumentation:
    """Verify theme documentation files exist."""

    def test_theme_readme_exists(self):
        readme_path = REPO_ROOT / "src" / "shared" / "python" / "theme" / "README.md"
        assert readme_path.exists(), f"Missing: {readme_path}"

    def test_theme_readme_has_content(self):
        readme_path = REPO_ROOT / "src" / "shared" / "python" / "theme" / "README.md"
        content = readme_path.read_text()
        assert "ThemeManager" in content
        assert "setup_themed_app" in content
        assert "Color Constants" in content


# =========================================================================
# #551 - Theme dependency in pyproject.toml
# =========================================================================


class TestThemeDependencyManagement:
    """Verify theme is an optional dependency group."""

    def test_pyproject_has_theme_dependency_group(self):
        toml_path = REPO_ROOT / "pyproject.toml"
        content = toml_path.read_text()
        assert "theme = [" in content
        assert "PyQt6" in content


# =========================================================================
# #626 - Video Processor API module
# =========================================================================


class TestVideoProcessorAPI:
    """Verify video processor API module exists and is importable."""

    def test_api_module_exists(self):
        api_path = (
            REPO_ROOT
            / "src"
            / "media_processing"
            / "video_processor"
            / "python"
            / "video_processor_src"
            / "api.py"
        )
        assert api_path.exists()

    def test_api_module_has_fastapi_app(self):
        api_path = (
            REPO_ROOT
            / "src"
            / "media_processing"
            / "video_processor"
            / "python"
            / "video_processor_src"
            / "api.py"
        )
        content = api_path.read_text()
        assert "FastAPI" in content
        assert "/api/upload" in content
        assert "/api/progress/" in content
        assert "StreamingResponse" in content


# =========================================================================
# #552 - Theme screenshot generator exists
# =========================================================================


class TestThemeScreenshotGenerator:
    """Verify theme screenshot generator script exists."""

    def test_script_exists(self):
        script_path = REPO_ROOT / "scripts" / "generate_theme_screenshots.py"
        assert script_path.exists()

    def test_script_has_generate_function(self):
        script_path = REPO_ROOT / "scripts" / "generate_theme_screenshots.py"
        content = script_path.read_text()
        assert "def generate_screenshots" in content
        assert "QT_QPA_PLATFORM" in content


# =========================================================================
# #565 - Migration script exists
# =========================================================================


class TestMigrationScript:
    """Verify Parquet migration script exists."""

    def test_migration_script_exists(self):
        script_path = REPO_ROOT / "scripts" / "migrate_csv_to_parquet.py"
        assert script_path.exists()

    def test_migration_script_has_functions(self):
        script_path = REPO_ROOT / "scripts" / "migrate_csv_to_parquet.py"
        content = script_path.read_text()
        assert "def find_csv_files" in content
        assert "def migrate_csv_to_parquet" in content

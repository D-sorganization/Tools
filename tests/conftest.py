"""
Root conftest.py with shared fixtures for all tests.

This module provides common pytest fixtures and configuration used across
all test suites in the repository.
"""

import logging
import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Headless / thread-safety env vars — must be set BEFORE any scipy/matplotlib
# import so that both the main process and any xdist worker sub-processes
# (which re-execute conftest.py on startup) get a stable non-GUI backend.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# vessel_drafter path — not managed by pytest.ini pythonpath due to worktree
# import ordering; add here so collection succeeds.
# ---------------------------------------------------------------------------
_VESSEL_DRAFTER_PYTHON = (
    Path(__file__).resolve().parent.parent / "src" / "vessel_drafter" / "python"
)
if _VESSEL_DRAFTER_PYTHON.exists() and str(_VESSEL_DRAFTER_PYTHON) not in sys.path:
    sys.path.insert(0, str(_VESSEL_DRAFTER_PYTHON))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HEADLESS", "true")

import pytest

# =============================================================================
# Path Constants
# =============================================================================
# Note: Python path is now configured via pytest.ini pythonpath setting.
# These constants are kept for use in fixtures that need path references.

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
PYTHON_SRC_DIR = SRC_DIR / "python" / "src"
TOOLS_DIR = SRC_DIR / "tools"


# =============================================================================
# Pytest Configuration Hooks
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network"
    )
    config.addinivalue_line(
        "markers", "requires_database: mark test as requiring database"
    )
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify collected test items.

    Automatically adds 'unit' marker to tests not marked as integration/e2e.
    """
    for item in items:
        # Auto-mark tests without explicit markers as unit tests
        if not any(
            marker.name in ("integration", "e2e", "unit")
            for marker in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests based on markers and available resources."""
    # Skip network tests if SKIP_NETWORK_TESTS is set
    if item.get_closest_marker("requires_network"):
        if os.environ.get("SKIP_NETWORK_TESTS", "").lower() in ("1", "true"):
            pytest.skip("Network tests disabled")

    # Skip GPU tests if no GPU available
    if item.get_closest_marker("requires_gpu"):
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not installed")


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root directory.

    Returns:
        Path to repository root
    """
    return REPO_ROOT


@pytest.fixture(scope="session")
def src_dir() -> Path:
    """Return the src directory.

    Returns:
        Path to src directory
    """
    return SRC_DIR


@pytest.fixture(scope="session")
def test_assets_dir(repo_root: Path) -> Path:
    """Return the test assets directory, creating if needed.

    Args:
        repo_root: Repository root path

    Returns:
        Path to test assets directory
    """
    assets_dir = repo_root / "tests" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


@pytest.fixture(scope="session")
def session_temp_dir() -> Generator[Path, None, None]:
    """Create a session-scoped temporary directory.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory(prefix="pytest_session_") as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Function-Scoped Fixtures
# =============================================================================


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Return a function-scoped temporary directory.

    Args:
        tmp_path: pytest's built-in tmp_path fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def temp_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary file.

    Args:
        tmp_path: pytest's built-in tmp_path fixture

    Yields:
        Path to temporary file
    """
    file_path = tmp_path / "test_file.txt"
    file_path.touch()
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file with content.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to sample text file
    """
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Line 1\nLine 2\nLine 3\n")
    return file_path


@pytest.fixture
def sample_json_file(tmp_path: Path) -> Path:
    """Create a sample JSON file.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to sample JSON file
    """
    import json

    file_path = tmp_path / "sample.json"
    data = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}
    file_path.write_text(json.dumps(data, indent=2))
    return file_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    """Create a sample CSV file.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to sample CSV file
    """
    file_path = tmp_path / "sample.csv"
    content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago\n"
    file_path.write_text(content)
    return file_path


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger.

    Returns:
        Mock logger object
    """
    mock = MagicMock(spec=logging.Logger)
    mock.level = logging.DEBUG
    mock.handlers = []
    mock.name = "mock_logger"
    return mock


@pytest.fixture
def mock_file() -> MagicMock:
    """Create a mock file object.

    Returns:
        Mock file object
    """
    mock = MagicMock()
    mock.name = "mock_file.txt"
    mock.configure_mock(**{"read.return_value": "", "write.return_value": None})
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


@pytest.fixture
def mock_response() -> MagicMock:
    """Create a mock HTTP response.

    Returns:
        Mock response object
    """
    mock = MagicMock()
    mock.status_code = 200
    mock.configure_mock(**{"json.return_value": {}})
    mock.text = ""
    mock.headers = {}
    mock.ok = True
    return mock


@pytest.fixture
def mock_path(tmp_path: Path) -> MagicMock:
    """Create a mock Path object.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Mock Path object
    """
    mock = MagicMock(spec=Path)
    mock.configure_mock(
        **{
            "exists.return_value": True,
            "is_file.return_value": True,
            "is_dir.return_value": False,
            "read_text.return_value": "",
            "read_bytes.return_value": b"",
        }
    )
    mock.parent = tmp_path
    return mock


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture
def clean_environment() -> Generator[dict[str, str], None, None]:
    """Provide a clean environment, restoring after test.

    Yields:
        Copy of original environment
    """
    original_env = os.environ.copy()
    yield original_env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def debug_environment(clean_environment: dict[str, str]) -> Generator[None, None, None]:
    """Set up debug environment variables.

    Args:
        clean_environment: Original environment to restore

    Yields:
        None
    """
    os.environ["DEBUG"] = "1"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield


# =============================================================================
# Logging Fixtures
# =============================================================================


@pytest.fixture
def capture_logs() -> Generator[list[logging.LogRecord], None, None]:
    """Capture log records during test.

    Yields:
        List of captured LogRecord objects

    Note:
        Uses captured_logs context manager from utils.test_utils for consistency.
    """
    from utils.test_utils import captured_logs as captured_logs_ctx

    with captured_logs_ctx() as records:
        yield records


@pytest.fixture
def silent_logging() -> Generator[None, None, None]:
    """Suppress all logging output during test.

    Yields:
        None
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_dict() -> dict[str, Any]:
    """Provide a sample dictionary for testing.

    Returns:
        Sample dictionary
    """
    return {
        "string": "hello",
        "number": 42,
        "float": 3.14,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
    }


@pytest.fixture
def sample_list() -> list[Any]:
    """Provide a sample list for testing.

    Returns:
        Sample list
    """
    return [1, "two", 3.0, True, None, [4, 5], {"key": "value"}]


@pytest.fixture
def large_dataset() -> list[dict[str, Any]]:
    """Generate a larger dataset for performance testing.

    Returns:
        List of dictionaries
    """
    import random

    random.seed(42)
    return [
        {
            "id": i,
            "name": f"item_{i}",
            "value": random.random() * 100,
            "category": random.choice(["A", "B", "C"]),
        }
        for i in range(1000)
    ]


# =============================================================================
# Timing and Performance Fixtures
# =============================================================================


@pytest.fixture
def timing_threshold() -> float:
    """Default timing threshold for performance tests.

    Returns:
        Threshold in seconds
    """
    return 1.0


@pytest.fixture
def performance_watchdog() -> Generator[Any, None, None]:
    """Provide performance watchdog for tracking timings.

    Yields:
        PerformanceWatchdog instance
    """
    try:
        from utils.debug_utils import PerformanceWatchdog

        watchdog = PerformanceWatchdog(
            warn_threshold_ms=500.0,
            error_threshold_ms=2000.0,
        )
        yield watchdog
    except ImportError:
        # Fallback if debug_utils not available
        yield MagicMock()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_logging() -> Generator[None, None, None]:
    """Reset logging configuration after each test.

    Yields:
        None
    """
    yield
    # Clear all handlers from root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path: Path) -> Generator[None, None, None]:
    """Ensure temporary files are cleaned up.

    Args:
        tmp_path: Temporary directory path

    Yields:
        None
    """
    yield
    # tmp_path is automatically cleaned up by pytest


# =============================================================================
# Skip Condition Helpers
# =============================================================================


def requires_pandas() -> pytest.MarkDecorator:
    """Skip test if pandas is not available.

    Returns:
        pytest skip marker
    """
    try:
        import pandas  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="pandas not installed")


def requires_numpy() -> pytest.MarkDecorator:
    """Skip test if numpy is not available.

    Returns:
        pytest skip marker
    """
    try:
        import numpy  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="numpy not installed")


def requires_torch() -> pytest.MarkDecorator:
    """Skip test if torch is not available.

    Returns:
        pytest skip marker
    """
    try:
        import torch  # noqa: F401

        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skip(reason="torch not installed")


# Export helper functions for use in tests
__all__ = [
    "requires_pandas",
    "requires_numpy",
    "requires_torch",
]

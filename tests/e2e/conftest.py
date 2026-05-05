"""Fixtures for end-to-end tests.

Provides fixtures that simulate real-world usage patterns and complete workflows
across multiple tools and systems.
"""

import json
import logging
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def e2e_assets_dir(repo_root: Path) -> Path:
    """Provide the E2E test assets directory.

    Creates the directory if it doesn't exist.

    Args:
        repo_root: Repository root path

    Returns:
        Path to E2E test assets directory
    """
    assets_dir = repo_root / "tests" / "assets" / "e2e"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


@pytest.fixture
def e2e_temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for E2E tests.

    Args:
        tmp_path: pytest's built-in tmp_path fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def real_world_sensor_data(e2e_temp_dir: Path) -> Path:
    """Create realistic sensor data file for end-to-end testing.

    Simulates real-world sensor measurements with timestamps, multiple
    sensor types, and realistic data variations.

    Args:
        e2e_temp_dir: Temporary directory for test files

    Returns:
        Path to CSV file with sensor data
    """
    csv_path = e2e_temp_dir / "sensor_measurements.csv"

    # Create realistic sensor dataset
    timestamps = pd.date_range("2024-01-01 00:00", periods=1000, freq="15min")
    data = {
        "timestamp": timestamps,
        "sensor_id": [f"SENSOR_{i % 5}" for i in range(1000)],
        "temperature_c": [20 + 10 * (i % 50) / 50 for i in range(1000)],
        "humidity_percent": [50 + 30 * (i % 40) / 40 for i in range(1000)],
        "pressure_kpa": [101.325 + 5 * ((i // 100) % 10) / 10 for i in range(1000)],
        "quality_flag": ["OK" if i % 20 != 0 else "MAINTENANCE" for i in range(1000)],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def configuration_file(e2e_temp_dir: Path) -> Path:
    """Create a configuration file for workflow tests.

    Args:
        e2e_temp_dir: Temporary directory for test files

    Returns:
        Path to configuration JSON file
    """
    config_path = e2e_temp_dir / "workflow_config.json"

    config = {
        "workflow_name": "end_to_end_test",
        "version": "1.0.0",
        "description": "E2E workflow configuration",
        "stages": [
            {
                "name": "data_ingestion",
                "type": "import",
                "source": "sensor_measurements.csv",
                "format": "csv",
            },
            {
                "name": "data_filtering",
                "type": "filter",
                "field": "quality_flag",
                "condition": "eq",
                "value": "OK",
            },
            {
                "name": "data_transformation",
                "type": "transform",
                "operations": [
                    {"field": "temperature_c", "type": "round", "precision": 2},
                    {"field": "humidity_percent", "type": "normalize", "min": 0, "max": 100},
                ],
            },
            {
                "name": "export",
                "type": "export",
                "destination": "processed_data.json",
                "format": "json",
            },
        ],
    }

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    return config_path


@pytest.fixture
def expected_workflow_outputs(e2e_temp_dir: Path) -> dict[str, Path]:
    """Provide expected output file paths for E2E tests.

    Args:
        e2e_temp_dir: Temporary directory for test files

    Returns:
        Dictionary with expected output paths
    """
    return {
        "filtered_data": e2e_temp_dir / "filtered_data.csv",
        "transformed_data": e2e_temp_dir / "transformed_data.json",
        "statistics": e2e_temp_dir / "statistics.json",
        "report": e2e_temp_dir / "workflow_report.txt",
        "audit_log": e2e_temp_dir / "audit.log",
    }


@pytest.fixture
def e2e_logger() -> logging.Logger:
    """Provide a configured logger for E2E tests.

    Returns:
        Logger instance for E2E tests
    """
    logger = logging.getLogger("e2e_tests")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

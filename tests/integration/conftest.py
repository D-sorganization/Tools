"""Fixtures for integration tests.

Provides shared fixtures for testing tool interactions, data flows, and
cross-module functionality. All fixtures in this module support realistic
scenarios with actual file I/O and tool APIs.
"""

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Configure logging for integration tests
logger = logging.getLogger(__name__)


# ============================================================================
# Session and Function Scoped Path Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def integration_assets_dir(repo_root: Path) -> Path:
    """Provide the integration test assets directory.

    Creates the directory if it doesn't exist.

    Args:
        repo_root: Repository root path

    Returns:
        Path to integration test assets directory
    """
    assets_dir = repo_root / "tests" / "assets" / "integration"
    assets_dir.mkdir(parents=True, exist_ok=True)
    return assets_dir


@pytest.fixture
def integration_temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for integration tests.

    Args:
        tmp_path: pytest's built-in tmp_path fixture

    Returns:
        Path to temporary directory
    """
    return tmp_path


# ============================================================================
# Data File Fixtures (CSV, JSON, etc.)
# ============================================================================


@pytest.fixture
def sample_pressure_data_csv(integration_temp_dir: Path) -> Path:
    """Create a sample CSV file with pressure drop test data.

    Creates a realistic pressure calculation input file with pipe parameters
    and gas properties.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to CSV file with pressure data
    """
    csv_path = integration_temp_dir / "pressure_data.csv"

    # Create realistic pressure drop calculation data.
    # Flow rates are kept low relative to pipe size and pressure to avoid
    # choked-flow conditions in the calculator.
    data = {
        "pipe_diameter_m": [0.15, 0.15, 0.20, 0.20, 0.15],
        "pipe_length_m": [100.0, 200.0, 150.0, 250.0, 100.0],
        "flow_rate_kg_s": [0.3, 0.5, 0.6, 0.8, 0.4],
        "inlet_pressure_pa": [
            500_000.0,
            500_000.0,
            500_000.0,
            500_000.0,
            500_000.0,
        ],
        "inlet_temperature_k": [300.0, 300.0, 300.0, 300.0, 300.0],
        "gas_type": ["air", "air", "syngas", "air", "natural_gas"],
        "pipe_roughness": [0.000045, 0.000045, 0.000045, 0.000045, 0.000045],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_processing_data_csv(integration_temp_dir: Path) -> Path:
    """Create a sample CSV file for data processing tests.

    Creates a realistic data set with multiple columns for filtering
    and transformation operations.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to CSV file with processing data
    """
    csv_path = integration_temp_dir / "processing_data.csv"

    # Create realistic data processing dataset
    data = {
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature_c": [20 + 5 * (i % 10) for i in range(100)],
        "pressure_bar": [1.0 + 0.1 * (i % 5) for i in range(100)],
        "flow_rate_l_min": [50 + 10 * (i % 7) for i in range(100)],
        "quality_flag": ["OK" if i % 5 != 0 else "WARN" for i in range(100)],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_data_file(integration_temp_dir: Path) -> Path:
    """Create a sample JSON file for data loading tests.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to JSON file with sample data
    """
    json_path = integration_temp_dir / "data.json"

    # Use a flat list-of-dicts structure so pd.read_json() can parse it
    # without "Mixing dicts with non-Series" errors.
    data = [
        {"id": 1, "value": 10.5, "unit": "bar", "source": "integration_test"},
        {"id": 2, "value": 20.3, "unit": "bar", "source": "integration_test"},
        {"id": 3, "value": 15.8, "unit": "bar", "source": "integration_test"},
    ]

    with json_path.open("w") as f:
        json.dump(data, f, indent=2)

    return json_path


@pytest.fixture
def sample_excel_file(integration_temp_dir: Path) -> Path:
    """Create a sample Excel file for data loading tests.

    Requires openpyxl to be installed.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to Excel file with sample data
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        pytest.skip("openpyxl not installed")

    excel_path = integration_temp_dir / "data.xlsx"

    data = {
        "pipe_id": [1, 2, 3, 4, 5],
        "diameter_mm": [50, 75, 100, 125, 150],
        "material": ["Steel", "Steel", "PVC", "Steel", "Aluminum"],
        "pressure_rating_bar": [10, 16, 6, 20, 10],
    }

    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False, sheet_name="Pipes")
    return excel_path


# ============================================================================
# Pressure Drop Calculator Fixtures
# ============================================================================


@pytest.fixture
def pressure_drop_simple_inputs() -> dict[str, Any]:
    """Provide simple pressure drop calculation inputs.

    Returns:
        Dictionary of pressure drop calculator parameters
    """
    return {
        "pipe_diameter": 0.15,  # 150 mm (6-inch)
        "pipe_length": 100.0,  # 100 meters
        "pipe_roughness": 0.000045,  # Commercial steel
        "flow_rate": 0.5,  # 0.5 kg/s — kept low to avoid choked flow
        "inlet_pressure": 500_000.0,  # 5 bar in Pa
        "inlet_temperature": 300.0,  # ~27°C in K
        "elevation_change": 0.0,
        "gas_composition": {"N2": 0.79, "O2": 0.21},  # Air
    }


@pytest.fixture
def pressure_drop_complex_inputs() -> dict[str, Any]:
    """Provide complex pressure drop calculation inputs.

    Includes multiple fittings and custom gas composition.

    Returns:
        Dictionary of pressure drop calculator parameters
    """
    return {
        "pipe_diameter": 0.20,  # 200 mm (8-inch)
        "pipe_length": 250.0,  # 250 meters
        "pipe_roughness": 0.000045,
        "flow_rate": 0.8,  # 0.8 kg/s — kept low to avoid choked flow
        "inlet_pressure": 800_000.0,  # 8 bar in Pa
        "inlet_temperature": 323.15,  # 50°C in K
        "elevation_change": 50.0,  # 50 meters elevation change
        "gas_composition": {
            "CO": 0.4,
            "H2": 0.3,
            "CO2": 0.2,
            "N2": 0.1,
        },
        "fittings": [
            {"type": "elbow_90", "count": 2},
            {"type": "tee", "count": 1},
            {"type": "valve_gate", "count": 1},
        ],
    }


@pytest.fixture
def pressure_drop_edge_case_inputs() -> dict[str, Any]:
    """Provide edge case pressure drop calculation inputs.

    Tests boundary conditions and unusual parameter combinations.

    Returns:
        Dictionary of pressure drop calculator parameters
    """
    return {
        "pipe_diameter": 0.01,  # 10 mm - small diameter
        "pipe_length": 1000.0,  # 1000 meters - very long
        "pipe_roughness": 0.00015,  # Rough pipe
        "flow_rate": 0.1,  # Very low flow
        "inlet_pressure": 500000.0,  # 5 bar
        "inlet_temperature": 273.15,  # 0°C - very cold
        "elevation_change": -100.0,  # Downward slope
        "gas_composition": {"H2": 1.0},  # Pure hydrogen
    }


# ============================================================================
# Data Processor Fixtures
# ============================================================================


@pytest.fixture
def data_processor_sample_dataframe() -> pd.DataFrame:
    """Provide a sample DataFrame for data processor tests.

    Returns:
        pandas DataFrame with realistic data
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="30min"),
            "temperature": [20.0 + i * 0.5 for i in range(50)],
            "pressure": [1.0 + (i % 10) * 0.1 for i in range(50)],
            "flow_rate": [100.0 - (i % 20) * 2 for i in range(50)],
            "status": ["OK" if i % 5 != 0 else "WARNING" for i in range(50)],
        }
    )


@pytest.fixture
def data_processor_with_outliers() -> pd.DataFrame:
    """Provide a DataFrame with outliers for filter testing.

    Returns:
        pandas DataFrame with intentional outliers
    """
    df = pd.DataFrame(
        {
            "value": [10.0 + i for i in range(50)],
            "measured_at": pd.date_range("2024-01-01", periods=50, freq="1h"),
        }
    )

    # Inject outliers
    df.loc[10, "value"] = 200.0  # Spike
    df.loc[25, "value"] = -50.0  # Dip
    df.loc[40, "value"] = 150.0  # Another spike

    return df


# ============================================================================
# Plugin System Fixtures
# ============================================================================


@pytest.fixture
def mock_plugin_directory(integration_temp_dir: Path) -> Path:
    """Create a mock plugin directory structure.

    Creates a directory with sample plugin files for plugin discovery tests.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to mock plugin directory
    """
    plugins_dir = integration_temp_dir / "plugins"
    plugins_dir.mkdir(exist_ok=True)

    # Create plugin files
    plugin_a = plugins_dir / "plugin_a.py"
    plugin_a.write_text("""
\"\"\"Sample plugin A for testing.\"\"\"

PLUGIN_NAME = "Plugin A"
PLUGIN_VERSION = "1.0.0"

def activate():
    return {"status": "activated", "name": PLUGIN_NAME}

def get_tools():
    return {"tool_a": "function_a", "tool_b": "function_b"}
""")

    plugin_b = plugins_dir / "plugin_b.py"
    plugin_b.write_text("""
\"\"\"Sample plugin B for testing.\"\"\"

PLUGIN_NAME = "Plugin B"
PLUGIN_VERSION = "2.0.0"

def activate():
    return {"status": "activated", "name": PLUGIN_NAME}

def get_tools():
    return {"tool_c": "function_c", "tool_d": "function_d"}
""")

    # Create __init__.py
    init_file = plugins_dir / "__init__.py"
    init_file.write_text('"""Plugin package."""')

    return plugins_dir


@pytest.fixture
def mock_manifest_file(integration_temp_dir: Path) -> Path:
    """Create a mock plugin manifest file.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Path to manifest JSON file
    """
    manifest_path = integration_temp_dir / "manifest.json"

    manifest = {
        "version": "1.0",
        "plugins": [
            {
                "name": "pressure_drop_calculator",
                "version": "1.2.0",
                "module": "upstream_drift_tools.process_calculators.pressure_drop_calculator",  # noqa: E501
                "enabled": True,
            },
            {
                "name": "data_processor",
                "version": "2.1.0",
                "module": "upstream_drift_tools.data_processing",
                "enabled": True,
            },
        ],
    }

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ============================================================================
# Workflow and Pipeline Fixtures
# ============================================================================


@pytest.fixture
def multi_step_workflow_data(integration_temp_dir: Path) -> dict[str, Path]:
    """Provide a complete workflow dataset with multiple file formats.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Dictionary of file paths keyed by format
    """
    files = {}

    # CSV input file
    csv_path = integration_temp_dir / "input_data.csv"
    input_df = pd.DataFrame(
        {
            "sensor_id": [1, 2, 3, 4, 5] * 10,
            "temperature": [20 + i * 0.1 for i in range(50)],
            "humidity": [50 + i * 0.2 for i in range(50)],
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
        }
    )
    input_df.to_csv(csv_path, index=False)
    files["csv_input"] = csv_path

    # JSON metadata
    json_path = integration_temp_dir / "metadata.json"
    metadata = {
        "source": "sensor_array",
        "location": "facility_A",
        "calibration_date": "2024-01-01",
    }
    with json_path.open("w") as f:
        json.dump(metadata, f)
    files["json_metadata"] = json_path

    return files


@pytest.fixture
def workflow_output_paths(integration_temp_dir: Path) -> dict[str, Path]:
    """Provide output paths for workflow tests.

    Args:
        integration_temp_dir: Temporary directory for test files

    Returns:
        Dictionary of prepared output paths
    """
    return {
        "filtered_csv": integration_temp_dir / "filtered_data.csv",
        "transformed_json": integration_temp_dir / "transformed_data.json",
        "statistics_json": integration_temp_dir / "statistics.json",
        "report_txt": integration_temp_dir / "report.txt",
    }


# ============================================================================
# Logger and Monitoring Fixtures
# ============================================================================


@pytest.fixture
def integration_logger() -> logging.Logger:
    """Provide a configured logger for integration tests.

    Returns:
        Logger instance for integration tests
    """
    logger = logging.getLogger("integration_tests")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def temporary_env_vars(monkeypatch) -> Generator[dict[str, str], None, None]:
    """Provide a context manager for temporary environment variable changes.

    Args:
        monkeypatch: pytest's monkeypatch fixture

    Yields:
        Dictionary to store and manage environment variables
    """
    env_vars = {}

    def set_env(key: str, value: str) -> None:
        """Set environment variable temporarily."""
        env_vars[key] = value
        monkeypatch.setenv(key, value)

    yield {"set": set_env, "vars": env_vars}

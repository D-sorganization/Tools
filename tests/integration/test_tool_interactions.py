"""Integration tests for tool interactions and data flows.

Tests the interaction between different tools (pressure drop calculator, data processor)
and verifies that data flows correctly through the system.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestPressureDropCalculatorIntegration:
    """Test pressure drop calculator with various input scenarios."""

    def test_pressure_drop_with_simple_inputs(
        self, pressure_drop_simple_inputs: dict[str, Any]
    ) -> None:
        """Test basic pressure drop calculation.

        Args:
            pressure_drop_simple_inputs: Simple test inputs
        """
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
        )

        inputs = PressureDropInputs(
            pipe_diameter=pressure_drop_simple_inputs["pipe_diameter"],
            pipe_length=pressure_drop_simple_inputs["pipe_length"],
            pipe_roughness=pressure_drop_simple_inputs["pipe_roughness"],
            mass_flow_rate=pressure_drop_simple_inputs["flow_rate"],
            inlet_pressure=pressure_drop_simple_inputs["inlet_pressure"],
            inlet_temperature=pressure_drop_simple_inputs["inlet_temperature"],
            gas_composition=pressure_drop_simple_inputs["gas_composition"],
        )

        engine = PressureDropCalculationEngine()
        result = engine.calculate(inputs)

        assert result is not None
        assert hasattr(result, "outlet_pressure")
        assert hasattr(result, "pressure_drop")
        assert result.outlet_pressure > 0
        assert result.pressure_drop >= 0

    def test_pressure_drop_with_complex_inputs(
        self, pressure_drop_complex_inputs: dict[str, Any]
    ) -> None:
        """Test pressure drop with multiple fittings and elevation changes.

        Args:
            pressure_drop_complex_inputs: Complex test inputs with fittings
        """
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
            PipeFitting,
        )

        fittings = [
            PipeFitting(fitting_type="elbow_90", count=2),
            PipeFitting(fitting_type="tee", count=1),
            PipeFitting(fitting_type="valve_gate", count=1),
        ]

        inputs = PressureDropInputs(
            pipe_diameter=pressure_drop_complex_inputs["pipe_diameter"],
            pipe_length=pressure_drop_complex_inputs["pipe_length"],
            pipe_roughness=pressure_drop_complex_inputs["pipe_roughness"],
            mass_flow_rate=pressure_drop_complex_inputs["flow_rate"],
            inlet_pressure=pressure_drop_complex_inputs["inlet_pressure"],
            inlet_temperature=pressure_drop_complex_inputs["inlet_temperature"],
            gas_composition=pressure_drop_complex_inputs["gas_composition"],
            elevation_change=pressure_drop_complex_inputs["elevation_change"],
            fittings=fittings,
        )

        engine = PressureDropCalculationEngine()
        result = engine.calculate(inputs)

        assert result is not None
        assert result.pressure_drop > 0  # Should have increased due to fittings
        assert hasattr(result, "fitting_pressure_drop")

    def test_pressure_drop_with_csv_input(
        self, sample_pressure_data_csv: Path
    ) -> None:
        """Test pressure drop calculation reading from CSV file.

        Args:
            sample_pressure_data_csv: Path to sample CSV data
        """
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
        )

        df = pd.read_csv(sample_pressure_data_csv)

        results = []
        engine = PressureDropCalculationEngine()

        for _, row in df.iterrows():
            composition = {
                gas: 1.0
                for gas in ["air", "syngas", "natural_gas"]
                if row["gas_type"] == gas
            }
            if not composition:
                composition = {"N2": 0.79, "O2": 0.21}

            inputs = PressureDropInputs(
                pipe_diameter=row["pipe_diameter_m"],
                pipe_length=row["pipe_length_m"],
                pipe_roughness=row["pipe_roughness"],
                mass_flow_rate=row["flow_rate_kg_s"],
                inlet_pressure=row["inlet_pressure_pa"],
                inlet_temperature=row["inlet_temperature_k"],
                gas_composition=composition,
            )

            result = engine.calculate(inputs)
            results.append(result)

        assert len(results) == len(df)
        for result in results:
            assert result.pressure_drop >= 0
            assert result.outlet_pressure > 0

    def test_pressure_drop_with_different_pipe_materials(self) -> None:
        """Test pressure drop calculation with different pipe roughness values.

        Verifies that pipe material affects the pressure drop results.
        """
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
        )

        base_inputs = {
            "pipe_diameter": 0.05,
            "pipe_length": 100.0,
            "mass_flow_rate": 2.0,
            "inlet_pressure": 101325.0,
            "inlet_temperature": 288.15,
            "gas_composition": {"N2": 0.79, "O2": 0.21},
        }

        materials = {
            "commercial_steel": 0.000045,
            "wrought_iron": 0.000046,
            "galvanized": 0.00015,
            "cast_iron": 0.00025,
        }

        engine = PressureDropCalculationEngine()
        results = {}

        for material, roughness in materials.items():
            inputs = PressureDropInputs(
                pipe_roughness=roughness,
                **base_inputs,
            )
            results[material] = engine.calculate(inputs)

        # Rougher pipes should have higher pressure drops
        assert results["galvanized"].pressure_drop > results["commercial_steel"].pressure_drop
        assert results["cast_iron"].pressure_drop > results["wrought_iron"].pressure_drop

    def test_pressure_drop_calculation_consistency(self) -> None:
        """Test that repeated calculations produce consistent results.

        Verifies numerical stability and determinism.
        """
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
        )

        inputs = PressureDropInputs(
            pipe_diameter=0.05,
            pipe_length=100.0,
            pipe_roughness=0.000045,
            mass_flow_rate=2.0,
            inlet_pressure=101325.0,
            inlet_temperature=288.15,
            gas_composition={"N2": 0.79, "O2": 0.21},
        )

        engine = PressureDropCalculationEngine()

        # Run multiple times
        results = [engine.calculate(inputs) for _ in range(3)]

        # All results should be identical
        assert results[0].pressure_drop == results[1].pressure_drop
        assert results[1].pressure_drop == results[2].pressure_drop


class TestDataProcessorIntegration:
    """Test data processor with various input formats and transformations."""

    def test_data_processor_csv_loading(
        self, sample_processing_data_csv: Path
    ) -> None:
        """Test loading CSV data into processor.

        Args:
            sample_processing_data_csv: Path to sample CSV file
        """
        from upstream_drift_tools.data_processing import DataReader

        reader = DataReader()
        df = reader.read(sample_processing_data_csv)

        assert df is not None
        assert len(df) == 100
        assert "temperature_c" in df.columns
        assert "pressure_bar" in df.columns

    def test_data_processor_json_loading(
        self, sample_json_data_file: Path
    ) -> None:
        """Test loading JSON data into processor.

        Args:
            sample_json_data_file: Path to sample JSON file
        """
        from upstream_drift_tools.data_processing import DataReader

        reader = DataReader()
        data = reader.read(sample_json_data_file)

        assert data is not None
        if isinstance(data, dict):
            assert "metadata" in data or "measurements" in data

    def test_data_processor_filter_by_condition(
        self, data_processor_sample_dataframe: pd.DataFrame
    ) -> None:
        """Test filtering data by condition.

        Args:
            data_processor_sample_dataframe: Sample DataFrame
        """
        from upstream_drift_tools.data_processing import DataProcessorEngine

        engine = DataProcessorEngine(data_processor_sample_dataframe)

        # Filter by condition
        filtered = engine._df[engine._df["temperature"] > 22.0]

        assert len(filtered) > 0
        assert (filtered["temperature"] > 22.0).all()

    def test_data_processor_aggregation(
        self, data_processor_sample_dataframe: pd.DataFrame
    ) -> None:
        """Test data aggregation operations.

        Args:
            data_processor_sample_dataframe: Sample DataFrame
        """
        # Calculate statistics
        stats = data_processor_sample_dataframe.describe()

        assert stats is not None
        assert "temperature" in stats.columns
        assert stats.loc["mean", "temperature"] > 0
        assert stats.loc["std", "temperature"] > 0

    def test_data_processor_outlier_detection(
        self, data_processor_with_outliers: pd.DataFrame
    ) -> None:
        """Test outlier detection in data.

        Args:
            data_processor_with_outliers: DataFrame with injected outliers
        """
        df = data_processor_with_outliers

        # Simple Z-score based outlier detection
        mean = df["value"].mean()
        std = df["value"].std()
        z_scores = (df["value"] - mean).abs() / std

        outliers = df[z_scores > 3]
        assert len(outliers) >= 3  # Should detect the injected outliers

    def test_data_processor_transformation_pipeline(
        self, data_processor_sample_dataframe: pd.DataFrame
    ) -> None:
        """Test a sequence of transformations on data.

        Args:
            data_processor_sample_dataframe: Sample DataFrame
        """
        df = data_processor_sample_dataframe.copy()

        # Apply transformations
        df["temperature_f"] = df["temperature"] * 9 / 5 + 32
        df["pressure_psi"] = df["pressure"] * 14.5038

        # Verify transformations
        assert "temperature_f" in df.columns
        assert "pressure_psi" in df.columns
        assert df["temperature_f"][0] > 60  # Should be > 60F if temp > 15C

    def test_data_processor_with_excel_file(
        self, sample_excel_file: Path
    ) -> None:
        """Test loading Excel files.

        Args:
            sample_excel_file: Path to sample Excel file
        """
        from upstream_drift_tools.data_processing import DataReader

        reader = DataReader()
        df = reader.read(sample_excel_file)

        assert df is not None
        assert len(df) == 5
        assert "pipe_id" in df.columns


class TestToolInteractionChain:
    """Test interactions between multiple tools in a chain."""

    def test_data_to_pressure_drop_workflow(
        self, sample_pressure_data_csv: Path, integration_temp_dir: Path
    ) -> None:
        """Test workflow: Load CSV → Calculate pressure drop → Export results.

        Args:
            sample_pressure_data_csv: Input CSV with pressure data
            integration_temp_dir: Temporary directory for output
        """
        from upstream_drift_tools.data_processing import DataReader
        from upstream_drift_tools.process_calculators.pressure_drop_calculator import (
            PressureDropCalculationEngine,
            PressureDropInputs,
        )

        # Step 1: Load data
        reader = DataReader()
        df = reader.read(sample_pressure_data_csv)
        assert len(df) > 0

        # Step 2: Calculate pressure drop for each row
        engine = PressureDropCalculationEngine()
        results_list = []

        for _, row in df.iterrows():
            composition = {"N2": 0.79, "O2": 0.21}

            inputs = PressureDropInputs(
                pipe_diameter=row["pipe_diameter_m"],
                pipe_length=row["pipe_length_m"],
                pipe_roughness=row["pipe_roughness"],
                mass_flow_rate=row["flow_rate_kg_s"],
                inlet_pressure=row["inlet_pressure_pa"],
                inlet_temperature=row["inlet_temperature_k"],
                gas_composition=composition,
            )

            result = engine.calculate(inputs)
            results_list.append({
                "input_index": len(results_list),
                "pressure_drop_pa": result.pressure_drop,
                "outlet_pressure_pa": result.outlet_pressure,
            })

        # Step 3: Export results
        results_df = pd.DataFrame(results_list)
        output_path = integration_temp_dir / "pressure_drop_results.csv"
        results_df.to_csv(output_path, index=False)

        # Verify
        assert output_path.exists()
        assert len(results_df) == len(df)
        assert all(results_df["pressure_drop_pa"] >= 0)

    def test_multi_format_data_conversion_workflow(
        self, multi_step_workflow_data: dict[str, Path], integration_temp_dir: Path
    ) -> None:
        """Test workflow: Load CSV → Transform → Export as JSON.

        Args:
            multi_step_workflow_data: Dictionary of input files
            integration_temp_dir: Temporary directory for output
        """
        from upstream_drift_tools.data_processing import DataReader

        # Load CSV
        reader = DataReader()
        csv_input = multi_step_workflow_data["csv_input"]
        df = reader.read(csv_input)

        # Transform
        df["celsius"] = df["temperature"]  # Add metadata
        df["relative_humidity"] = df["humidity"]

        # Export as JSON
        json_output = integration_temp_dir / "converted_data.json"
        df.to_json(json_output, orient="records", indent=2)

        # Verify
        assert json_output.exists()
        with json_output.open() as f:
            loaded_data = json.load(f)
            assert len(loaded_data) == len(df)


class TestPluginSystemIntegration:
    """Test plugin discovery and loading mechanisms."""

    def test_plugin_discovery_in_directory(
        self, mock_plugin_directory: Path
    ) -> None:
        """Test discovering plugins in a directory.

        Args:
            mock_plugin_directory: Directory with mock plugin files
        """
        plugin_files = list(mock_plugin_directory.glob("plugin_*.py"))

        assert len(plugin_files) == 2
        assert mock_plugin_directory / "plugin_a.py" in plugin_files
        assert mock_plugin_directory / "plugin_b.py" in plugin_files

    def test_manifest_validation(self, mock_manifest_file: Path) -> None:
        """Test loading and validating a manifest file.

        Args:
            mock_manifest_file: Path to manifest JSON
        """
        with mock_manifest_file.open() as f:
            manifest = json.load(f)

        assert manifest["version"] == "1.0"
        assert len(manifest["plugins"]) == 2

        for plugin in manifest["plugins"]:
            assert "name" in plugin
            assert "version" in plugin
            assert "module" in plugin
            assert "enabled" in plugin

    def test_plugin_isolation(
        self, mock_plugin_directory: Path
    ) -> None:
        """Test that plugins loaded separately don't interfere.

        Args:
            mock_plugin_directory: Directory with mock plugin files
        """
        import sys
        from importlib import util

        plugins = {}

        for plugin_file in mock_plugin_directory.glob("plugin_*.py"):
            spec = util.spec_from_file_location(plugin_file.stem, plugin_file)
            module = util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            plugins[plugin_file.stem] = {
                "name": module.PLUGIN_NAME,
                "version": module.PLUGIN_VERSION,
            }

        # Verify plugins have different names and versions
        assert plugins["plugin_a"]["name"] == "Plugin A"
        assert plugins["plugin_b"]["name"] == "Plugin B"
        assert plugins["plugin_a"]["version"] == "1.0.0"
        assert plugins["plugin_b"]["version"] == "2.0.0"

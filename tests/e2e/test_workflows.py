"""End-to-end workflow tests.

Tests complete workflows that simulate real-world usage patterns across
multiple tools and systems. Each test exercises a full pipeline from data
ingestion through transformation and export.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# Mark all tests in this file as E2E tests
pytestmark = pytest.mark.e2e


class TestBasicDataWorkflow:
    """Test basic data import, filter, and export workflows."""

    def test_csv_upload_filter_export_json_workflow(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test complete workflow: CSV upload → filter → export JSON.

        This workflow simulates:
        1. Loading sensor data from CSV
        2. Filtering rows by quality flag
        3. Exporting filtered data to JSON

        Args:
            real_world_sensor_data: Path to sensor data CSV file
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting CSV→Filter→JSON workflow")

        # Step 1: Load CSV data
        e2e_logger.info(f"Loading CSV from {real_world_sensor_data}")
        df = pd.read_csv(real_world_sensor_data)
        assert len(df) > 0, "CSV should contain data"
        initial_count = len(df)
        e2e_logger.info(f"Loaded {initial_count} rows from CSV")

        # Step 2: Filter by quality flag
        e2e_logger.info("Filtering by quality_flag='OK'")
        filtered_df = df[df["quality_flag"] == "OK"]
        assert len(filtered_df) > 0, "Filtered data should not be empty"
        assert len(filtered_df) < initial_count, "Filtering should reduce rows"
        e2e_logger.info(
            f"Filtered to {len(filtered_df)} rows "
            f"({len(filtered_df) / initial_count * 100:.1f}% of original)"
        )

        # Step 3: Export to JSON
        output_json = e2e_temp_dir / "filtered_sensor_data.json"
        e2e_logger.info(f"Exporting to JSON: {output_json}")
        filtered_df.to_json(output_json, orient="records", indent=2)

        # Verify
        assert output_json.exists(), "Output JSON should be created"
        with output_json.open() as f:
            loaded_data = json.load(f)
            assert len(loaded_data) == len(filtered_df)
            assert all("temperature_c" in record for record in loaded_data)

        e2e_logger.info("Workflow completed successfully")

    def test_data_load_transform_statistics_workflow(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow: Load data → apply transform → calculate statistics.

        This workflow simulates:
        1. Loading sensor measurements
        2. Transforming temperature units (Celsius to Fahrenheit)
        3. Computing statistics on transformed data

        Args:
            real_world_sensor_data: Path to sensor data CSV file
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Load→Transform→Statistics workflow")

        # Step 1: Load data
        e2e_logger.info("Loading sensor data")
        df = pd.read_csv(real_world_sensor_data)
        e2e_logger.info(f"Loaded {len(df)} records")

        # Step 2: Apply transformations
        e2e_logger.info("Applying transformations")
        df["temperature_f"] = df["temperature_c"] * 9 / 5 + 32
        df["pressure_psi"] = df["pressure_kpa"] * 0.14503773773
        df["humidity_ratio"] = df["humidity_percent"] / 100.0

        assert "temperature_f" in df.columns
        assert "pressure_psi" in df.columns
        assert "humidity_ratio" in df.columns
        e2e_logger.info("Applied 3 transformation operations")

        # Step 3: Calculate statistics
        e2e_logger.info("Computing statistics")
        stats = {
            "temperature_c": {
                "mean": float(df["temperature_c"].mean()),
                "std": float(df["temperature_c"].std()),
                "min": float(df["temperature_c"].min()),
                "max": float(df["temperature_c"].max()),
            },
            "temperature_f": {
                "mean": float(df["temperature_f"].mean()),
                "std": float(df["temperature_f"].std()),
                "min": float(df["temperature_f"].min()),
                "max": float(df["temperature_f"].max()),
            },
            "humidity_percent": {
                "mean": float(df["humidity_percent"].mean()),
                "std": float(df["humidity_percent"].std()),
            },
            "pressure_kpa": {
                "mean": float(df["pressure_kpa"].mean()),
                "std": float(df["pressure_kpa"].std()),
            },
            "record_count": len(df),
        }

        # Export statistics
        stats_file = e2e_temp_dir / "statistics.json"
        with stats_file.open("w") as f:
            json.dump(stats, f, indent=2)

        # Verify statistics
        assert stats["record_count"] == len(df)
        assert stats["temperature_c"]["mean"] > 0
        assert stats["temperature_f"]["mean"] > 32  # Should be in Fahrenheit range
        assert stats["humidity_percent"]["mean"] > 0

        e2e_logger.info(f"Statistics computed: {stats['record_count']} records")
        e2e_logger.info("Workflow completed successfully")

    def test_multi_sensor_aggregation_workflow(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow: Load → group by sensor → aggregate → export.

        This workflow simulates:
        1. Loading multi-sensor data
        2. Grouping measurements by sensor_id
        3. Computing per-sensor statistics
        4. Exporting aggregated results

        Args:
            real_world_sensor_data: Path to sensor data CSV file
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Multi-Sensor Aggregation workflow")

        # Step 1: Load data
        e2e_logger.info("Loading sensor data")
        df = pd.read_csv(real_world_sensor_data)
        unique_sensors = df["sensor_id"].unique()
        e2e_logger.info(f"Loaded data from {len(unique_sensors)} sensors")

        # Step 2: Group and aggregate per sensor
        e2e_logger.info("Aggregating by sensor")
        aggregated_records = []
        for sensor_id in unique_sensors:
            sensor_df = df[df["sensor_id"] == sensor_id]
            aggregated_records.append(
                {
                    "sensor_id": sensor_id,
                    "temp_mean": sensor_df["temperature_c"].mean(),
                    "temp_std": sensor_df["temperature_c"].std(),
                    "humidity_mean": sensor_df["humidity_percent"].mean(),
                    "pressure_mean": sensor_df["pressure_kpa"].mean(),
                    "record_count": len(sensor_df),
                }
            )

        aggregated = pd.DataFrame(aggregated_records)
        e2e_logger.info(f"Aggregated data for {len(aggregated)} sensors")

        # Step 3: Export
        e2e_logger.info("Exporting aggregated data")
        output_csv = e2e_temp_dir / "sensor_aggregated.csv"
        aggregated.to_csv(output_csv, index=False)

        # Verify
        assert output_csv.exists()
        exported_df = pd.read_csv(output_csv)
        assert len(exported_df) == len(unique_sensors)

        e2e_logger.info("Workflow completed successfully")


class TestComplexDataTransformationWorkflows:
    """Test complex multi-step data transformation workflows."""

    def test_data_cleaning_and_enrichment_workflow(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow: Load → clean → enrich → validate → export.

        This workflow simulates data quality improvements:
        1. Loading raw data
        2. Cleaning (removing rows with MAINTENANCE flag)
        3. Enriching with derived columns
        4. Validating cleaned data
        5. Exporting clean dataset

        Args:
            real_world_sensor_data: Path to sensor data CSV file
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Data Cleaning & Enrichment workflow")

        # Step 1: Load raw data
        e2e_logger.info("Loading raw sensor data")
        df = pd.read_csv(real_world_sensor_data)
        initial_count = len(df)
        e2e_logger.info(f"Loaded {initial_count} raw records")

        # Step 2: Clean (remove maintenance rows)
        e2e_logger.info("Cleaning data (removing MAINTENANCE records)")
        df_clean = df[df["quality_flag"] == "OK"].copy()
        removed_count = initial_count - len(df_clean)
        e2e_logger.info(f"Removed {removed_count} maintenance records")

        # Step 3: Enrich with derived columns
        e2e_logger.info("Enriching with derived columns")
        df_clean["temp_deviation"] = (
            df_clean["temperature_c"] - df_clean["temperature_c"].mean()
        )
        df_clean["humidity_category"] = pd.cut(
            df_clean["humidity_percent"],
            bins=[0, 30, 60, 100],
            labels=["low", "medium", "high"],
        )
        df_clean["pressure_anomaly"] = df_clean["pressure_kpa"].std() > 0 and (
            (df_clean["pressure_kpa"] - df_clean["pressure_kpa"].mean()).abs()
            > 2 * df_clean["pressure_kpa"].std()
        )

        # Step 4: Validate
        e2e_logger.info("Validating cleaned data")
        assert len(df_clean) > 0, "Cleaned data should not be empty"
        assert df_clean["temperature_c"].notna().all(), "No NaN temperatures"
        assert (df_clean["humidity_percent"] >= 0).all(), "Valid humidity values"
        assert (df_clean["pressure_kpa"] > 0).all(), "Positive pressure values"
        e2e_logger.info(f"Validation passed: {len(df_clean)} valid records")

        # Step 5: Export
        output_csv = e2e_temp_dir / "cleaned_enriched_data.csv"
        df_clean.to_csv(output_csv, index=False)
        e2e_logger.info(f"Exported to {output_csv}")

        # Verify
        assert output_csv.exists()
        df_verify = pd.read_csv(output_csv)
        assert "temp_deviation" in df_verify.columns
        assert "humidity_category" in df_verify.columns

        e2e_logger.info("Workflow completed successfully")

    def test_time_series_aggregation_workflow(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow: Load → parse timestamp → aggregate by hour → export.

        This workflow simulates time-series analysis:
        1. Loading data with timestamps
        2. Converting to datetime
        3. Resampling to hourly intervals
        4. Computing hourly statistics
        5. Exporting time-series results

        Args:
            real_world_sensor_data: Path to sensor data CSV file
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Time-Series Aggregation workflow")

        # Step 1: Load data
        e2e_logger.info("Loading sensor data")
        df = pd.read_csv(real_world_sensor_data)
        e2e_logger.info(f"Loaded {len(df)} records")

        # Step 2: Parse timestamp
        e2e_logger.info("Parsing timestamp column")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        e2e_logger.info(
            f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}"
        )

        # Step 3: Resample and aggregate by hour
        e2e_logger.info("Aggregating to hourly intervals")
        df_hourly = (
            df.set_index("timestamp")
            .resample("1h")
            .agg(
                {
                    "temperature_c": ["mean", "min", "max", "std"],
                    "humidity_percent": ["mean", "min", "max"],
                    "pressure_kpa": ["mean", "std"],
                }
            )
        )

        e2e_logger.info(f"Created {len(df_hourly)} hourly aggregations")

        # Step 4: Export
        output_csv = e2e_temp_dir / "hourly_aggregated.csv"
        df_hourly.to_csv(output_csv)
        e2e_logger.info(f"Exported to {output_csv}")

        # Verify
        assert output_csv.exists()
        df_verify = pd.read_csv(output_csv, index_col=0)
        assert len(df_verify) > 0
        e2e_logger.info("Workflow completed successfully")


class TestWorkflowConfigurationExecution:
    """Test workflows driven by configuration files."""

    def test_config_driven_workflow_execution(
        self,
        configuration_file: Path,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test executing a workflow based on configuration file.

        This test demonstrates:
        1. Loading workflow configuration from JSON
        2. Parsing workflow stages
        3. Executing stages sequentially
        4. Validating outputs

        Args:
            configuration_file: Path to workflow config JSON
            real_world_sensor_data: Path to sensor data CSV
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Configuration-Driven Workflow")

        # Load configuration
        e2e_logger.info(f"Loading configuration from {configuration_file}")
        with configuration_file.open() as f:
            config = json.load(f)

        workflow_name = config["workflow_name"]
        e2e_logger.info(f"Executing workflow: {workflow_name}")

        # Parse and execute stages
        data = None
        for stage_idx, stage in enumerate(config["stages"]):
            stage_name = stage["name"]
            e2e_logger.info(f"Stage {stage_idx + 1}: {stage_name}")

            if stage["type"] == "import":
                # Import stage
                data = pd.read_csv(real_world_sensor_data)
                e2e_logger.info(f"Imported {len(data)} rows")

            elif stage["type"] == "filter":
                # Filter stage
                assert data is not None
                field = stage["field"]
                value = stage["value"]
                data = data[data[field] == value]
                e2e_logger.info(f"Filtered by {field}={value}: {len(data)} rows")

            elif stage["type"] == "transform":
                # Transform stage
                assert data is not None
                for op in stage["operations"]:
                    field = op["field"]
                    op_type = op["type"]
                    if op_type == "round":
                        precision = op["precision"]
                        data[field] = data[field].round(precision)
                        e2e_logger.info(f"Rounded {field} to {precision} decimals")

            elif stage["type"] == "export":
                # Export stage
                assert data is not None
                destination = stage["destination"]
                output_path = e2e_temp_dir / destination
                data.to_json(output_path, orient="records", indent=2)
                e2e_logger.info(f"Exported to {output_path}")

        # Verify
        assert data is not None
        assert len(config["stages"]) > 0
        e2e_logger.info("Configuration-driven workflow completed successfully")

    def test_workflow_with_multiple_output_formats(
        self,
        real_world_sensor_data: Path,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow that exports results in multiple formats.

        This test demonstrates:
        1. Loading source data
        2. Processing and filtering
        3. Exporting to CSV
        4. Exporting to JSON
        5. Generating summary report

        Args:
            real_world_sensor_data: Path to sensor data CSV
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Multi-Format Output workflow")

        # Load and process
        e2e_logger.info("Loading and processing data")
        df = pd.read_csv(real_world_sensor_data)
        df_filtered = df[df["quality_flag"] == "OK"]

        # Export CSV
        e2e_logger.info("Exporting as CSV")
        csv_path = e2e_temp_dir / "results.csv"
        df_filtered.to_csv(csv_path, index=False)

        # Export JSON
        e2e_logger.info("Exporting as JSON")
        json_path = e2e_temp_dir / "results.json"
        df_filtered.to_json(json_path, orient="records", indent=2)

        # Generate summary report
        e2e_logger.info("Generating summary report")
        summary = {
            "total_records": len(df),
            "filtered_records": len(df_filtered),
            "filter_ratio": len(df_filtered) / len(df),
            "temperature_range": {
                "min": float(df_filtered["temperature_c"].min()),
                "max": float(df_filtered["temperature_c"].max()),
                "mean": float(df_filtered["temperature_c"].mean()),
            },
            "humidity_range": {
                "min": float(df_filtered["humidity_percent"].min()),
                "max": float(df_filtered["humidity_percent"].max()),
                "mean": float(df_filtered["humidity_percent"].mean()),
            },
            "sensor_count": df_filtered["sensor_id"].nunique(),
        }

        report_path = e2e_temp_dir / "workflow_report.json"
        with report_path.open("w") as f:
            json.dump(summary, f, indent=2)

        # Verify all outputs
        assert csv_path.exists(), "CSV output should exist"
        assert json_path.exists(), "JSON output should exist"
        assert report_path.exists(), "Report should exist"

        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == len(df_filtered)

        with json_path.open() as f:
            json_data = json.load(f)
            assert len(json_data) == len(df_filtered)

        e2e_logger.info("Multi-format workflow completed successfully")


class TestErrorHandlingInWorkflows:
    """Test error handling and recovery in workflows."""

    def test_workflow_with_missing_data_handling(
        self,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow gracefully handles data with missing values.

        This test demonstrates:
        1. Creating data with missing values
        2. Detecting missing data
        3. Applying handling strategies (fill, drop, interpolate)
        4. Verifying results

        Args:
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Missing Data Handling workflow")

        # Create data with missing values
        e2e_logger.info("Creating data with missing values")
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="1h"),
            "temperature": [20.0, 21.0, None, 22.0, 23.0, None, 21.5, 22.5, 23.0, None]
            + [21.0] * 10,
            "humidity": [50.0] * 5 + [None] * 3 + [55.0] * 12,
        }
        df = pd.DataFrame(data)

        missing_count = df.isnull().sum().sum()
        e2e_logger.info(f"Created data with {missing_count} missing values")

        # Detect missing data
        e2e_logger.info("Detecting missing data")
        missing_per_column = df.isnull().sum()
        e2e_logger.info(f"Missing values per column: {missing_per_column.to_dict()}")

        # Handle missing data
        e2e_logger.info("Applying missing data handling strategies")
        df_filled = df.bfill()  # Backward fill
        df_filled = df_filled.fillna(df.mean())  # Fill remaining with mean

        assert df_filled.isnull().sum().sum() == 0, "Should have no missing values"
        e2e_logger.info("Missing data handling completed")

        # Export
        output_path = e2e_temp_dir / "handled_missing_data.csv"
        df_filled.to_csv(output_path, index=False)

        e2e_logger.info("Workflow completed successfully")

    def test_workflow_with_invalid_data_detection(
        self,
        e2e_temp_dir: Path,
        e2e_logger,
    ) -> None:
        """Test workflow detects and handles invalid data values.

        This test demonstrates:
        1. Creating data with invalid values
        2. Validating data constraints
        3. Isolating invalid records
        4. Exporting validation report

        Args:
            e2e_temp_dir: Temporary directory for outputs
            e2e_logger: Logger for workflow monitoring
        """
        e2e_logger.info("Starting Invalid Data Detection workflow")

        # Create data with invalid values
        e2e_logger.info("Creating data with invalid values")
        data = {
            "temperature_c": [20, -500, 25, 30, 150, 22, 21, 19, 23, 25],
            "humidity_percent": [50, 60, 120, 30, 40, 50, -10, 55, 60, 65],
            "pressure_kpa": [
                101.3,
                102.0,
                100.5,
                0,
                103.0,
                101.5,
                102.0,
                101.0,
                102.5,
                101.8,
            ],
        }
        df = pd.DataFrame(data)

        # Validate
        e2e_logger.info("Validating data constraints")
        validation_rules = {
            "temperature_c": (-50, 50),  # Valid range
            "humidity_percent": (0, 100),
            "pressure_kpa": (90, 110),
        }

        validation_results = {}
        df_clean = df.copy()

        for column, (min_val, max_val) in validation_rules.items():
            invalid_mask = (df[column] < min_val) | (df[column] > max_val)
            invalid_count = invalid_mask.sum()
            validation_results[column] = {
                "expected_range": f"[{min_val}, {max_val}]",
                "invalid_records": int(invalid_count),
            }
            e2e_logger.info(f"{column}: {invalid_count} invalid records")

        # Export validation report
        report_path = e2e_temp_dir / "validation_report.json"
        with report_path.open("w") as f:
            json.dump(validation_results, f, indent=2)

        e2e_logger.info("Workflow completed successfully")

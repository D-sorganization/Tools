"""Tests for package installation and importability.

These tests verify the packaging contract: dwsim_model must be
importable and expose its public API after pip install.

NOTE: These tests do NOT require DWSIM runtime. They only verify that
the Python package structure is correct and importable.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

MIN_SEMVER_PARTS = 2


@pytest.mark.unit
class TestPackageInstallation:
    """Verify package is installable and importable."""

    def test_dwsim_model_is_importable(self) -> None:
        """The top-level package must be importable."""
        mod = importlib.import_module("dwsim_model")
        assert hasattr(mod, "__version__")
        assert isinstance(mod.__version__, str)
        assert len(mod.__version__) > 0

    def test_version_format(self) -> None:
        """Version string must follow semver-like format."""
        dwsim_model = importlib.import_module("dwsim_model")

        parts = dwsim_model.__version__.split(".")
        assert len(parts) >= MIN_SEMVER_PARTS, "Version must have at least major.minor"
        for part in parts:
            assert part.isdigit(), f"Version part {part!r} must be numeric"

    def test_public_api_gasification_accessible(self) -> None:
        """Core public classes must be importable without DWSIM runtime."""
        # These imports should succeed because they are pure Python at
        # module-definition time. DWSIM runtime is only needed when
        # FlowsheetBuilder is instantiated.
        gasification_module = importlib.import_module("dwsim_model.gasification")

        assert gasification_module.GasificationFlowsheet is not None
        assert gasification_module.ReactorMode.MIXED.value == "mixed"

    def test_constants_importable(self) -> None:
        """Shared constants must be importable."""
        constants_module = importlib.import_module("dwsim_model.constants")

        assert len(constants_module.COMPOUNDS_STANDARD) > 0
        assert isinstance(constants_module.DEFAULT_PROPERTY_PACKAGE, str)
        assert "final_syngas" in constants_module.STREAM_NAMES

    def test_results_modules_importable(self) -> None:
        """Results extraction and metrics must be importable."""
        extractor_module = importlib.import_module("dwsim_model.results.extractor")
        metrics_module = importlib.import_module("dwsim_model.results.metrics")

        assert extractor_module.ResultsExtractor is not None
        assert extractor_module.StreamResult is not None
        assert metrics_module.MetricsCalculator is not None
        assert "Hydrogen" in metrics_module.LHV_MJ_KG

    def test_standalone_units_importable(self) -> None:
        """Standalone unit op modules must be importable at class level."""
        gasifier_module = importlib.import_module(
            "dwsim_model.standalone.gasifier_model"
        )
        pem_module = importlib.import_module("dwsim_model.standalone.pem_model")
        trc_module = importlib.import_module("dwsim_model.standalone.trc_model")

        assert gasifier_module.GasifierStandaloneFlowsheet is not None
        assert pem_module.PEMStandaloneFlowsheet is not None
        assert trc_module.TRCStandaloneFlowsheet is not None

    def test_topology_importable(self) -> None:
        """Topology builder functions must be importable."""
        topology_module = importlib.import_module("dwsim_model.topology")

        assert callable(topology_module.build_gasifier_stage)
        assert callable(topology_module.build_pem_stage)
        assert callable(topology_module.build_trc_stage)

    def test_units_api_importable(self) -> None:
        """Per-unit-op convenience API must be importable."""
        units_module = importlib.import_module("dwsim_model.units")

        assert callable(units_module.run_gasifier)
        assert callable(units_module.run_pem)
        assert callable(units_module.run_trc)

    def test_cli_entry_point_exists(self) -> None:
        """The dwsim-model CLI entry point must respond to --help."""
        result = subprocess.run(
            [sys.executable, "-m", "dwsim_model", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0
        assert "subcommand" in result.stdout.lower() or "usage" in result.stdout.lower()

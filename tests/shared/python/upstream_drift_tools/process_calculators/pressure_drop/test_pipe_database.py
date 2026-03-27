"""Comprehensive tests for pipe_database module.

Tests cover get_roughness, get_pipe_spec, list_available_sizes,
list_schedules_for_size, and create_custom_pipe.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.models.pressure_drop_data_models import (
    PipeSpecification,
)
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.pipe_database import (
    MATERIAL_ROUGHNESS,
    STEEL_PIPE_DIMENSIONS,
    create_custom_pipe,
    get_pipe_spec,
    get_roughness,
    list_available_sizes,
    list_schedules_for_size,
)

# ─── get_roughness Tests ─────────────────────────────────────


class TestGetRoughness:
    def test_mm(self) -> None:
        val = get_roughness("Commercial Steel", unit="mm")
        assert val == 0.045

    def test_ft(self) -> None:
        val = get_roughness("Commercial Steel", unit="ft")
        assert val == 0.00015

    def test_meters(self) -> None:
        val = get_roughness("Commercial Steel", unit="m")
        assert abs(val - 0.000045) < 1e-8

    def test_unknown_material_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_roughness("Unobtanium")

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="not recognized"):
            get_roughness("Commercial Steel", unit="furlongs")

    def test_glass_smoothest(self) -> None:
        glass = get_roughness("Glass", unit="mm")
        steel = get_roughness("Commercial Steel", unit="mm")
        assert glass < steel

    def test_all_materials_positive(self) -> None:
        for mat in MATERIAL_ROUGHNESS:
            assert get_roughness(mat, unit="mm") >= 0.0


# ─── get_pipe_spec Tests ─────────────────────────────────────


class TestGetPipeSpec:
    def test_returns_pipe_spec(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert isinstance(spec, PipeSpecification)

    def test_4inch_sch40_dimensions(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert abs(spec.inner_diameter - 102.26) < 0.01
        assert abs(spec.outer_diameter - 114.3) < 0.01

    def test_unknown_size_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_pipe_spec("99", "40")

    def test_unknown_schedule_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            get_pipe_spec("4", "XYZ")

    def test_custom_material(self) -> None:
        spec = get_pipe_spec("4", "40", material="Stainless Steel 316")
        assert spec.material == "Stainless Steel 316"

    def test_default_material(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert spec.material == "Commercial Steel"

    def test_id_less_than_od(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert spec.inner_diameter < spec.outer_diameter


# ─── list_available_sizes Tests ──────────────────────────────


class TestListAvailableSizes:
    def test_returns_list(self) -> None:
        sizes = list_available_sizes()
        assert isinstance(sizes, list)
        assert len(sizes) > 0

    def test_contains_common_sizes(self) -> None:
        sizes = list_available_sizes()
        assert "4" in sizes
        assert "6" in sizes
        assert "8" in sizes


# ─── list_schedules_for_size Tests ───────────────────────────


class TestListSchedulesForSize:
    def test_returns_list(self) -> None:
        schedules = list_schedules_for_size("4")
        assert isinstance(schedules, list)
        assert len(schedules) > 0

    def test_4inch_has_sch40(self) -> None:
        schedules = list_schedules_for_size("4")
        assert "40" in schedules

    def test_unknown_size_empty(self) -> None:
        schedules = list_schedules_for_size("999")
        assert schedules == []


# ─── create_custom_pipe Tests ────────────────────────────────


class TestCreateCustomPipe:
    def test_returns_pipe_spec(self) -> None:
        spec = create_custom_pipe(100.0)
        assert isinstance(spec, PipeSpecification)

    def test_custom_label(self) -> None:
        spec = create_custom_pipe(100.0)
        assert spec.nominal_size == "Custom"
        assert spec.schedule == "Custom"

    def test_id_matches(self) -> None:
        spec = create_custom_pipe(150.0)
        assert spec.inner_diameter == 150.0

    def test_custom_material(self) -> None:
        spec = create_custom_pipe(100.0, material="PVC")
        assert spec.material == "PVC"


# ─── Database Integrity Tests ────────────────────────────────


class TestDatabaseIntegrity:
    def test_all_entries_have_consistent_dimensions(self) -> None:
        for key, (od, wall, id_val) in STEEL_PIPE_DIMENSIONS.items():
            expected_id = od - 2 * wall
            assert abs(id_val - expected_id) < 0.1, (
                f"Pipe {key}: ID={id_val} != OD-2*wall={expected_id}"
            )

    def test_database_not_empty(self) -> None:
        assert len(STEEL_PIPE_DIMENSIONS) > 50

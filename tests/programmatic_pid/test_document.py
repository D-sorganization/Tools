"""Functional tests for programmatic_pid.document.PIDDocument.

Tests cover: construction, spatial queries, validation output, export,
and precondition enforcement (#3172 epic — programmatic_pid coverage).
"""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile

import pytest
from programmatic_pid.document import PIDDocument
from programmatic_pid.types import BBox, Point

_MINIMAL_SPEC: dict = {
    "project": {"id": "TEST-001", "title": "Test P&ID"},
    "equipment": [
        {
            "id": "V-101",
            "type": "vessel",
            "label": "Feed Tank",
            "x": 10.0,
            "y": 20.0,
            "w": 20.0,
            "h": 15.0,
        },
        {
            "id": "P-101",
            "type": "pump",
            "label": "Feed Pump",
            "x": 50.0,
            "y": 22.0,
            "w": 10.0,
            "h": 10.0,
        },
    ],
}

requires_ezdxf = pytest.mark.skipif(
    importlib.util.find_spec("ezdxf") is None,
    reason="ezdxf not installed",
)


class TestPIDDocumentConstruction:
    def test_constructs_from_valid_dict(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        assert doc is not None

    def test_raises_on_none_spec(self) -> None:
        with pytest.raises(ValueError):
            PIDDocument(None)

    def test_equipment_ids_populated(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        assert "V-101" in doc.equipment_ids
        assert "P-101" in doc.equipment_ids

    def test_instrument_ids_empty_when_none(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        assert isinstance(doc.instrument_ids, list)

    def test_stream_ids_empty_when_none(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        assert isinstance(doc.stream_ids, list)

    def test_spec_property_returns_validated_dict(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        assert isinstance(doc.spec, dict)
        assert doc.spec["project"]["id"] == "TEST-001"

    def test_from_partial_returns_document_on_valid_spec(self) -> None:
        result = PIDDocument.from_partial(_MINIMAL_SPEC)
        assert result is not None
        assert isinstance(result, PIDDocument)

    def test_from_partial_returns_none_on_missing_required_fields(self) -> None:
        bad: dict = {"project": {}, "equipment": []}
        result = PIDDocument.from_partial(bad)
        assert result is None

    def test_from_partial_raises_on_none(self) -> None:
        with pytest.raises(ValueError):
            PIDDocument.from_partial(None)


class TestSpatialQueries:
    def setup_method(self) -> None:
        self.doc = PIDDocument(_MINIMAL_SPEC)

    def test_equipment_bbox_known_id(self) -> None:
        bbox = self.doc.equipment_bbox("V-101")
        assert isinstance(bbox, BBox)
        assert bbox.x_min == pytest.approx(10.0)
        assert bbox.y_min == pytest.approx(20.0)
        assert bbox.x_max == pytest.approx(30.0)
        assert bbox.y_max == pytest.approx(35.0)

    def test_equipment_bbox_unknown_id_returns_none(self) -> None:
        assert self.doc.equipment_bbox("DOES-NOT-EXIST") is None

    def test_equipment_bbox_raises_on_none_id(self) -> None:
        with pytest.raises(ValueError):
            self.doc.equipment_bbox(None)

    def test_equipment_position_known_id_is_center(self) -> None:
        pos = self.doc.equipment_position("V-101")
        assert isinstance(pos, Point)
        assert pos.x == pytest.approx(20.0)
        assert pos.y == pytest.approx(27.5)

    def test_equipment_position_unknown_id_returns_none(self) -> None:
        assert self.doc.equipment_position("NONE") is None

    def test_equipment_position_raises_on_none(self) -> None:
        with pytest.raises(ValueError):
            self.doc.equipment_position(None)

    def test_process_bbox_covers_all_equipment(self) -> None:
        bbox = self.doc.process_bbox()
        assert isinstance(bbox, BBox)
        assert bbox.x_min <= 10.0
        assert bbox.x_max >= 60.0

    def test_find_free_region_returns_bbox_with_correct_size(self) -> None:
        result = self.doc.find_free_region(10.0, 10.0)
        if result is not None:
            assert isinstance(result, BBox)
            assert result.width >= 10.0
            assert result.height >= 10.0

    def test_find_free_region_raises_on_none_width(self) -> None:
        with pytest.raises(ValueError):
            self.doc.find_free_region(None, 10.0)


class TestValidation:
    def test_validate_json_returns_list(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        result = doc.validate_json()
        assert isinstance(result, list)

    def test_validate_json_items_have_required_keys(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        for item in doc.validate_json():
            assert "path" in item
            assert "message" in item
            assert "severity" in item

    def test_validate_json_valid_spec_no_errors(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        errors = [i for i in doc.validate_json() if i["severity"] == "error"]
        assert errors == []


class TestExport:
    @requires_ezdxf
    def test_export_dxf_creates_file(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "test.dxf"
            doc.export_dxf(out, sheet_set="one")
            assert out.exists()
            assert out.stat().st_size > 0

    def test_export_dxf_raises_on_none_path(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        with pytest.raises(ValueError):
            doc.export_dxf(None)

    @requires_ezdxf
    def test_export_dxf_two_sheets_creates_process_file(self) -> None:
        doc = PIDDocument(_MINIMAL_SPEC)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "test.dxf"
            doc.export_dxf(out, sheet_set="two")
            assert out.exists()

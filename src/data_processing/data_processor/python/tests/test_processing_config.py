"""Tests for pipeline configuration and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from data_processor.models import FilterConfig, PipelineConfig


def test_pipeline_config_normalizes_and_validates(tmp_path: Path) -> None:
    """Pipeline config should normalize file paths, filters, and output."""
    pipeline = PipelineConfig.from_mapping(
        {
            "files": [tmp_path / "a.csv", tmp_path / "b.csv"],
            "combine": False,
            "selected_signals": ["time", "pressure"],
            "filter": {"filter_type": "Moving Average", "ma_window": 7},
            "output": {"path": tmp_path / "out", "format": "csv"},
        }
    )

    assert pipeline.files == [str(tmp_path / "a.csv"), str(tmp_path / "b.csv")]
    assert pipeline.combine is False
    assert pipeline.selected_signals == ["time", "pressure"]
    assert pipeline.filter is not None
    assert pipeline.filter.parameters["ma_window"] == 7
    assert pipeline.output is not None
    assert pipeline.output.format == "csv"


def test_pipeline_config_requires_files() -> None:
    """Missing files should raise a validation error."""
    with pytest.raises(ValueError):
        PipelineConfig.from_mapping({})


def test_filter_config_rejects_unknown_parameter() -> None:
    """Unknown filter parameters must be rejected to avoid silent ignores."""
    with pytest.raises(ValueError):
        FilterConfig.from_mapping({"filter_type": "Moving Average", "unknown": 1})


def test_filter_config_requires_supported_filter() -> None:
    """Unsupported filters should fail fast."""
    with pytest.raises(ValueError):
        FilterConfig.from_mapping({"filter_type": "NotAFilter"})

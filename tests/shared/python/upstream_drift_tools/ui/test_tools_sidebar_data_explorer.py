from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from upstream_drift_tools.ui.tools_sidebar import WorkspaceRegistry
from upstream_drift_tools.ui.tools_sidebar.data_explorer_service import (
    DataExplorerError,
    DataExplorerService,
)


def test_data_explorer_service_previews_csv_with_bounded_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "temperature": [293.15, None, 295.2],
            "status": ["ok", "warn", "ok"],
        }
    ).to_csv(csv_path, index=False)

    preview = DataExplorerService(project_root=tmp_path, preview_rows=2).preview_file(
        csv_path
    )

    assert preview.format == "csv"
    assert preview.total_rows == 3
    assert preview.total_columns == 2
    assert preview.load_mode == "full"
    assert preview.truncated is True
    assert len(preview.preview_rows) == 2
    assert preview.columns[0].name == "temperature"
    assert preview.columns[0].missing_count == 1
    assert preview.columns[1].dtype == "object"


def test_data_explorer_service_samples_large_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "large.csv"
    pd.DataFrame(
        {
            "index": list(range(20)),
            "value": [number * 2 for number in range(20)],
        }
    ).to_csv(csv_path, index=False)

    preview = DataExplorerService(
        project_root=tmp_path,
        preview_rows=3,
        max_file_size_bytes=32,
    ).preview_file(csv_path)

    assert preview.load_mode == "sampled"
    assert preview.truncated is True
    assert preview.total_rows == 20
    assert [row["index"] for row in preview.preview_rows] == [0, 1, 2]


def test_data_explorer_service_raises_structured_errors(tmp_path: Path) -> None:
    service = DataExplorerService(project_root=tmp_path)

    unsupported_path = tmp_path / "sample.txt"
    unsupported_path.write_text("demo", encoding="utf-8")
    with pytest.raises(DataExplorerError) as unsupported:
        service.preview_file(unsupported_path)
    assert unsupported.value.code == "unsupported_format"

    corrupt_path = tmp_path / "broken.json"
    corrupt_path.write_text("{bad json", encoding="utf-8")
    with pytest.raises(DataExplorerError) as corrupt:
        service.preview_file(corrupt_path)
    assert corrupt.value.code == "read_failed"


def test_data_explorer_service_exports_selected_columns_to_workspace(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "temperature": [293.15, 294.0, 295.2],
            "status": ["ok", "warn", "ok"],
        }
    ).to_csv(csv_path, index=False)
    service = DataExplorerService(project_root=tmp_path, preview_rows=3)
    preview = service.preview_file(csv_path)
    registry = WorkspaceRegistry()

    variable = service.export_selection(
        preview,
        registry,
        "temperature_preview",
        selected_columns=["temperature"],
        row_limit=2,
    )

    assert registry.get("temperature_preview") == [293.15, 294.0]
    assert variable.name == "temperature_preview"
    assert variable.summary == "2"
    assert variable.size == 2

    with pytest.raises(DataExplorerError) as missing_column:
        service.export_selection(
            preview,
            registry,
            "bad_export",
            selected_columns=["missing"],
        )
    assert missing_column.value.code == "unknown_columns"


def test_data_explorer_service_builds_data_processor_handoff(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"temperature": [293.15, 294.0], "status": ["ok", "warn"]}).to_csv(
        csv_path,
        index=False,
    )
    service = DataExplorerService(project_root=tmp_path, preview_rows=2)
    preview = service.preview_file(csv_path)

    request = service.build_data_processor_request(
        preview,
        selected_columns=["temperature"],
        row_limit=1,
    )

    assert request["tool_id"] == "data_processor"
    assert request["source_path"] == str(csv_path)
    assert request["selected_columns"] == ["temperature"]
    assert request["row_limit"] == 1

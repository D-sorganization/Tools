"""Regressions for Data Processor UI work moved off the Qt main thread."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from data_processor.ui.async_workers import DataLoadResult, DataLoadWorker

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PROCESSOR_ROOT = (
    REPO_ROOT / "src/data_processing/data_processor/python/data_processor"
)


class FakeLoader:
    def __init__(self) -> None:
        self.loaded_paths: list[str] = []

    def load_csv_file(self, file_path: str) -> pd.DataFrame:
        self.loaded_paths.append(file_path)
        return pd.DataFrame({"time": [0, 1], "signal": [1.0, 2.0], "label": ["a", "b"]})

    def detect_time_column(self, data: pd.DataFrame) -> str | None:
        return "time" if "time" in data.columns else None

    def convert_time_column(self, data: pd.DataFrame, _time_column: str) -> pd.DataFrame:
        converted = data.copy()
        converted["time"] = pd.to_datetime(converted["time"], unit="s")
        return converted

    def get_numeric_signals(self, data: pd.DataFrame) -> list[str]:
        return list(data.select_dtypes(include="number").columns)


def _class_method_source(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(source, item) or ""
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def test_data_load_worker_prepares_metadata_off_main_thread() -> None:
    loader = FakeLoader()
    worker = DataLoadWorker(
        ["sample.csv"],
        loader,  # type: ignore[arg-type]
        convert_time_column=True,
    )
    results: list[DataLoadResult] = []
    errors: list[str] = []
    worker.result_ready.connect(results.append)
    worker.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert loader.loaded_paths == ["sample.csv"]
    assert len(results) == 1
    assert results[0].time_column == "time"
    assert results[0].available_signals == ["signal"]


def test_main_window_load_slot_delegates_file_io_to_worker() -> None:
    method_source = _class_method_source(
        DATA_PROCESSOR_ROOT / "ui/pyqt6/main_window.py",
        "DataProcessorMainWindow",
        "_load_data",
    )

    assert "DataLoadWorker(" in method_source
    assert ".load_csv_file(" not in method_source
    assert ".load_multiple_files(" not in method_source
    assert "processEvents" not in method_source


def test_filter_slots_delegate_processing_to_worker() -> None:
    main_window_filter = _class_method_source(
        DATA_PROCESSOR_ROOT / "ui/pyqt6/main_window_data_ops.py",
        "DataOperationsMixin",
        "_apply_filter",
    )
    wrapper_filter = _class_method_source(
        DATA_PROCESSOR_ROOT / "pyqt_widget.py",
        "DataProcessorWidget",
        "process_data",
    )

    assert "ProcessingWorker(" in main_window_filter
    assert ".apply_filter(" not in main_window_filter
    assert "AsyncProcessingWorker(" in wrapper_filter
    assert ".apply_filter(" not in wrapper_filter


def test_wrapper_load_slot_delegates_file_io_to_worker() -> None:
    method_source = _class_method_source(
        DATA_PROCESSOR_ROOT / "pyqt_widget.py",
        "DataProcessorWidget",
        "load_file",
    )

    assert "_start_load_file(" in method_source
    assert ".load_csv_file(" not in method_source

"""Unit tests for plotting identity and export plumbing (Issue #4740)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from shared.python.plotting.export import (
    ExportConfig,
    export_all_figures,
    export_figure,
    export_plot_data,
)
from shared.python.plotting.identity import (
    PlotIdentity,
    apply_identity_footer,
    resolve_and_apply_identity_footer,
)


class _EngineStub:
    def __init__(
        self, engine_type: Any = "mujoco", model_name: str | None = "golfer"
    ) -> None:
        self.engine_type = engine_type
        self.model_name = model_name


class _MockRecorder:
    def __init__(self, engine: Any = None) -> None:
        self.engine = engine


@pytest.mark.unit
class TestPlotIdentity:
    def test_default_empty(self) -> None:
        identity = PlotIdentity()
        assert identity.is_empty()
        assert identity.label() is None
        assert identity.as_metadata_dict() == {}

    def test_populated_fields(self) -> None:
        identity = PlotIdentity(
            engine="mujoco",
            model="golfer_v3",
            run_id="run-101",
            version="1.0.0",
        )
        assert not identity.is_empty()
        label = identity.label()
        assert label is not None
        assert "Engine: mujoco" in label
        assert "Model: golfer_v3" in label
        assert "Run: run-101" in label
        assert "Version: 1.0.0" in label

        meta = identity.as_metadata_dict()
        assert meta == {
            "engine": "mujoco",
            "model": "golfer_v3",
            "run_id": "run-101",
            "version": "1.0.0",
        }

    def test_from_recorder(self) -> None:
        stub = _EngineStub(engine_type="drake", model_name="pendulum")
        recorder = _MockRecorder(engine=stub)
        identity = PlotIdentity.from_recorder(recorder, run_id="run-5")
        assert identity.engine == "drake"
        assert identity.model == "pendulum"
        assert identity.run_id == "run-5"


@pytest.mark.unit
class TestApplyIdentityFooter:
    def test_noop_for_none_identity(self) -> None:
        fig, _ = plt.subplots()
        apply_identity_footer(fig, None)
        assert len(fig.texts) == 0
        plt.close(fig)

    def test_noop_for_empty_identity(self) -> None:
        fig, _ = plt.subplots()
        apply_identity_footer(fig, PlotIdentity())
        assert len(fig.texts) == 0
        plt.close(fig)

    def test_renders_populated_identity(self) -> None:
        fig, _ = plt.subplots()
        apply_identity_footer(fig, PlotIdentity(engine="drake", model="pendulum"))
        assert len(fig.texts) == 1
        assert "drake" in fig.texts[0].get_text()
        assert "pendulum" in fig.texts[0].get_text()
        plt.close(fig)

    def test_resolve_and_apply(self) -> None:
        fig, _ = plt.subplots()
        stub = _EngineStub(engine_type="pinocchio", model_name="robot")
        recorder = _MockRecorder(engine=stub)
        resolved = resolve_and_apply_identity_footer(fig, recorder, None)
        assert resolved.engine == "pinocchio"
        assert len(fig.texts) == 1
        plt.close(fig)


@pytest.mark.unit
class TestExportPlumbing:
    def test_export_figure_png(self, tmp_path: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        identity = PlotIdentity(engine="mujoco", model="humanoid", version="1.0")
        config = ExportConfig(output_dir=tmp_path, dpi=100)
        saved = export_figure(
            fig, "test_fig", config=config, formats=["png"], identity=identity
        )
        assert len(saved) == 1
        assert saved[0].exists()

        info = Image.open(saved[0]).info
        assert info.get("engine") == "mujoco"
        assert info.get("model") == "humanoid"
        assert info.get("version") == "1.0"
        plt.close(fig)

    def test_export_plot_data_json(self, tmp_path: Path) -> None:
        data = {"x": np.array([1, 2, 3]), "y": [4, 5, 6]}
        identity = PlotIdentity(engine="mujoco", run_id="r1")
        config = ExportConfig(output_dir=tmp_path)
        out = export_plot_data(
            data, "test_data", config=config, fmt="json", identity=identity
        )
        assert out.exists()

        import json

        payload = json.loads(out.read_text())
        assert "_meta" in payload
        assert payload["_meta"]["engine"] == "mujoco"
        assert payload["_meta"]["run_id"] == "r1"
        assert payload["x"] == [1, 2, 3]

    def test_export_plot_data_csv(self, tmp_path: Path) -> None:
        data = {"col1": [1.0, 2.0], "col2": [3.0, 4.0]}
        identity = PlotIdentity(engine="drake", model="arm", run_id="run-7")
        config = ExportConfig(output_dir=tmp_path)
        out = export_plot_data(
            data, "test_csv", config=config, fmt="csv", identity=identity
        )
        assert out.exists()

        content = out.read_text()
        assert "# engine: drake" in content
        assert "# model: arm" in content
        assert "# run_id: run-7" in content
        assert "col1,col2" in content

    def test_export_all_figures(self, tmp_path: Path) -> None:
        fig1, _ = plt.subplots()
        fig2, _ = plt.subplots()
        config = ExportConfig(output_dir=tmp_path, dpi=100)
        saved_dict = export_all_figures(
            {"f1": fig1, "f2": fig2},
            config=config,
            formats=["png"],
            identity=PlotIdentity(engine="mujoco"),
        )
        assert "f1" in saved_dict
        assert "f2" in saved_dict
        assert saved_dict["f1"][0].exists()
        assert saved_dict["f2"][0].exists()
        plt.close(fig1)
        plt.close(fig2)

"""TDD coverage for optimizer GUI swingset and chain tabs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def test_gui_registration_describes_movement_optimizer() -> None:
    """The launcher metadata presents the expanded movement optimizer."""
    from optimizer_gui import gui_registration

    info = gui_registration.get_gui_info()

    assert info["name"] == "Movement Optimizer"
    assert info["catalog_visible"] is False
    assert "swingset" in info["description"].lower()
    assert "chain" in info["description"].lower()


def test_main_window_registers_motion_tabs_with_qt_mocks(monkeypatch: Any) -> None:
    """The optimizer window adds separate swingset and chain analysis tabs."""
    from optimizer_gui.ui.pyqt6 import main_window

    added_tabs: list[str] = []

    class FakeTabWidget:
        def addTab(self, _widget: object, label: str) -> None:
            added_tabs.append(label)

    window = main_window.OptimizerWindow.__new__(main_window.OptimizerWindow)
    window.tab_widget = FakeTabWidget()
    monkeypatch.setattr(main_window.OptimizerWindow, "_create_parameters_tab", object)
    monkeypatch.setattr(
        main_window.OptimizerWindow,
        "_create_adam_settings_tab",
        object,
    )
    monkeypatch.setattr(main_window.OptimizerWindow, "_create_results_tab", object)
    monkeypatch.setattr(main_window, "create_swingset_tab", lambda: object())
    monkeypatch.setattr(main_window, "create_chain_tab", lambda: object())

    window._add_optimizer_tabs()

    assert added_tabs == [
        "Parameters",
        "Adam Settings",
        "Results",
        "Swingset Model",
        "Chain Dynamics",
    ]


def test_tools_json_has_movement_optimizer_tile() -> None:
    """Tools launcher tile points to the canonical Movement Optimizer app."""
    tools_json = Path(__file__).resolve().parents[3] / "tools.json"
    data = json.loads(tools_json.read_text(encoding="utf-8"))
    optimizer_entries = data["Optimization"]

    tile = next(
        entry
        for entry in optimizer_entries
        if entry["path"] == "src/movement_optimizer/launch_pyqt6.py"
    )

    assert tile["name"] == "Movement Optimizer"
    assert "biomechanics" in tile["desc"].lower()


def test_optimizer_gui_model_pack_is_retired() -> None:
    """Provider discovery must not advertise the old optimizer_gui surface."""
    manifest_path = Path(__file__).resolve().parents[1] / "model_pack.yaml"
    assert not manifest_path.exists()

    root_manifest = Path(__file__).resolve().parents[3] / "model_pack.yaml"
    root_text = root_manifest.read_text(encoding="utf-8")
    assert 'id: "tools_movement_optimizer"' in root_text
    assert 'path: "src/movement_optimizer/launch_pyqt6.py"' in root_text

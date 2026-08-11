"""Native combined regional-ground variation File-command tests."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QWidget  # noqa: E402

from rate_of_closure.ui.pyqt6 import (  # noqa: E402
    regional_ground_variation_request_io as request_io,
)
from rate_of_closure.variation.regional_ground_variation import (  # noqa: E402
    GROUND_NORMAL_RESTITUTION_KEY,
    GROUND_ROLLING_RESISTANCE_KEY,
    GroundRegionalVariationRequest,
    register_ground_variation_variables,
)
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (  # noqa: E402
    _plan,
)
from shared.python.swing_sim.variation import NoiseSpec, VariationPlan  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _request() -> GroundRegionalVariationRequest:
    register_ground_variation_variables()
    plan = VariationPlan(
        mode="launch",
        base_variables={
            GROUND_NORMAL_RESTITUTION_KEY: 0.4,
            GROUND_ROLLING_RESISTANCE_KEY: 0.04,
        },
        noise=(
            NoiseSpec(
                GROUND_ROLLING_RESISTANCE_KEY,
                "uniform",
                0.02,
                0.02,
                0.08,
                "rolling-resistance",
            ),
        ),
        n_runs=4,
        seed=1729,
    )
    return GroundRegionalVariationRequest(
        plan, _plan(), "study", "pytest/native-file-controls", 8, "driver"
    )


class _Host(QWidget):
    def __init__(self, request: GroundRegionalVariationRequest) -> None:
        super().__init__()
        self.request = request
        self.applied: GroundRegionalVariationRequest | None = None
        self.messages: list[tuple[str, bool]] = []

    def current_regional_ground_variation_request(
        self,
    ) -> GroundRegionalVariationRequest:
        return self.request

    def apply_regional_ground_variation_request(
        self, request: GroundRegionalVariationRequest
    ) -> None:
        self.applied = request

    def show_regional_ground_variation_file_status(
        self, message: str, *, error: bool
    ) -> None:
        self.messages.append((message, error))


def test_open_validates_before_applying_and_reports_visible_success(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    request = _request()
    source = tmp_path / "request.json"
    source.write_text(
        request_io.regional_ground_variation_request_to_json(request),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        request_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(source), "JSON files (*.json)"),
    )
    host = _Host(request)
    qtbot.addWidget(host)
    actions = request_io.RegionalGroundVariationRequestFileActions(host, host)

    actions.open()

    assert host.applied == request
    assert actions.recent_path == source
    assert host.messages == [("Opened request.json. No physics executed.", False)]


def test_save_validates_before_dialog_and_writes_atomic_canonical_bytes(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    request = _request()
    target = tmp_path / "saved.json"
    monkeypatch.setattr(
        request_io.QFileDialog,
        "getSaveFileName",
        lambda *_args: (str(target), "JSON files (*.json)"),
    )
    host = _Host(request)
    qtbot.addWidget(host)
    actions = request_io.RegionalGroundVariationRequestFileActions(host, host)

    actions.save_as()

    assert target.read_text(encoding="utf-8") == (
        request_io.regional_ground_variation_request_to_json(request)
    )
    assert actions.recent_path == target
    assert host.messages == [
        ("Saved saved.json atomically. No physics executed.", False)
    ]


def test_cancelled_dialogs_are_no_ops(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        request_io.QFileDialog, "getOpenFileName", lambda *_args: ("", "")
    )
    monkeypatch.setattr(
        request_io.QFileDialog, "getSaveFileName", lambda *_args: ("", "")
    )
    host = _Host(_request())
    qtbot.addWidget(host)
    actions = request_io.RegionalGroundVariationRequestFileActions(host, host)

    actions.open()
    actions.save_as()

    assert host.applied is None
    assert host.messages == []
    assert actions.recent_path is None


def test_corrupt_open_and_invalid_snapshot_fail_closed(
    qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    monkeypatch.setattr(
        request_io.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(corrupt), "JSON files (*.json)"),
    )
    host = _Host(_request())
    qtbot.addWidget(host)
    actions = request_io.RegionalGroundVariationRequestFileActions(host, host)

    actions.open()
    assert host.applied is None
    assert host.messages[-1][1]
    assert "open failed" in host.messages[-1][0].lower()

    def invalid_snapshot() -> GroundRegionalVariationRequest:
        raise ValueError("regional plan must be explicitly validated")

    host.current_regional_ground_variation_request = invalid_snapshot
    actions.save_as()
    assert host.messages[-1] == (
        "Save failed: regional plan must be explicitly validated",
        True,
    )

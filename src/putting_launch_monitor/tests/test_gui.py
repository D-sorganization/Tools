"""The PyQt6 window, offscreen.

The window is built with and without a saved calibration; the wizard is
driven by scripted clicks and must produce the same :class:`Calibration`
a hand would build from the same numbers; synthetic :class:`FrameEvent`
objects update the readout; the GSPro panel mirrors a fake client; the
monitor runs end to end on a rendered putt through the worker thread; and
no module in the UI package carries a colour literal.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("cv2")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from putting_launch_monitor.calibration import Calibration  # noqa: E402
from putting_launch_monitor.detect import Frame, HsvRange  # noqa: E402
from putting_launch_monitor.geometry import Launch  # noqa: E402
from putting_launch_monitor.gspro import GsproClient  # noqa: E402
from putting_launch_monitor.monitor import FrameEvent  # noqa: E402
from putting_launch_monitor.track import Phase, Putt  # noqa: E402
from putting_launch_monitor.ui.pyqt6 import main_window as mw  # noqa: E402
from putting_launch_monitor.ui.pyqt6.calibration_wizard import (  # noqa: E402
    CalibrationWizard,
)
from putting_launch_monitor.ui.pyqt6.gspro_panel import GsproPanel  # noqa: E402
from putting_launch_monitor.ui.pyqt6.hsv_tuner import HsvTuner  # noqa: E402
from putting_launch_monitor.ui.pyqt6.worker import EventBridge  # noqa: E402
from shared.python.camera import CaptureMode  # noqa: E402

from .synthetic import MAT_L_MM, MAT_W_MM, SyntheticCamera, render_frame  # noqa: E402
from .test_gspro import FakeSocket  # noqa: E402
from .test_pipeline import RenderedPuttSource  # noqa: E402

pytestmark = pytest.mark.unit

# A dropped QApplication aborts the process silently under pytest: hold it.
_APP = QApplication.instance() or QApplication([])

CAMERA_ID = "USB\\VID_32E4&PID_5234&MI_00\\9&2A7EE39F&0&0000"
UI_DIR = Path(mw.__file__).parent


def _calibration(cam: SyntheticCamera) -> Calibration:
    corners = tuple((float(x), float(y)) for x, y in cam.mat_corners_px())
    return Calibration(
        camera_instance_id=CAMERA_ID,
        mat_corners_px=corners,
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        capture_width=cam.width,
        capture_height=cam.height,
    )


def _event(
    sequence: int, phase: Phase, putt: Putt | None = None, outcome: str = ""
) -> FrameEvent:
    return FrameEvent(sequence, sequence * 16_666_667, None, phase, putt, outcome)


def _putt(speed_mps: float = 1.0, hla: float = 1.5, accepted: bool = True) -> Putt:
    launch = Launch(speed_mps, (0.0, 1.0), points=8, span_mm=300.0, r2=0.995)
    return Putt(launch, hla, (0.0, 0.0), accepted, "" if accepted else "noisy")


@pytest.fixture
def cam() -> SyntheticCamera:
    return SyntheticCamera()


@pytest.fixture
def saved(cam: SyntheticCamera, tmp_path: Path) -> Path:
    path = tmp_path / "calibration.json"
    _calibration(cam).save(path)
    return path


# -- window ----------------------------------------------------------------------------
def test_window_builds_without_a_calibration(qtbot: Any, tmp_path: Path) -> None:
    window = mw.PuttingMonitorWindow(tmp_path / "missing.json")
    qtbot.addWidget(window)
    assert window.calibration is None
    assert not window.start_button.isEnabled()
    assert "No calibration" in window.calibration_label.text()
    assert window.live_view.region is None


def test_window_loads_the_saved_calibration(qtbot: Any, saved: Path) -> None:
    window = mw.PuttingMonitorWindow(saved)
    qtbot.addWidget(window)
    assert window.calibration is not None
    assert window.camera_id.text() == CAMERA_ID
    assert window.start_button.isEnabled()
    assert window.live_view.region == window._scaled().search_region()
    assert "1220 x 1520" in window.calibration_label.text()


def test_every_control_has_a_tooltip(qtbot: Any, saved: Path) -> None:
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractSlider,
        QAbstractSpinBox,
        QComboBox,
        QLineEdit,
    )

    window = mw.PuttingMonitorWindow(saved)
    qtbot.addWidget(window)
    kinds = (QAbstractButton, QAbstractSpinBox, QLineEdit, QComboBox, QAbstractSlider)
    bare = [
        w.objectName() or type(w).__name__
        for w in window.centralWidget().findChildren(QtWidgets.QWidget)
        if isinstance(w, kinds)
        and not w.toolTip()
        and not w.objectName().startswith("qt_")  # Qt's internal sub-widgets
    ]
    assert bare == []


# -- events ----------------------------------------------------------------------------
def test_observer_events_update_the_readout_and_live_view(
    qtbot: Any, saved: Path, cam: SyntheticCamera
) -> None:
    window = mw.PuttingMonitorWindow(saved)
    qtbot.addWidget(window)
    window.on_event(_event(3, Phase.ARMED))
    assert window.readout.phase.text() == "armed"
    assert window.readout.frames.text() == "3"
    assert window.readout.speed.text() == "—"
    frame = render_frame(cam, (610.0, 700.0), noise=2.0)
    assert window.calibration is not None
    seen = window.calibration.detector().detect(frame)
    assert seen is not None
    event = FrameEvent(
        9, 0, seen, Phase.ROLLING, _putt(1.0, 1.5), "logged", (seen,), frame
    )
    window.on_event(event)
    assert window.live_view.image_size == (cam.width, cam.height)
    assert window.live_view.tracked == seen and window.live_view.candidates == (seen,)
    assert window.readout.phase.text() == "rolling"
    assert window.readout.speed.text() == "2.24"
    assert window.readout.hla.text() == "+1.5"
    assert window.readout.points.text() == "8"
    assert window.readout.r2.text() == "0.995"
    assert window.readout.outcome.text() == "logged"
    window.on_event(_event(10, Phase.WAITING, _putt(accepted=False), ""))
    assert window.readout.outcome.text() == "noisy"


def test_bridge_coalesces_frames_but_never_drops_a_putt(qtbot: Any) -> None:
    bridge = EventBridge()
    got: list[FrameEvent] = []
    bridge.arrived.connect(got.append)
    bridge(_event(1, Phase.WAITING))
    bridge(_event(2, Phase.WAITING))  # superseded while the first is pending
    bridge(_event(3, Phase.WAITING, _putt(), "logged"))
    bridge.release()
    bridge(_event(4, Phase.WAITING))
    _APP.processEvents()
    assert [e.sequence for e in got] == [1, 3, 4]
    assert bridge.dropped == 1


def test_monitor_runs_on_the_worker_and_the_putt_reaches_the_window(
    qtbot: Any, saved: Path, cam: SyntheticCamera
) -> None:
    sources: list[RenderedPuttSource] = []

    def factory(cal: Calibration, width: int) -> RenderedPuttSource:
        assert width == cam.width
        sources.append(RenderedPuttSource(cam, speed=1.8, hla=-2.0))
        return sources[-1]

    window = mw.PuttingMonitorWindow(saved, source_factory=factory)
    qtbot.addWidget(window)
    window.decode_width.setValue(cam.width)
    window.start()
    assert window.running and window.start_button.text() == "Stop"
    qtbot.waitUntil(
        lambda: not window.running and window.start_button.text() == "Start",
        timeout=30_000,
    )
    qtbot.waitUntil(lambda: window.readout.speed.text() != "—", timeout=5_000)
    assert sources[0].closed
    assert float(window.readout.speed.text()) == pytest.approx(1.8 * 2.2369, rel=0.03)
    assert float(window.readout.hla.text()) == pytest.approx(-2.0, abs=0.7)
    assert window.readout.outcome.text().startswith("logged")
    assert window.start_button.text() == "Start"
    assert window.live_view.image_size == (cam.width, cam.height)


# -- wizard ----------------------------------------------------------------------------
def test_wizard_matches_a_hand_built_calibration(
    qtbot: Any, cam: SyntheticCamera, tmp_path: Path
) -> None:
    frame = render_frame(cam, (610.0, 700.0), noise=2.0)
    asked: list[tuple[str, CaptureMode]] = []

    def snapshot(camera_id: str, mode: CaptureMode) -> Frame:
        asked.append((camera_id, mode))
        return frame

    out = tmp_path / "wizard.json"
    wizard = CalibrationWizard(out, snapshot=snapshot)
    qtbot.addWidget(wizard)
    wizard.show()
    page = wizard.camera_page
    page.camera_id.setText(CAMERA_ID)
    page.width_px.setValue(cam.width)
    page.height_px.setValue(cam.height)
    assert not page.isComplete()
    page.take()
    assert asked == [(CAMERA_ID, CaptureMode(cam.width, cam.height, 60))]
    assert page.isComplete()
    wizard.next()
    corners = wizard.corners_page
    assert wizard.currentPage() is corners and not corners.isComplete()
    clicks = [(float(x), float(y)) for x, y in cam.mat_corners_px()]
    for x, y in clicks:
        corners.canvas.click_at_image(x, y)
    assert corners.isComplete() and "4 of 4" in corners.status.text()
    wizard.next()
    mat = wizard.mat_page
    assert wizard.currentPage() is mat
    mat.width_mm.setValue(MAT_W_MM)
    mat.length_mm.setValue(MAT_L_MM)
    mat.target_deg.setValue(2.5)
    mat.colour.setCurrentText("orange")
    mat.notes.setText("lab")
    wizard.next()
    review = wizard.review_page
    assert wizard.currentPage() is review
    assert review.error_px.text().endswith("px")
    assert review.canvas.region is not None
    expected = Calibration(
        camera_instance_id=CAMERA_ID,
        mat_corners_px=tuple(clicks),
        mat_width_mm=MAT_W_MM,
        mat_length_mm=MAT_L_MM,
        target_deg=2.5,
        colour="orange",
        capture_width=cam.width,
        capture_height=cam.height,
        notes="lab",
    )
    assert wizard.calibration() == expected
    assert float(review.error_px.text().split()[0]) < 0.5
    wizard.accept()
    assert Calibration.load(out) == expected
    assert wizard.result_calibration == expected


def test_wizard_fifth_click_restarts_and_seeds_from_an_existing_calibration(
    qtbot: Any, cam: SyntheticCamera, tmp_path: Path
) -> None:
    wizard = CalibrationWizard(tmp_path / "x.json", initial=_calibration(cam))
    qtbot.addWidget(wizard)
    assert wizard.camera_page.camera_id.text() == CAMERA_ID
    assert wizard.mat_page.width_mm.value() == MAT_W_MM
    canvas = wizard.corners_page.canvas
    for i in range(5):
        canvas.click_at_image(10.0 * i, 5.0)
    assert canvas.corners == [(40.0, 5.0)]
    with pytest.raises(ValueError, match="four corners"):
        wizard.calibration()


# -- HSV tuning ------------------------------------------------------------------------
def test_hsv_tuner_shows_the_detectors_mask(qtbot: Any, cam: SyntheticCamera) -> None:
    cal = _calibration(cam)
    frame = render_frame(cam, (610.0, 700.0), noise=2.0)
    tuner = HsvTuner(cal.detector(), frame)
    qtbot.addWidget(tuner)
    assert tuner.mask_view.image_size == (cam.width, cam.height)
    assert tuner.hsv_range() == cal.hsv()
    tuner.sliders["v_min"].setValue(255)
    tuner.sliders["v_max"].setValue(200)  # crossed: swapped into order
    assert tuner.hsv_range() == HsvRange(0, 0, 200, 179, 70, 255)
    assert tuner.detector().colour == tuner.hsv_range()


def test_window_persists_a_tuned_colour(qtbot: Any, saved: Path) -> None:
    window = mw.PuttingMonitorWindow(saved)
    qtbot.addWidget(window)
    window.apply_colour(HsvRange(5, 10, 20, 30, 40, 50))
    assert Calibration.load(saved).custom_colour == HsvRange(5, 10, 20, 30, 40, 50)
    assert window.calibration is not None
    assert window.calibration.hsv() == HsvRange(5, 10, 20, 30, 40, 50)


# -- GSPro -----------------------------------------------------------------------------
def test_gspro_panel_reflects_a_fake_client(qtbot: Any) -> None:
    player = (
        b'{"Code":201,"Message":"GSPro Player Information",'
        b'"Player":{"Handed":"RH","Club":"PT"}}'
    )
    sock = FakeSocket([b'{"Code":200,"Message":"OK"}' + player, player])
    made: list[tuple[str, int]] = []

    def factory(host: str, port: int) -> GsproClient:
        made.append((host, port))
        return GsproClient(device_id="cam", socket_factory=lambda: sock)

    panel = GsproPanel(factory)
    qtbot.addWidget(panel)
    assert panel.status.text() == "disconnected" and panel.mode.text() == "not putting"
    assert not isinstance(panel.sink_for_monitor(), type(panel.sink))
    panel.port.setValue(1250)
    panel.open_connection()
    assert made == [("127.0.0.1", 1250)]
    assert panel.connected and panel.connect_button.text() == "Disconnect"
    assert panel.reply.text() == "201 GSPro Player Information"  # the glued 201
    assert panel.club.text() == "PT" and panel.mode.text() == "PUTTING"
    assert panel.sink is panel.sink_for_monitor()
    assert panel.sink is not None and not panel.sink.always
    panel.always.setChecked(True)
    assert panel.sink.always
    panel.heartbeat()
    assert panel.reply.text().startswith("201")
    panel.toggle()
    assert not panel.connected and sock.closed
    assert panel.status.text() == "disconnected"
    assert panel.connect_button.text() == "Connect"


def test_gspro_panel_reports_a_refused_connection(qtbot: Any) -> None:
    def factory(host: str, port: int) -> GsproClient:
        def refuse() -> None:
            raise OSError("refused")

        return GsproClient(device_id="cam", socket_factory=refuse)

    panel = GsproPanel(factory)
    qtbot.addWidget(panel)
    panel.open_connection()
    assert not panel.connected and panel.status.text() == "failed: refused"


# -- house rules -----------------------------------------------------------------------
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")  # six hex digits: a pasted colour
_RGB = re.compile(r"QColor\(\s*\d|rgb\(|rgba\(|Qt\.GlobalColor")


@pytest.mark.parametrize("module", sorted(p.name for p in UI_DIR.glob("*.py")))
def test_ui_modules_carry_no_colour_literals(module: str) -> None:
    source = (UI_DIR / module).read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert not _HEX.search(source), module
    assert not _RGB.search(code), module
    assert "print(" not in code, module

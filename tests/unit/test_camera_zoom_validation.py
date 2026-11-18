"""Camera zoom boundary validation must precede viewport side effects."""

from __future__ import annotations

import pytest

from shared.python.ui.camera_controls.controls import CameraViewportMixin
from shared.python.ui.camera_controls.models import CameraState, Vector3

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _Viewport(CameraViewportMixin):
    """Headless viewport that records render and preference notifications."""

    def __init__(self, *, auto_fit: bool = False) -> None:
        self._camera_state = CameraState(zoom=1.25, auto_fit_enabled=auto_fit)
        self._camera_controls_widget = None
        self.preferences: list[CameraState] = []
        self._camera_preference_listener = self.preferences.append
        self.renders = 0

    def _camera_subject_m(self) -> Vector3:
        return (0.0, 0.0, 0.0)

    def _camera_base_half_extent_m(self) -> float:
        return 1.0

    def _camera_subject_radius_m(self) -> float:
        return 0.5

    def _camera_state_changed(self) -> None:
        self.renders += 1


@pytest.mark.parametrize("zoom", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("auto_fit", [False, True])
def test_nonfinite_zoom_leaves_viewport_and_preferences_unchanged(
    zoom: float, auto_fit: bool
) -> None:
    """Invalid input must neither render nor persist a replacement state."""
    viewport = _Viewport(auto_fit=auto_fit)
    initial = viewport.camera_state()
    with pytest.raises(ValueError, match="finite"):
        viewport.set_camera_zoom(zoom)
    assert viewport.camera_state() is initial
    assert viewport.renders == 0
    assert viewport.preferences == []


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(-100.0, 0.25), (0.25, 0.25), (2.0, 2.0), (8.0, 8.0), (100.0, 8.0)],
)
def test_finite_zoom_preserves_clamping_and_notification(
    requested: float, expected: float
) -> None:
    """Finite zoom requests retain existing limits and one notification."""
    viewport = _Viewport()
    viewport.set_camera_zoom(requested)
    assert viewport.camera_zoom() == expected
    assert viewport.renders == 1
    assert viewport.preferences == [viewport.camera_state()]


def test_finite_zoom_still_respects_auto_fit_clearance() -> None:
    """The existing radius-based clearance remains active after validation."""
    viewport = _Viewport(auto_fit=True)
    viewport.set_camera_zoom(8.0)
    assert viewport.camera_zoom() == pytest.approx(1.68)
    assert viewport.renders == 1

"""Persisted PyQt controls for independently selectable impact layers."""

from __future__ import annotations

import json
from collections.abc import Callable

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QCheckBox

_SETTINGS_ORG = "RateOfClosure"
_SETTINGS_APP = "ImpactScene"
_SETTINGS_KEY = "visible_layers_v1"
_DEFAULT_LAYERS = frozenset(
    {"face_normal", "face_center_travel", "dplane_normal", "spin_loft_sector"}
)
_LAYER_DEFINITIONS = (
    (
        "face_normal",
        "Face Normal",
        "Show the delivered face-center normal in the app frame.",
    ),
    (
        "face_center_travel",
        "Face-Center Travel",
        "Show rigid-body face-center travel including omega cross r.",
    ),
    (
        "dplane_normal",
        "D-Plane Normal",
        "Show the normal to the plane spanned by face-center travel and normal.",
    ),
    (
        "spin_loft_sector",
        "Spin Loft",
        "Show the shaded exact 3D angle between face-center travel and normal.",
    ),
)


class ImpactLayerControls:
    """Own impact-layer checkboxes and their validated persistent state."""

    def __init__(
        self,
        settings: QSettings | None,
        on_changed: Callable[[], None],
    ) -> None:
        self._settings = settings or QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._on_changed = on_changed
        saved_layers = self._load()
        self.checks: dict[str, QCheckBox] = {}
        for key, label, explanation in _LAYER_DEFINITIONS:
            check = QCheckBox(label)
            check.setChecked(key in saved_layers)
            check.setToolTip(
                "Suggested range: on for engineering review; turn off to isolate "
                f"other layers. {explanation} Source: standard 3D D-plane "
                "geometry. Frame: x target, y up, z right."
            )
            check.toggled.connect(self._persist_and_notify)
            self.checks[key] = check

    def visible_layers(self) -> frozenset[str]:
        """Return the checked layer identifiers as an immutable set."""
        return frozenset(key for key, check in self.checks.items() if check.isChecked())

    def _load(self) -> frozenset[str]:
        raw = self._settings.value(_SETTINGS_KEY)
        if not isinstance(raw, str):
            return _DEFAULT_LAYERS
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return _DEFAULT_LAYERS
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value in _DEFAULT_LAYERS for value in values
        ):
            return _DEFAULT_LAYERS
        return frozenset(values)

    def _persist_and_notify(self, _checked: bool) -> None:
        self._settings.setValue(
            _SETTINGS_KEY,
            json.dumps(sorted(self.visible_layers())),
        )
        self._on_changed()

"""PyQt parity coverage for localized paired-attribution presentation."""

from __future__ import annotations

import json
from pathlib import Path

from rate_of_closure.ui.pyqt6.localized_attribution_view import (
    LocalizedAttributionView,
)
from rate_of_closure.variation.localized_attribution import (
    attribution_authority_from_dict,
    attribution_view_from_json,
)

FIXTURE = Path(__file__).parent / "fixtures" / "localized_attribution_authority_v1.json"


def _authority():  # type: ignore[no-untyped-def]
    return attribution_authority_from_dict(
        json.loads(FIXTURE.read_text(encoding="utf-8"))
    )


def test_view_fails_closed_without_retained_pair_authority(qapp) -> None:  # noqa: ARG001
    view = LocalizedAttributionView()

    assert "Attribution unavailable" in view._status.text()
    assert "not substituted" in view._status.text()
    assert not view._source.isEnabled()
    assert not view._raw_export.isEnabled()
    assert view.selected_view() is None
    assert view.raw_csv() is None
    assert view.view_json() is None


def test_view_selects_source_target_pair_and_preserves_unavailability(qapp) -> None:  # noqa: ARG001
    view = LocalizedAttributionView()
    authority = _authority()

    view.set_authority(authority)

    assert view._source.accessibleName() == "Localized attribution source specification"
    assert view._target.accessibleName() == "Localized attribution target"
    assert view._pair.accessibleName() == "Localized attribution retained pair"
    assert "joint.shoulder" in view._locus.text()
    assert "[0.001, 0.003)" in view._locus.text()
    assert "swing.clubhead.reference at 0.002 s" in view._locus.text()
    assert "2/3 available" in view._denominator.text()

    view._target.setCurrentIndex(view._target.findData("impact.clubhead_speed"))
    view._pair.setCurrentIndex(1)

    selected = view.selected_view()
    assert selected is not None
    assert selected.selected.availability.value == "no_impact_unavailable"
    assert selected.selected.perturbed_target_value is None
    assert view._table.item(0, 1).text().startswith("Unavailable")
    assert view._table.item(0, 2).text() == "Unavailable"
    assert "1/3 available" in view._denominator.text()
    assert "no_impact_unavailable" in (view.raw_csv() or "")
    definition = attribution_view_from_json(view.view_json() or "")
    assert definition.target_id == "impact.clubhead_speed"
    assert definition.perturbed_trial_index == 2


def test_replacing_authority_clears_prior_selection_atomically(qapp) -> None:  # noqa: ARG001
    view = LocalizedAttributionView()
    view.set_authority(_authority())
    view._target.setCurrentIndex(view._target.findData("impact.clubhead_speed"))

    view.set_authority(None, "Producer authority is not retained.")

    assert view.authority() is None
    assert view._source.count() == 0
    assert view._target.count() == 0
    assert view._pair.count() == 0
    assert view._status.text() == "Producer authority is not retained."
    assert view._table.item(0, 0).text() == "—"

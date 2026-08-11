"""Hand-calculated mass-property assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.golf_club import (
    ClubComponent,
    ComponentMassProperties,
    ComponentRole,
    RigidTransform,
    assemble_mass_properties,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_ZERO = ((0.0, 0.0, 0.0),) * 3
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _point_component(
    component_id: str,
    role: ComponentRole,
    mass_kg: float,
    x_m: float,
    *,
    translation_y_m: float = 0.0,
) -> ClubComponent:
    frame_id = f"{component_id}.frame"
    return ClubComponent(
        mass_properties=ComponentMassProperties(
            component_id=component_id,
            role=role,
            frame_id=frame_id,
            mass_kg=mass_kg,
            center_of_mass_m=(0.0, 0.0, 0.0),
            inertia_at_com_kg_m2=_ZERO,
        ),
        transform_to_club=RigidTransform(
            from_frame_id=frame_id,
            to_frame_id="club.frame",
            translation_m=(x_m, translation_y_m, 0.0),
        ),
    )


def test_two_point_masses_match_hand_calculation() -> None:
    result = assemble_mass_properties(
        (
            _point_component("head", ComponentRole.HEAD, 1.0, 0.0),
            _point_component("grip", ComponentRole.GRIP, 1.0, 2.0),
        ),
        assembly_frame_id="club.frame",
    )

    assert result.total_mass_kg == pytest.approx(2.0)
    assert result.center_of_mass_m == pytest.approx((1.0, 0.0, 0.0))
    np.testing.assert_allclose(result.inertia_at_com_kg_m2, np.diag([0.0, 2.0, 2.0]))


def test_three_component_fixture_matches_hand_calculation() -> None:
    result = assemble_mass_properties(
        (
            _point_component("head", ComponentRole.HEAD, 1.0, 0.0),
            _point_component("shaft", ComponentRole.SHAFT, 2.0, 1.0),
            _point_component("grip", ComponentRole.GRIP, 1.0, 2.0),
        ),
        assembly_frame_id="club.frame",
    )

    assert result.total_mass_kg == pytest.approx(4.0)
    assert result.center_of_mass_m == pytest.approx((1.0, 0.0, 0.0))
    np.testing.assert_allclose(result.inertia_at_com_kg_m2, np.diag([0.0, 2.0, 2.0]))


def test_component_inertia_is_rotated_into_club_frame() -> None:
    properties = ComponentMassProperties(
        component_id="head",
        role=ComponentRole.HEAD,
        frame_id="head.frame",
        mass_kg=1.0,
        center_of_mass_m=(0.0, 0.0, 0.0),
        inertia_at_com_kg_m2=np.diag([1.0, 2.0, 3.0]),
    )
    component = ClubComponent(
        mass_properties=properties,
        transform_to_club=RigidTransform(
            from_frame_id="head.frame",
            to_frame_id="club.frame",
            rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
    )

    result = assemble_mass_properties((component,), "club.frame")

    np.testing.assert_allclose(result.inertia_at_com_kg_m2, np.diag([2.0, 1.0, 3.0]))


def test_translation_changes_center_but_not_com_inertia() -> None:
    base = (
        _point_component("head", ComponentRole.HEAD, 1.0, 0.0),
        _point_component("grip", ComponentRole.GRIP, 1.0, 2.0),
    )
    translated = (
        _point_component("head", ComponentRole.HEAD, 1.0, 4.0, translation_y_m=-3.0),
        _point_component("grip", ComponentRole.GRIP, 1.0, 6.0, translation_y_m=-3.0),
    )

    first = assemble_mass_properties(base, "club.frame")
    second = assemble_mass_properties(translated, "club.frame")

    assert second.center_of_mass_m == pytest.approx((5.0, -3.0, 0.0))
    np.testing.assert_allclose(second.inertia_at_com_kg_m2, first.inertia_at_com_kg_m2)


def test_assembly_rejects_duplicate_ids_and_frame_mismatch() -> None:
    component = _point_component("head", ComponentRole.HEAD, 1.0, 0.0)
    with pytest.raises(ValueError, match="unique"):
        assemble_mass_properties((component, component), "club.frame")

    with pytest.raises(ValueError, match="assembly_frame_id"):
        assemble_mass_properties((component,), "other.frame")


def test_assembly_rejects_empty_or_wrong_component_container() -> None:
    with pytest.raises(ValueError, match="at least one"):
        assemble_mass_properties((), "club.frame")
    with pytest.raises(TypeError, match="components"):
        assemble_mass_properties("head", "club.frame")  # type: ignore[arg-type]

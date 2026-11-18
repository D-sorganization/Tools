"""Behavioral contracts for the curated moon catalog."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
from solar_system.data.moon_systems import MOONS, moons_by_parent


def test_moons_by_parent_preserves_catalog_order_and_entries() -> None:
    grouped = moons_by_parent()

    assert list(grouped) == list(dict.fromkeys(moon.parent for moon in MOONS))
    assert [moon for moons in grouped.values() for moon in moons] == MOONS
    assert all(
        moon.parent == parent for parent, moons in grouped.items() for moon in moons
    )


def test_moons_by_parent_returns_fresh_group_lists() -> None:
    first = moons_by_parent()
    first["Earth"].clear()

    second = moons_by_parent()

    assert [moon.name for moon in second["Earth"]] == ["Moon"]


def test_moon_catalog_entries_are_immutable_and_physically_valid() -> None:
    names = [moon.name for moon in MOONS]
    assert len(names) == len(set(names))

    for moon in MOONS:
        assert moon.name.strip()
        assert moon.parent.strip()
        assert math.isfinite(moon.elements.semi_major_axis)
        assert moon.elements.semi_major_axis > 0
        assert 0 <= moon.elements.eccentricity < 1
        assert moon.properties.mass > 0
        assert moon.properties.radius > 0
        assert moon.properties.density > 0
        assert all(0 <= component <= 1 for component in moon.properties.color)

    with pytest.raises(FrozenInstanceError):
        MOONS[0].name = "changed"

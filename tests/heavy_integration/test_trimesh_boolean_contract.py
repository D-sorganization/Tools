"""Unit tests for trimesh boolean contract helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.heavy_integration.trimesh_boolean_contract import (
    EXPECTED_TRANSLATED_BOX_INTERSECTION_VOLUME,
    require_trimesh_boolean_backend,
)


def test_require_trimesh_boolean_backend_skips_when_no_engine() -> None:
    trimesh = SimpleNamespace(boolean=SimpleNamespace(engines_available=set()))
    with pytest.raises(pytest.skip.Exception, match="boolean backend not available"):
        require_trimesh_boolean_backend(trimesh, skip=pytest.skip)


def test_require_trimesh_boolean_backend_allows_provisioned_engine() -> None:
    trimesh = SimpleNamespace(boolean=SimpleNamespace(engines_available={"manifold"}))
    require_trimesh_boolean_backend(trimesh, skip=pytest.skip)


def test_intersection_oracle_volume_is_half_cubic_meter() -> None:
    assert EXPECTED_TRANSLATED_BOX_INTERSECTION_VOLUME == pytest.approx(0.5)

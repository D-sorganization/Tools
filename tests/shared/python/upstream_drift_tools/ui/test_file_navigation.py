"""Tests for Sidekick file explorer navigation backend contracts."""

from __future__ import annotations

from pathlib import Path

from upstream_drift_tools.ui.tools_sidebar import (
    CommonLocation,
    FileNavigationController,
)


class StaticLocationsProvider:
    def __init__(self, locations: list[CommonLocation]) -> None:
        self._locations = locations

    def locations(self, project_root: Path) -> list[CommonLocation]:
        return [CommonLocation("Project", project_root, "project"), *self._locations]


def test_navigation_history_and_disabled_states_are_predictable(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    controller = FileNavigationController(tmp_path)

    assert controller.state().can_go_back is False
    assert controller.state().can_go_forward is False
    assert controller.navigate_to(alpha) is True
    assert controller.navigate_to(beta) is True

    assert controller.state().current_path == beta.resolve()
    assert controller.state().can_go_back is True
    assert controller.state().can_go_forward is False
    assert controller.back() is True
    assert controller.state().current_path == alpha.resolve()
    assert controller.state().can_go_forward is True
    assert controller.navigate_to(beta) is True
    assert controller.state().can_go_forward is False


def test_up_navigation_stops_at_project_boundary(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    controller = FileNavigationController(tmp_path, current_path=child)

    assert controller.state().can_go_up is True
    assert controller.up() is True
    assert controller.state().current_path == tmp_path.resolve()
    assert controller.state().can_go_up is False
    assert controller.up() is False
    assert controller.state().current_path == tmp_path.resolve()


def test_host_policy_can_allow_navigation_outside_project(tmp_path: Path) -> None:
    outside = tmp_path.parent
    scoped = FileNavigationController(tmp_path)
    unscoped = FileNavigationController(tmp_path, allow_outside_project=True)

    assert scoped.navigate_to(outside) is False
    assert scoped.state().current_path == tmp_path.resolve()
    assert unscoped.navigate_to(outside) is True
    assert unscoped.state().current_path == outside.resolve()


def test_common_locations_use_injected_provider_and_policy_filter(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path.parent
    missing = tmp_path / "missing"
    provider = StaticLocationsProvider(
        [
            CommonLocation("Inside", inside),
            CommonLocation("Outside", outside),
            CommonLocation("Missing", missing),
            CommonLocation("Inside duplicate", inside),
        ]
    )

    scoped = FileNavigationController(tmp_path, common_locations_provider=provider)
    unscoped = FileNavigationController(
        tmp_path,
        allow_outside_project=True,
        common_locations_provider=provider,
    )

    assert [(item.label, item.path) for item in scoped.common_locations()] == [
        ("Project", tmp_path.resolve()),
        ("Inside", inside.resolve()),
    ]
    assert [(item.label, item.path) for item in unscoped.common_locations()] == [
        ("Project", tmp_path.resolve()),
        ("Inside", inside.resolve()),
        ("Outside", outside.resolve()),
    ]


def test_stale_persisted_path_falls_back_to_project_root(tmp_path: Path) -> None:
    controller = FileNavigationController(
        tmp_path,
        persisted_path=tmp_path / "deleted",
    )

    assert controller.state().current_path == tmp_path.resolve()

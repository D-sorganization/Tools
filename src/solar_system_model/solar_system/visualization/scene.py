"""Solar System scene controller.

Event handling is in :mod:`.scene_event_mixin`.
Rendering logic is in :mod:`.scene_render_mixin`.
"""

from __future__ import annotations

import math
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from utils.compatibility import UTC

from ..core.celestial_body import (
    BodyType,
    CelestialBody,
    Moon,
    Planet,
    Spacecraft,
    Star,
)
from ..core.constants import (
    DWARF_PLANETS,
    INNER_PLANETS,
    PLANET_ORDER,
)
from ..core.time_manager import TimeManager
from ..data.asteroids import MAJOR_ASTEROIDS, generate_belt_particles
from ..data.comets import COMETS
from ..data.famous_missions import FAMOUS_MISSIONS
from ..data.moon_systems import moons_by_parent
from ..data.planet_info import PLANET_DESCRIPTIONS
from ..physics.trajectory_planner import (
    TrajectoryPlanner,
    TransferTrajectory,
    TransferType,
)
from ..ui.widgets import (
    DateTimePicker,
    EducationalInfoPanel,
    HelpOverlay,
    HistoricalEventsPanel,
    ImmersionChecklistPanel,
    MissionListPanel,
    NavigationPanel,
    SettingsPanel,
    SidebarPanel,
    TimeNavigationPanel,
    UnifiedControlPanel,
)
from .camera import CameraMode
from .renderer import Renderer, RenderSettings
from .scene_event_mixin import SceneEventMixin
from .scene_render_mixin import SceneRenderMixin

try:
    import pygame  # noqa: F401

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from OpenGL.GL import glViewport  # noqa: F401

    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False


@dataclass
class ViewState:
    show_inner_planets: bool = True
    show_outer_planets: bool = True
    show_dwarf_planets: bool = True
    show_minor_bodies: bool = True
    show_orbits: bool = True
    show_labels: bool = True
    show_trajectories: bool = True
    show_info_panel: bool = True
    show_help: bool = True  # Show help by default for new users
    focus_inner_system: bool = False
    show_immersion_checklist: bool = True


class SolarSystemScene(SceneEventMixin, SceneRenderMixin):
    """Main scene controller for the Solar System simulation.

    Core state and lifecycle methods live here.
    Event handling is provided by :class:`SceneEventMixin`.
    Rendering is provided by :class:`SceneRenderMixin`.
    """

    def __init__(self, settings: RenderSettings | None = None) -> None:
        """Initialize the Solar System scene.

        Args:
            settings: Optional rendering settings. If None, default settings are used.
        """
        self.settings = settings or RenderSettings()
        self.renderer: Renderer | None = None
        self.time_manager = TimeManager()
        self.trajectory_planner = TrajectoryPlanner()

        # Celestial bodies
        self.sun: Star | None = None
        self.planets: dict[str, Planet] = {}
        self.moons: dict[str, Moon] = {}
        self.asteroids: dict[str, CelestialBody] = {}
        self.comets: dict[str, CelestialBody] = {}
        self.spacecraft: dict[str, Spacecraft] = {}

        # Pre-computed asteroid belt cloud
        self.asteroid_belt_points = generate_belt_particles()

        # Active trajectories
        self.trajectories: list[TransferTrajectory] = []

        # View state
        self.view_state = ViewState()

        # Recent action feedback for status bar
        self._action_message: str = ""

        # Selection
        self.selected_body: CelestialBody | None = None

        # Mouse state
        self._mouse_dragging = False
        self._last_mouse_pos = (0, 0)

        # Control bindings displayed in help
        self.controls = [
            ("MOUSE:", ""),
            ("  Scroll Wheel", "Zoom in/out"),
            ("  Left Drag", "Rotate camera"),
            ("  Right Drag", "Pan camera"),
            ("", ""),
            ("KEYBOARD:", ""),
            ("  SPACE", "Pause/Resume"),
            ("  + / -", "Speed up/slow down time"),
            ("  R", "Reverse time flow"),
            ("  D", "Toggle date picker"),
            ("  N", "Toggle time navigation panel"),
            ("  E", "Toggle historical events"),
            ("  [ / ]", "Jump backward/forward 1 day"),
            ("  PgUp/Dn", "Jump backward/forward 1 month"),
            ("  T", "Plan trip to Mars"),
            ("  M", "Toggle immersion checklist"),
            ("", ""),
            ("  0-9", "Select planet (0=Sun, 3=Earth, 4=Mars)"),
            ("  F", "Focus camera on selected"),
            ("  C", "Cycle camera modes"),
            ("  HOME", "Reset camera view"),
            ("", ""),
            ("  O", "Toggle orbital paths"),
            ("  L", "Toggle planet labels"),
            ("  I", "Toggle info panel"),
            ("  G", "Toggle reference grid"),
            ("  V", "Toggle stereo/VR view"),
            ("  H", "Toggle this help"),
            ("  ESC", "Quit simulation"),
        ]

        # Enhanced UI widgets
        self.date_picker: DateTimePicker | None = None
        self.time_nav_panel: TimeNavigationPanel | None = None

        # Container Panels
        self.sidebar_panel: SidebarPanel | None = None
        self.unified_controls: UnifiedControlPanel | None = None

        # Child panels (kept for logic/data)
        self.educational_panel: EducationalInfoPanel | None = None
        self.historical_events: HistoricalEventsPanel | None = None
        self.immersion_checklist: ImmersionChecklistPanel | None = None
        self.missions_panel: MissionListPanel | None = None

        # Legacy references
        self.settings_panel: SettingsPanel | None = None
        self.nav_mode_panel: NavigationPanel | None = None
        self.help_overlay: HelpOverlay | None = None

        self._last_ui_sync_jd: float | None = None

    # ================================================================
    # Lifecycle
    # ================================================================

    def initialize(self) -> bool:
        """Initialize the scene, renderer, and simulation state.

        Returns:
            True if initialization was successful, False otherwise.
        """
        self.renderer = Renderer(self.settings)
        if not self.renderer.initialize():
            return False

        self._create_solar_system()
        self.time_manager.set_to_now()
        self.time_manager.time_warp = 86400  # 1 day per second
        self._initialize_ui_widgets()
        return True

    def run(self) -> None:
        """Run the main simulation loop.

        Raises:
            RuntimeError: If the scene has not been initialized.
        """
        if not self.renderer:
            raise RuntimeError("Scene not initialized. Call initialize() first.")
        running = True
        while running:
            running = self._handle_events()
            self._update()
            self._render()
        self.renderer.cleanup()

    # ================================================================
    # UI Initialization
    # ================================================================

    def _initialize_ui_widgets(self) -> None:
        """Initialize enhanced UI widgets with modern Unified Layout."""
        if not self.renderer:
            return

        self.date_picker = DateTimePicker(
            position=(20, 100), on_date_change=self._on_date_picker_change
        )
        self.date_picker.set_date(self.time_manager.current_time.datetime_utc)
        self.time_nav_panel = TimeNavigationPanel(position=(0, 0))

        # Sidebar (Right)
        sidebar_height = self.renderer.settings.window_height - 40
        sidebar_x = self.renderer.settings.window_width - 380
        self.sidebar_panel = SidebarPanel(
            position=(sidebar_x, 20), height=sidebar_height
        )

        self.educational_panel = EducationalInfoPanel(width=360)
        self.immersion_checklist = ImmersionChecklistPanel(width=360)
        self.historical_events = HistoricalEventsPanel(width=360)
        self.historical_events.set_date(self.time_manager.current_time.datetime_utc)

        # Unified Controls (Bottom)
        control_height = 180
        self.unified_controls = UnifiedControlPanel(
            position=(0, self.renderer.settings.window_height - control_height),
            width=self.renderer.settings.window_width,
        )
        self.unified_controls.height = control_height

        self.unified_controls.add_checkbox(
            "Show Labels", self.view_state.show_labels, "toggle_labels"
        )
        self.unified_controls.add_checkbox(
            "Show Grid", self.renderer.settings.show_grid, "toggle_grid"
        )
        self.unified_controls.add_checkbox(
            "Stereo View", self.settings.stereo_view, "toggle_stereo"
        )
        self.unified_controls.add_checkbox(
            "Inner Planets", self.view_state.show_inner_planets, "toggle_inner"
        )
        self.unified_controls.add_checkbox(
            "Outer Planets", self.view_state.show_outer_planets, "toggle_outer"
        )
        self.unified_controls.add_checkbox(
            "Dwarf Planets", self.view_state.show_dwarf_planets, "toggle_dwarf"
        )
        self.unified_controls.add_checkbox(
            "Moons/Small", self.view_state.show_minor_bodies, "toggle_moons"
        )

        self.unified_controls.add_button("Reset View", "reset_view")
        self.unified_controls.add_button("Toggle Orbits", "toggle_orbits_btn")
        self.unified_controls.set_mode("Orbit")

        self.help_overlay = HelpOverlay(
            position=(self.renderer.settings.window_width - 350, 20)
        )
        self.help_overlay.set_controls(self.controls)
        self.missions_panel = MissionListPanel()

    # ================================================================
    # Solar System Creation
    # ================================================================

    def _create_solar_system(self) -> None:
        """Create and populate the solar system with celestial bodies."""
        self.sun = Star("Sun")
        for planet_name in PLANET_ORDER:
            is_dwarf = planet_name in DWARF_PLANETS
            planet = Planet(name=planet_name, parent=self.sun, is_dwarf=is_dwarf)
            self.planets[planet_name] = planet

        for parent_name, moon_list in moons_by_parent().items():
            parent_body = self.planets.get(parent_name)
            if not parent_body:
                continue
            for descriptor in moon_list:
                moon = Moon(
                    name=descriptor.name,
                    parent=parent_body,
                    orbital_elements=descriptor.elements,
                    physical_properties=descriptor.properties,
                )
                self.moons[descriptor.name] = moon

        for asteroid in MAJOR_ASTEROIDS:
            asteroid_body = CelestialBody(
                name=asteroid.name,
                body_type=BodyType.ASTEROID,
                orbital_elements=asteroid.elements,
                physical_properties=asteroid.properties,
                parent=self.sun,
            )
            self.asteroids[asteroid.name] = asteroid_body

        for comet in COMETS:
            comet_body = CelestialBody(
                name=comet.name,
                body_type=BodyType.COMET,
                orbital_elements=comet.elements,
                physical_properties=comet.properties,
                parent=self.sun,
            )
            self.comets[comet.name] = comet_body

        self._load_famous_missions()

    def _load_famous_missions(self) -> None:
        """Load historical famous missions into the scene."""
        for name, mission_data in FAMOUS_MISSIONS.items():
            getter = mission_data.get("get_trajectory")
            if getter and callable(getter):
                trajectory = getter()
                craft = Spacecraft(name, trajectory)
                self.spacecraft[name] = craft

    # ================================================================
    # Body Queries & Selection
    # ================================================================

    def get_all_bodies(self) -> list[CelestialBody]:
        """Get a list of all celestial bodies in the scene."""
        bodies: list[CelestialBody] = []
        if self.sun:
            bodies.append(self.sun)
        bodies.extend(self.planets.values())
        bodies.extend(self.moons.values())
        bodies.extend(self.asteroids.values())
        bodies.extend(self.comets.values())
        bodies.extend(self.spacecraft.values())
        return bodies

    def get_body_by_name(self, name: str) -> CelestialBody | None:
        """Retrieve a celestial body by its name."""
        assert name is not None, "name must be provided"
        if name == "Sun":
            return self.sun
        for collection in (
            self.planets,
            self.moons,
            self.asteroids,
            self.comets,
            self.spacecraft,
        ):
            if name in collection:
                return collection[name]
        return None

    def select_body(self, body: CelestialBody) -> None:
        """Select a celestial body in the scene."""
        assert body is not None, "body must be provided"
        self.selected_body = body
        if self.renderer:
            self.renderer.selected_body = body
        self._mark_immersion_task("select_body")

    # ================================================================
    # Trajectory Planning
    # ================================================================

    def plan_trajectory(
        self,
        origin_name: str,
        destination_name: str,
        departure_date: float | None = None,
    ) -> TransferTrajectory | None:
        """Plan a transfer trajectory between two celestial bodies."""
        assert origin_name is not None, "origin_name must be provided"
        origin = self.get_body_by_name(origin_name)
        destination = self.get_body_by_name(destination_name)
        if not origin or not destination:
            return None
        if departure_date is None:
            departure_date = self.time_manager.julian_date

        trajectory = self.trajectory_planner.calculate_transfer(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            transfer_type=TransferType.HOHMANN,
        )
        spacecraft = self.trajectory_planner.create_spacecraft_from_transfer(
            trajectory, name=f"{origin_name}-{destination_name} Transfer"
        )
        self.spacecraft[spacecraft.name] = spacecraft
        self.trajectories.append(trajectory)
        return trajectory

    def get_transfer_summary(
        self, origin_name: str, destination_name: str
    ) -> dict[str, Any] | None:
        """Get a summary of a potential transfer between two bodies."""
        assert origin_name is not None, "origin_name must be provided"
        origin = self.get_body_by_name(origin_name)
        destination = self.get_body_by_name(destination_name)
        if not origin or not destination:
            return None
        return self.trajectory_planner.get_transfer_summary(origin, destination)

    # ================================================================
    # Time & Camera Helpers
    # ================================================================

    def _on_date_picker_change(self, new_date: datetime) -> None:
        """Handle date changes from the date picker."""
        assert new_date is not None, "new_date must be provided"
        if new_date.tzinfo is None:
            new_date = new_date.replace(tzinfo=UTC)
        self.time_manager.set_datetime(new_date)
        if self.historical_events:
            self.historical_events.set_date(new_date)
        self._mark_immersion_task("navigate_time")

    def _update_ui_date(self) -> None:
        """Update all UI widgets with current date."""
        current_dt = self.time_manager.current_time.datetime_utc
        if self.date_picker:
            self.date_picker.set_date(current_dt)
        if self.historical_events:
            self.historical_events.set_date(current_dt)

    def _update_educational_panel(self) -> None:
        """Update educational panel with selected body information."""
        if not self.selected_body or not self.educational_panel:
            return
        body_name = self.selected_body.name
        if body_name in PLANET_DESCRIPTIONS:
            info = cast(dict[str, Any], PLANET_DESCRIPTIONS[body_name])
            properties: dict[str, Any] = {}
            info_dict = dict(info)
            for key, value in info_dict.items():
                if key != "fun_facts":
                    properties[key.replace("_", " ").title()] = value
            fun_facts = info.get("fun_facts", [])
            self.educational_panel.set_body(body_name, properties, fun_facts)
        self._mark_immersion_task("select_body")

    def _mark_immersion_task(self, task_id: str) -> None:
        """Mark an immersion checklist task as complete if available."""
        if self.immersion_checklist:
            self.immersion_checklist.mark_complete(task_id)

    def _jump_time(self, days: float) -> None:
        """Jump time forward or backward by a number of days."""
        assert days is not None, "days must be provided"
        self.time_manager.advance_days(days)
        self._update_ui_date()
        self._mark_immersion_task("navigate_time")

    def _jump_month(self, months: int) -> None:
        """Jump time forward or backward by a number of months."""
        assert months is not None, "months must be provided"
        current_dt = self.time_manager.current_time.datetime_utc
        target_day = current_dt.day
        total_months = current_dt.month + months - 1
        new_year = current_dt.year + (total_months // 12)
        new_month = (total_months % 12) + 1
        max_days = monthrange(new_year, new_month)[1]
        actual_day = min(target_day, max_days)
        new_date = current_dt.replace(year=new_year, month=new_month, day=actual_day)
        self.time_manager.set_datetime(new_date)
        self._update_ui_date()
        self._mark_immersion_task("navigate_time")

    def _handle_time_nav_action(self, action: str) -> None:
        """Handle time navigation panel button actions."""
        assert action is not None, "action must be provided"
        _TIME_NAV_DISPATCH = {
            "prev_day": lambda: self._jump_time(-1),
            "next_day": lambda: self._jump_time(1),
            "prev_week": lambda: self._jump_time(-7),
            "next_week": lambda: self._jump_time(7),
            "prev_month": lambda: self._jump_month(-1),
            "next_month": lambda: self._jump_month(1),
            "faster": self.time_manager.increase_time_warp,
            "slower": self.time_manager.decrease_time_warp,
            "reverse": self.time_manager.reverse_time,
            "toggle_pause": self.time_manager.toggle_pause,
        }

        handler = _TIME_NAV_DISPATCH.get(action)
        if handler:
            handler()  # type: ignore
            return

        if action == "prev_year":
            self.time_manager.advance_years(-1)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")
        elif action == "next_year":
            self.time_manager.advance_years(1)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")
        elif action in ("goto_today", "reset"):
            self.time_manager.set_to_now()
            self._update_ui_date()
        elif action == "goto_j2000":
            self.time_manager.set_to_j2000()
            self._update_ui_date()
        elif action == "goto_j2030":
            if hasattr(self.time_manager, "J2030"):
                self.time_manager.set_datetime(self.time_manager.J2030)
                self._update_ui_date()

    def _cycle_camera_mode(self) -> None:
        """Cycle through camera modes."""
        if not self.renderer:
            return
        camera = self.renderer.camera
        modes = [CameraMode.FREE, CameraMode.HELIOCENTRIC, CameraMode.TOP_DOWN]
        if self.selected_body:
            modes.append(CameraMode.PLANET_CENTRIC)
        current_index = modes.index(camera.mode) if camera.mode in modes else 0
        next_index = (current_index + 1) % len(modes)
        camera.set_mode(modes[next_index], self.selected_body)

    def _focus_on_selected(self) -> None:
        """Focus the camera on the currently selected body."""
        if not self.selected_body or not self.renderer:
            return
        state = self.selected_body.get_state_at_time(self.time_manager.julian_date)
        pos = state.position * self.renderer.distance_scale
        self.renderer.camera.look_at(pos)
        if self.selected_body.body_type == BodyType.STAR:
            self.renderer.camera.set_distance(20)
        elif self.selected_body.name in INNER_PLANETS:
            self.renderer.camera.set_distance(5)
        else:
            self.renderer.camera.set_distance(15)

    def _update(self) -> None:
        """Update the simulation state (time, physics, camera)."""
        delta_jd = self.time_manager.update()
        current_jd = self.time_manager.julian_date

        if (delta_jd or self._last_ui_sync_jd is None) and (
            self._last_ui_sync_jd is None
            or not math.isclose(current_jd, self._last_ui_sync_jd)
        ):
            self._update_ui_date()
            self._last_ui_sync_jd = current_jd

        if self.renderer:
            self.renderer.camera.update(
                self.time_manager.julian_date, self.renderer.distance_scale
            )

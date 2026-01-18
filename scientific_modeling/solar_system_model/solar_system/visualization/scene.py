from __future__ import annotations

import math
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
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
    AU,
    DWARF_PLANETS,
    INNER_PLANETS,
    OUTER_PLANETS,
    PLANET_ORDER,
)
from ..core.time_manager import TimeManager
from ..data.asteroids import MAJOR_ASTEROIDS, generate_belt_particles
from ..data.comets import COMETS
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
    NavigationPanel,
    SettingsPanel,
    SidebarPanel,
    TimeNavigationPanel,
    UnifiedControlPanel,
)
from .camera import CameraMode
from .renderer import Renderer, RenderSettings

try:
    import pygame
    from pygame.locals import (
        K_0,
        K_1,
        K_9,
        K_EQUALS,
        K_ESCAPE,
        K_HOME,
        K_KP_MINUS,
        K_KP_PLUS,
        K_LEFTBRACKET,
        K_MINUS,
        K_PAGEDOWN,
        K_PAGEUP,
        K_PERIOD,
        K_PLUS,
        K_RIGHTBRACKET,
        K_SPACE,
        KEYDOWN,
        MOUSEBUTTONDOWN,
        MOUSEBUTTONUP,
        MOUSEMOTION,
        MOUSEWHEEL,
        QUIT,
        K_c,
        K_d,
        K_e,
        K_f,
        K_g,
        K_h,
        K_i,
        K_l,
        K_m,
        K_n,
        K_o,
        K_r,
        K_t,
        K_v,
    )

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    from OpenGL.GL import GL_DEPTH_BUFFER_BIT, glClear, glViewport

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


class SolarSystemScene:
    """
    Main scene controller for the Solar System simulation.

    Handles initialization, updates, event handling, and rendering coordination
    for celestial bodies, trajectories, and UI elements.
    """

    def __init__(self, settings: RenderSettings | None = None) -> None:
        """
        Initialize the Solar System scene.

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
            ("  T", "Plan trip to Mars 🚀"),
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
        # self.time_nav_panel is now visually inside UnifiedControlPanel
        # but kept for logic
        self.time_nav_panel: TimeNavigationPanel | None = None

        # New Container Panels
        self.sidebar_panel: SidebarPanel | None = None
        self.unified_controls: UnifiedControlPanel | None = None

        # Child panels (kept for logic/data, managed by containers for rendering state)
        self.educational_panel: EducationalInfoPanel | None = None
        self.historical_events: HistoricalEventsPanel | None = None
        self.immersion_checklist: ImmersionChecklistPanel | None = None

        # Legacy references (set to None or removed)
        self.settings_panel: SettingsPanel | None = None
        self.nav_mode_panel: NavigationPanel | None = None

        self.help_overlay: HelpOverlay | None = None

        self._last_ui_sync_jd: float | None = None

    def initialize(self) -> bool:
        """
        Initialize the scene, renderer, and simulation state.

        Returns:
            True if initialization was successful, False otherwise.
        """
        # Create renderer
        self.renderer = Renderer(self.settings)
        if not self.renderer.initialize():
            return False

        # Create celestial bodies
        self._create_solar_system()

        # Set initial time to current date
        self.time_manager.set_to_now()

        # Set initial time warp
        self.time_manager.time_warp = 86400  # 1 day per second

        # Initialize enhanced UI widgets
        self._initialize_ui_widgets()

        return True

    def _initialize_ui_widgets(self) -> None:
        """Initialize enhanced UI widgets with modern Unified Layout."""
        if not self.renderer:
            return

        # Date picker for manual time navigation
        self.date_picker = DateTimePicker(
            position=(20, 100), on_date_change=self._on_date_picker_change
        )
        self.date_picker.set_date(self.time_manager.current_time.datetime_utc)

        # Time navigation logic (reusing existing class for logic handling)
        self.time_nav_panel = TimeNavigationPanel(position=(0, 0))

        # --- Sidebar Panel (Right) ---
        sidebar_height = self.renderer.settings.window_height - 40
        sidebar_x = self.renderer.settings.window_width - 380
        self.sidebar_panel = SidebarPanel(
            position=(sidebar_x, 20), height=sidebar_height
        )

        # Initialize content panels used by Sidebar
        self.educational_panel = EducationalInfoPanel(width=360)
        self.immersion_checklist = ImmersionChecklistPanel(width=360)
        self.historical_events = HistoricalEventsPanel(width=360)
        self.historical_events.set_date(self.time_manager.current_time.datetime_utc)

        # --- Unified Control Panel (Bottom) ---
        control_height = 180  # Increased for granular controls
        self.unified_controls = UnifiedControlPanel(
            position=(0, self.renderer.settings.window_height - control_height),
            width=self.renderer.settings.window_width,
        )
        self.unified_controls.height = control_height

        # Granular Checkboxes
        # View Settings
        self.unified_controls.add_checkbox(
            "Show Labels", self.view_state.show_labels, "toggle_labels"
        )
        self.unified_controls.add_checkbox(
            "Show Grid", self.renderer.settings.show_grid, "toggle_grid"
        )
        self.unified_controls.add_checkbox(
            "Stereo View", self.settings.stereo_view, "toggle_stereo"
        )

        # Orbits - Granular
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

        # Add Action Buttons
        self.unified_controls.add_button("Reset View", "reset_view")
        self.unified_controls.add_button("Toggle Orbits", "toggle_orbits_btn")

        # Set initial Nav Mode
        self.unified_controls.set_mode("Orbit")

        # Help Overlay
        self.help_overlay = HelpOverlay(
            position=(self.renderer.settings.window_width - 350, 20)
        )
        self.help_overlay.set_controls(self.controls)

    def _on_date_picker_change(self, new_date: datetime) -> None:
        """
        Handle date changes from the date picker.

        Args:
            new_date: The new selected date
        """
        # Ensure timezone aware
        if new_date.tzinfo is None:

            new_date = new_date.replace(tzinfo=UTC)

        # Update simulation time
        self.time_manager.set_datetime(new_date)

        # Update historical events
        if self.historical_events:
            self.historical_events.set_date(new_date)

        self._mark_immersion_task("navigate_time")

    def _create_solar_system(self) -> None:
        """Create and populate the solar system with celestial bodies."""
        # Create the Sun
        self.sun = Star("Sun")

        # Create planets
        for planet_name in PLANET_ORDER:
            is_dwarf = planet_name in DWARF_PLANETS
            planet = Planet(name=planet_name, parent=self.sun, is_dwarf=is_dwarf)
            self.planets[planet_name] = planet

        # Create Earth's Moon
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

    def get_all_bodies(self) -> list[CelestialBody]:
        """
        Get a list of all celestial bodies in the scene.

        Returns:
            A list containing the Sun, planets, moons, asteroids, comets, and spacecraft.
        """
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
        """
        Retrieve a celestial body by its name.

        Args:
            name: The name of the body to retrieve.

        Returns:
            The CelestialBody object if found, otherwise None.
        """
        if name == "Sun":
            return self.sun
        if name in self.planets:
            return self.planets[name]
        if name in self.moons:
            return self.moons[name]
        if name in self.asteroids:
            return self.asteroids[name]
        if name in self.comets:
            return self.comets[name]
        if name in self.spacecraft:
            return self.spacecraft[name]
        return None

    def select_body(self, body: CelestialBody) -> None:
        """
        Select a celestial body in the scene.

        Args:
            body: The CelestialBody to select.
        """
        self.selected_body = body
        if self.renderer:
            self.renderer.selected_body = body
        self._mark_immersion_task("select_body")

    def plan_trajectory(
        self,
        origin_name: str,
        destination_name: str,
        departure_date: float | None = None,
    ) -> TransferTrajectory | None:
        """
        Plan a transfer trajectory between two celestial bodies.

        Args:
            origin_name: Name of the origin body.
            destination_name: Name of the destination body.
            departure_date: Optional departure date (Julian Date). If None, uses current time.

        Returns:
            The calculated TransferTrajectory if successful, otherwise None.
        """
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

        # Create spacecraft for the trajectory
        spacecraft = self.trajectory_planner.create_spacecraft_from_transfer(
            trajectory, name=f"{origin_name}-{destination_name} Transfer"
        )

        self.spacecraft[spacecraft.name] = spacecraft
        self.trajectories.append(trajectory)

        return trajectory

    def run(self) -> None:
        """
        Run the main simulation loop.

        This method blocks until the simulation exits.
        Raises:
            RuntimeError: If the scene has not been initialized.
        """
        if not self.renderer:
            raise RuntimeError("Scene not initialized. Call initialize() first.")

        running = True

        while running:
            # Handle events
            running = self._handle_events()

            # Update simulation
            self._update()

            # Render
            self._render()

        # Cleanup
        self.renderer.cleanup()

    def _handle_events(self) -> bool:
        """
        Process all pending Pygame events.

        Returns:
            False if a quit event was received, True otherwise.
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            elif event.type == KEYDOWN:
                if not self._handle_key(event.key):
                    return False

            elif event.type == MOUSEBUTTONDOWN:
                self._handle_mouse_button(event.button, True)

            elif event.type == MOUSEBUTTONUP:
                self._handle_mouse_button(event.button, False)

            elif event.type == MOUSEMOTION:
                self._handle_mouse_motion(event.pos, event.rel)

            elif event.type == MOUSEWHEEL:
                self._handle_mouse_wheel(event.y)

        return True

    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.

        Returns:
            False if should quit, True otherwise
        """
        if key == K_ESCAPE:
            return False

        if not self.renderer:
            return True

        elif key == K_SPACE:
            self.time_manager.toggle_pause()

        elif key in (K_EQUALS, K_PLUS, K_KP_PLUS):
            self.time_manager.increase_time_warp()

        elif key in (K_MINUS, K_KP_MINUS):
            self.time_manager.decrease_time_warp()

        elif key == K_r:
            self.time_manager.reverse_time()

        elif key == K_d:
            # Toggle date picker
            if self.date_picker:
                self.date_picker.toggle()
                if self.date_picker.visible:
                    self.date_picker.set_date(
                        self.time_manager.current_time.datetime_utc
                    )
                    self._mark_immersion_task("navigate_time")

        elif key == K_n:
            # Toggle time navigation panel
            if self.time_nav_panel:
                self.time_nav_panel.toggle()
                self._mark_immersion_task("navigate_time")

        elif key == K_e:
            # Toggle historical events panel
            if self.historical_events:
                self.historical_events.toggle()
                if self.historical_events.visible:
                    self._mark_immersion_task("historical_events")

        elif key == K_LEFTBRACKET:
            # Jump backward 1 day
            self.time_manager.advance_days(-1)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")

        elif key == K_RIGHTBRACKET:
            # Jump forward 1 day
            self.time_manager.advance_days(1)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")

        elif key == K_PAGEUP:
            # Jump backward 1 month, preserving day of month when possible
            current_dt = self.time_manager.current_time.datetime_utc
            target_day = current_dt.day

            # Calculate previous month
            if current_dt.month == 1:
                prev_month = 12
                prev_year = current_dt.year - 1
            else:
                prev_month = current_dt.month - 1
                prev_year = current_dt.year

            # Ensure day exists in previous month (handle cases like Jan 31 -> Dec 31)
            max_days_in_prev = monthrange(prev_year, prev_month)[1]
            actual_day = min(target_day, max_days_in_prev)

            prev_date = current_dt.replace(
                year=prev_year, month=prev_month, day=actual_day
            )
            self.time_manager.set_datetime(prev_date)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")

        elif key == K_PAGEDOWN:
            # Jump forward 1 month, preserving day of month when possible
            current_dt = self.time_manager.current_time.datetime_utc
            target_day = current_dt.day

            # Calculate next month
            if current_dt.month == 12:
                next_month = 1
                next_year = current_dt.year + 1
            else:
                next_month = current_dt.month + 1
                next_year = current_dt.year

            # Ensure day exists in next month (handle cases like Jan 31 -> Feb 28/29)
            max_days_in_next = monthrange(next_year, next_month)[1]
            actual_day = min(target_day, max_days_in_next)

            next_date = current_dt.replace(
                year=next_year, month=next_month, day=actual_day
            )
            self.time_manager.set_datetime(next_date)
            self._update_ui_date()
            self._mark_immersion_task("navigate_time")

        elif key == K_HOME:
            self.renderer.camera.reset()
            self.renderer.camera.mode = CameraMode.FREE

        elif key == K_o:
            self.view_state.show_orbits = not self.view_state.show_orbits
            self.renderer.settings.show_orbits = self.view_state.show_orbits
            self._mark_immersion_task("toggle_overlays")

        elif key == K_l:
            self.view_state.show_labels = not self.view_state.show_labels
            self.renderer.settings.show_labels = self.view_state.show_labels
            self._mark_immersion_task("toggle_overlays")

        elif key == K_i:
            self.view_state.show_info_panel = not self.view_state.show_info_panel

        elif key == K_g:
            self.renderer.settings.show_grid = not self.renderer.settings.show_grid
            self._mark_immersion_task("toggle_overlays")

        elif key == K_h:
            self.view_state.show_help = not self.view_state.show_help

        elif key == K_v:
            self.settings.stereo_view = not self.settings.stereo_view

        elif key == K_c:
            self._cycle_camera_mode()

        elif key == K_f:
            self._focus_on_selected()

        elif key == K_t:
            # Plan trajectory to Mars from Earth
            trajectory = self.plan_trajectory("Earth", "Mars")
            if trajectory:
                self._mark_immersion_task("plan_transfer")
                self._action_message = (
                    "Earth→Mars transfer: ΔV "
                    f"{trajectory.total_delta_v/1000:.2f} km/s, "
                    f"flight {trajectory.time_of_flight:.1f} days"
                )
            else:
                self._action_message = "Earth→Mars transfer could not be created"

        elif key == K_m:
            if self.immersion_checklist:
                self.immersion_checklist.toggle()
            self.view_state.show_immersion_checklist = (
                not self.view_state.show_immersion_checklist
            )

        # Period/comma for cycling fun facts
        elif key == K_PERIOD:
            if self.educational_panel and self.educational_panel.visible:
                self.educational_panel.cycle_fact()

        # Number keys for planet selection
        elif key == K_0:
            if self.sun:
                self.select_body(self.sun)
                self._update_educational_panel()

        elif K_1 <= key <= K_9:
            planet_index = key - K_1
            if planet_index < len(PLANET_ORDER):
                planet_name = PLANET_ORDER[planet_index]
                self.select_body(self.planets[planet_name])
                self._update_educational_panel()

        return True

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

        # Get educational info from PLANET_DESCRIPTIONS
        body_name = self.selected_body.name
        if body_name in PLANET_DESCRIPTIONS:
            info = cast(dict[str, Any], PLANET_DESCRIPTIONS[body_name])

            # Build properties dict
            properties: dict[str, Any] = {}
            info_dict = dict(info)
            for key, value in info_dict.items():
                if key != "fun_facts":
                    properties[key.replace("_", " ").title()] = value

            # Get fun facts
            fun_facts = info.get("fun_facts", [])

            self.educational_panel.set_body(body_name, properties, fun_facts)

        self._mark_immersion_task("select_body")

    def _mark_immersion_task(self, task_id: str) -> None:
        """Mark an immersion checklist task as complete if available."""
        if self.immersion_checklist:
            self.immersion_checklist.mark_complete(task_id)

    def _handle_time_nav_action(self, action: str) -> None:
        """
        Handle time navigation panel button actions.

        Args:
            action: The navigation action to perform
        """
        if action == "prev_day":
            self.time_manager.advance_days(-1)
        elif action == "next_day":
            self.time_manager.advance_days(1)
        elif action == "prev_week":
            self.time_manager.advance_days(-7)
        elif action == "next_week":
            self.time_manager.advance_days(7)
        elif action == "prev_month":
            self.time_manager.advance_days(-30)
        elif action == "next_month":
            self.time_manager.advance_days(30)
        elif action == "prev_year":
            self.time_manager.advance_years(-1)
        elif action == "next_year":
            self.time_manager.advance_years(1)
        elif action == "goto_today":
            self.time_manager.set_to_now()
        elif action == "goto_j2000":
            self.time_manager.set_to_j2000()
        elif action == "goto_j2030":
            if hasattr(self.time_manager, "J2030"):
                self.time_manager.set_datetime(self.time_manager.J2030)
        elif action == "reset":
            self.time_manager.set_to_now()
        elif action == "faster":
            self.time_manager.increase_time_warp()
        elif action == "slower":
            self.time_manager.decrease_time_warp()
        elif action == "reverse":
            self.time_manager.reverse_time()
        elif action == "toggle_pause":
            self.time_manager.toggle_pause()

        # Update UI after time change
        self._update_ui_date()
        self._mark_immersion_task("navigate_time")

    def _handle_mouse_button(self, button: int, pressed: bool) -> None:
        """Handle mouse button events."""
        if button == 1:  # Left button
            # Check UI clicks first
            if pressed and self._handle_ui_click(pygame.mouse.get_pos()):
                return

            self._mouse_dragging = pressed
            if pressed:
                self._last_mouse_pos = pygame.mouse.get_pos()

        elif button == 3:  # Right button
            if pressed:
                self._mouse_dragging = True
                self._last_mouse_pos = pygame.mouse.get_pos()
            else:
                self._mouse_dragging = False

    def _handle_mouse_motion(self, pos: tuple[int, int], rel: tuple[int, int]) -> None:
        """Handle mouse motion."""
        if not self.renderer:
            return

        if self._mouse_dragging:
            # Get mouse buttons
            buttons = pygame.mouse.get_pressed()

            mode = "Orbit"
            if self.unified_controls:
                mode = self.unified_controls.get_current_mode()
            elif self.nav_mode_panel:
                mode = self.nav_mode_panel.get_current_mode()

            if buttons[0]:  # Left button - depends on mode
                if mode == "Orbit":
                    self.renderer.camera.orbit(-rel[0], -rel[1])
                elif mode == "Pan":
                    self.renderer.camera.pan(-rel[0], rel[1])
                elif mode == "Zoom":
                    self.renderer.camera.zoom(rel[1] * 0.5)

            elif buttons[2]:  # Right button - pan camera
                self.renderer.camera.pan(-rel[0], rel[1])

    def _handle_mouse_wheel(self, y_offset: float) -> None:
        """Handle mouse wheel events."""
        if not self.renderer:
            return

        mode = "Orbit"
        if self.unified_controls:
            mode = self.unified_controls.get_current_mode()
        elif self.nav_mode_panel:
            mode = self.nav_mode_panel.get_current_mode()

        if mode == "Zoom" or mode == "Orbit":
            # Normal orbit zoom is always available via scroll,
            # but now we want to zoom towards the mouse cursor
            mx, my = pygame.mouse.get_pos()
            # Convert to NDC (-1 to 1)
            # Convert to NDC (-1 to 1)
            width = self.renderer.settings.window_width
            height = self.renderer.settings.window_height
            aspect = width / height

            ndc_x = (mx / width) * 2.0 - 1.0
            ndc_y = -((my / height) * 2.0 - 1.0)  # Y is inverted in OpenGL

            # Use new zoom_at capability
            self.renderer.camera.zoom_at(y_offset, (ndc_x, ndc_y), aspect)

        elif mode == "Pan":
            # Maybe scroll pans up/down? for now just zoom
            width = self.renderer.settings.window_width
            height = self.renderer.settings.window_height
            aspect = width / height
            self.renderer.camera.zoom_at(y_offset, (0, 0), aspect)

        self._mark_immersion_task(
            "toggle_overlays"
        )  # Close enough to 'interacting', trigger check

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

        new_mode = modes[next_index]
        camera.set_mode(new_mode, self.selected_body)

    def _focus_on_selected(self) -> None:
        """Focus the camera on the currently selected body."""
        if not self.selected_body or not self.renderer:
            return

        # Get body position
        state = self.selected_body.get_state_at_time(self.time_manager.julian_date)
        pos = state.position * self.renderer.distance_scale

        self.renderer.camera.look_at(pos)

        # Set appropriate distance based on body type
        if self.selected_body.body_type == BodyType.STAR:
            self.renderer.camera.set_distance(20)
        elif self.selected_body.name in INNER_PLANETS:
            self.renderer.camera.set_distance(5)
        else:
            self.renderer.camera.set_distance(15)

    def _update(self) -> None:
        """Update the simulation state (time, physics, camera)."""
        # Update time
        delta_jd = self.time_manager.update()

        current_jd = self.time_manager.julian_date

        if (delta_jd or self._last_ui_sync_jd is None) and (
            self._last_ui_sync_jd is None
            or not math.isclose(current_jd, self._last_ui_sync_jd)
        ):
            self._update_ui_date()
            self._last_ui_sync_jd = current_jd

        # Update camera
        if self.renderer:
            self.renderer.camera.update(
                self.time_manager.julian_date, self.renderer.distance_scale
            )

    def _render(self) -> None:
        """Render the current frame."""
        if not self.renderer:
            return
        renderer = self.renderer
        jd = self.time_manager.julian_date

        if self.settings.stereo_view:
            left_eye, right_eye = renderer.camera.stereo_states()
            half_width = renderer.settings.window_width // 2

            glViewport(0, 0, half_width, renderer.settings.window_height)
            renderer.begin_frame(camera_state=left_eye)
            self._render_view_contents(jd)

            glViewport(half_width, 0, half_width, renderer.settings.window_height)
            renderer.begin_frame(camera_state=right_eye, clear=False)
            glClear(GL_DEPTH_BUFFER_BIT)
            self._render_view_contents(jd)

            glViewport(
                0, 0, renderer.settings.window_width, renderer.settings.window_height
            )
            renderer.begin_frame(clear=False)
            self._render_overlays(jd)
            renderer.end_frame()
            return

        renderer.begin_frame()
        self._render_view_contents(jd)
        self._render_overlays(jd)
        renderer.end_frame()

    def _render_view_contents(self, julian_date: float) -> None:
        """
        Render the 3D contents of the view (bodies, stars, orbits).

        Args:
            julian_date: The current simulation time.
        """
        if not self.renderer:
            return
        renderer = self.renderer

        renderer.render_stars()
        renderer.render_grid()
        renderer.render_axes()

        if self.view_state.show_orbits:
            for planet in self.planets.values():
                if self._should_render_body(planet):
                    renderer.render_orbit(planet, julian_date)

        if self.sun:
            renderer.render_body(self.sun, julian_date, self.selected_body == self.sun)

        if self.view_state.show_labels:
            sun_pos = np.array([0, 0, 0])
            renderer.render_label("Sun", sun_pos)

        for planet in self.planets.values():
            if self._should_render_body(planet):
                is_selected = self.selected_body == planet
                renderer.render_body(planet, julian_date, is_selected)

                if self.view_state.show_labels:
                    state = planet.get_state_at_time(julian_date)
                    pos = state.position * renderer.distance_scale
                    renderer.render_label(planet.name, pos)

        if self.view_state.show_minor_bodies:
            renderer.render_asteroid_belt(self.asteroid_belt_points)
            for asteroid in self.asteroids.values():
                renderer.render_body(
                    asteroid, julian_date, self.selected_body == asteroid
                )
                if self.view_state.show_labels:
                    state = asteroid.get_state_at_time(julian_date)
                    renderer.render_label(
                        asteroid.name, state.position * renderer.distance_scale
                    )

            for comet in self.comets.values():
                renderer.render_body(comet, julian_date, self.selected_body == comet)
                if self.view_state.show_orbits:
                    renderer.render_orbit(
                        comet, julian_date, color=(0.6, 0.8, 1.0, 0.7)
                    )
                if self.view_state.show_labels:
                    state = comet.get_state_at_time(julian_date)
                    renderer.render_label(
                        comet.name, state.position * renderer.distance_scale
                    )

        for moon in self.moons.values():
            renderer.render_body(moon, julian_date, self.selected_body == moon)
            if self.view_state.show_labels:
                state = moon.get_state_at_time(julian_date)
                renderer.render_label(
                    moon.name, state.position * renderer.distance_scale
                )

        if self.view_state.show_trajectories:
            for trajectory in self.trajectories:
                renderer.render_trajectory(trajectory.trajectory_points)

        for spacecraft in self.spacecraft.values():
            if spacecraft.trajectory and len(spacecraft.trajectory) >= 2:
                start_time = spacecraft.trajectory[0].time
                end_time = spacecraft.trajectory[-1].time
                if start_time <= julian_date <= end_time:
                    state = spacecraft.get_state_at_time(julian_date)
                    pos = state.position * renderer.distance_scale
                    renderer.render_label("🚀 " + spacecraft.name, pos, (0, 255, 128))

    def _render_overlays(self, julian_date: float) -> None:
        """
        Render 2D UI overlays (sidebar, controls, HUD).

        Args:
            julian_date: The current simulation time.
        """
        if not self.renderer:
            return
        renderer = self.renderer

        # 1. Sidebar Panel (Right)
        if self.sidebar_panel:
            # We need to prepare the content data for the active tab
            content_key = self.sidebar_panel.tabs[
                self.sidebar_panel.current_tab_index
            ].content_renderer_key
            content_data = None

            # Helper to get info/edu data
            if content_key == "educational" and self.educational_panel:
                # If body selected, show it, otherwise show "Welcome" or something?
                # The educational panel class assumes a body is set.
                if self.selected_body:
                    # Refresh data
                    info = self.selected_body.get_info_dict()
                    state = self.selected_body.get_state_at_time(julian_date)
                    dist = np.linalg.norm(state.position) / AU
                    info["Distance"] = f"{dist:.2f} AU"
                    self.educational_panel.set_body(self.selected_body.name, info)
                content_data = self.educational_panel.get_render_data()

            elif content_key == "checklist" and self.immersion_checklist:
                content_data = self.immersion_checklist.get_render_data()
            elif content_key == "history" and self.historical_events:
                content_data = self.historical_events.get_render_data()
            elif content_key == "planets":
                # Generate planet list data
                bodies: list[dict[str, Any]] = []
                # Add Sun
                bodies.append(
                    {"name": "Sun", "selected": self.selected_body == self.sun}
                )
                # Add Planets
                for name in PLANET_ORDER:
                    if name in self.planets:
                        bodies.append(
                            {
                                "name": name,
                                "selected": self.selected_body == self.planets[name],
                            }
                        )

                content_data = {"visible": True, "bodies": bodies}

            # Pass to sidebar renderer (handles frame + invokes content)
            renderer.render_sidebar(self.sidebar_panel.get_render_data(), content_data)

        # 2. Unified Control Panel (Bottom)
        if self.unified_controls:
            # Pass TimeNav data to it
            time_data = (
                self.time_nav_panel.get_render_data() if self.time_nav_panel else {}
            )
            renderer.render_unified_controls(
                self.unified_controls.get_render_data(), time_data
            )

        # 3. Floating Overlays
        # Basic status info (FPS etc) - simplified now that we have panels?
        status = f"FPS: {renderer.get_fps():.0f}"
        if self.selected_body:
            status += f"  |  Selected: {self.selected_body.name}"
        renderer.render_status_bar(status)

        if (
            self.view_state.show_help
            and hasattr(self, "help_overlay")
            and self.help_overlay
        ):
            renderer.render_help_overlay(self.help_overlay.get_render_data())

        if self.date_picker and self.date_picker.visible:
            renderer.render_date_picker(self.date_picker.get_render_data())

        # 4. HUD Elements
        renderer.render_speed_indicator(self.time_manager.time_warp)
        renderer.render_compass(renderer.camera.yaw)

    def _should_render_body(self, body: CelestialBody) -> bool:
        """
        Determine if a celestial body should be rendered based on view settings.

        Args:
            body: The body to check.

        Returns:
            True if the body is visible, False otherwise.
        """
        # Check granular visibility flags
        if body.name in INNER_PLANETS:
            return self.view_state.show_inner_planets
        elif body.name in OUTER_PLANETS:
            return self.view_state.show_outer_planets
        elif body.name in DWARF_PLANETS:
            return self.view_state.show_dwarf_planets
        # For Moons, we might need a specific flag if we want to toggle them separately
        elif body.body_type == BodyType.MOON:
            # For now, let's tie moon visibility to general minor bodies
            # or parent visibility?
            # User asked for granular. Let's start with minor bodies flag
            # for moons/asteroids.
            return self.view_state.show_minor_bodies
        elif body.body_type in {BodyType.ASTEROID, BodyType.COMET}:
            return self.view_state.show_minor_bodies
        return True

    def get_transfer_summary(
        self, origin_name: str, destination_name: str
    ) -> dict[str, Any] | None:
        """
        Get a summary of a potential transfer between two bodies.

        Args:
            origin_name: Name of the origin body.
            destination_name: Name of the destination body.

        Returns:
            A dictionary with transfer details, or None if bodies are invalid.
        """
        origin = self.get_body_by_name(origin_name)
        destination = self.get_body_by_name(destination_name)

        if not origin or not destination:
            return None

        return self.trajectory_planner.get_transfer_summary(origin, destination)

    def _handle_ui_click(self, pos: tuple[int, int]) -> bool:
        """Handle clicks on UI overlays."""
        x, y = pos

        # 1. Check Date Picker
        if self.date_picker and self.date_picker.visible:
            # Add logic if DatePicker had proper hit testing exposed
            pass  # Placeholder for future hit testing logic

        # 2. Check Sidebar
        if self.sidebar_panel:
            sx, sy = self.sidebar_panel.position
            if (
                sx <= x <= sx + self.sidebar_panel.width
                and sy <= y <= sy + self.sidebar_panel.height
            ):
                rel_x, rel_y = x - sx, y - sy
                action = self.sidebar_panel.handle_click(rel_x, rel_y)
                if action == "tab_changed":
                    return True

                # Handle content clicks
                current_tab = self.sidebar_panel.tabs[
                    self.sidebar_panel.current_tab_index
                ]
                if current_tab.content_renderer_key == "planets":
                    # Simple list click detection matching UIRenderer layout
                    # Header ~35px + Title ~30px = ~65px offset.
                    # Items start at y + 35 + 30 + 10 (padding) = 75px
                    # relative to sidebar
                    # Item height 25px

                    list_start_y = 75
                    if rel_y > list_start_y:
                        idx = (rel_y - list_start_y) // 25
                        bodies = ["Sun"] + [
                            p for p in PLANET_ORDER if p in self.planets
                        ]
                        if 0 <= idx < len(bodies):
                            name = bodies[idx]
                            body = self.get_body_by_name(name)
                            if body:
                                self.select_body(body)
                                self._focus_on_selected()

                return True

        # 3. Check Unified Controls
        if self.unified_controls:
            cx, cy = self.unified_controls.position
            cw = self.unified_controls.width
            ch = self.unified_controls.height
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                rel_x = x - cx
                rel_y = y - cy

                # A. Navigation Modes (Left) [20, 20] -> width ~300
                if 20 <= rel_x <= 350 and 20 <= rel_y <= 80:
                    idx = (rel_x - 20) // 80
                    if 0 <= idx < len(self.unified_controls.modes):
                        self.unified_controls.set_mode(self.unified_controls.modes[idx])
                    return True

                # B. Buttons (Left, below modes)
                if 20 <= rel_x <= 350 and 80 <= rel_y <= 120:
                    bx = 20
                    for btn in self.unified_controls.buttons:
                        if bx <= rel_x <= bx + btn.width:
                            if btn.action == "reset_view":
                                if self.renderer:
                                    self.renderer.camera.reset()
                            elif btn.action == "toggle_orbits_btn":
                                self._handle_setting_action("toggle_orbits")
                            return True
                        bx += btn.width + 10

                # C. View Settings (Right)
                set_x = cw - 350
                if rel_x >= set_x:
                    # Checkboxes
                    # 2 columns: col1 at 0, col2 at 160 relative to set_x
                    # row height 30 start at y=55 relative to panel
                    start_y = 55
                    if rel_y >= start_y:
                        row = (rel_y - start_y) // 30
                        col = 0 if (rel_x - set_x) < 160 else 1
                        idx = row * 2 + col
                        if idx < len(self.unified_controls.checkboxes):
                            action = self.unified_controls.toggle_checkbox(int(idx))
                            if action:
                                self._handle_setting_action(action)
                    return True

                return True

        return False

    def _handle_setting_action(self, action: str) -> None:
        """
        Handle settings actions triggered by UI controls.

        Args:
            action: The action identifier string.
        """
        if action == "toggle_orbits":
            self.view_state.show_orbits = not self.view_state.show_orbits
        elif action == "toggle_labels":
            self.view_state.show_labels = not self.view_state.show_labels
            if self.renderer:
                self.renderer.settings.show_labels = self.view_state.show_labels
        elif action == "toggle_grid":
            if self.renderer:
                self.renderer.settings.show_grid = not self.renderer.settings.show_grid
        elif action == "toggle_stereo":
            self.settings.stereo_view = not self.settings.stereo_view
        # Granular
        elif action == "toggle_inner":
            self.view_state.show_inner_planets = not self.view_state.show_inner_planets
        elif action == "toggle_outer":
            self.view_state.show_outer_planets = not self.view_state.show_outer_planets
        elif action == "toggle_dwarf":
            self.view_state.show_dwarf_planets = not self.view_state.show_dwarf_planets
        elif action == "toggle_moons":
            self.view_state.show_minor_bodies = not self.view_state.show_minor_bodies

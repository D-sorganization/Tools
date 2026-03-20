"""Rendering mixin for SolarSystemScene.

Extracts 3D and overlay rendering logic from the main scene class
to reduce class size and improve single-responsibility adherence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.celestial_body import BodyType
from ..core.constants import (
    DWARF_PLANETS,
    INNER_PLANETS,
    OUTER_PLANETS,
    PLANET_ORDER,
)
from ..data.famous_missions import FAMOUS_MISSIONS

if TYPE_CHECKING:
    pass

try:
    from OpenGL.GL import GL_DEPTH_BUFFER_BIT, glClear, glViewport

    _OPENGL_AVAILABLE = True
except ImportError:
    _OPENGL_AVAILABLE = False


class SceneRenderMixin:
    """Mixin providing rendering logic for SolarSystemScene."""

    def _render(self) -> None:
        """Render the current frame."""
        if not self.renderer:  # type: ignore
            return
        renderer = self.renderer  # type: ignore
        jd = self.time_manager.julian_date  # type: ignore

        if self.settings.stereo_view:  # type: ignore
            self._render_stereo(renderer, jd)
            return

        renderer.begin_frame()
        self._render_view_contents(jd)
        self._render_overlays(jd)
        renderer.end_frame()

    def _render_stereo(self, renderer: Any, jd: float) -> None:
        """Render a stereo/VR split-screen frame."""
        assert jd is not None, "jd must be provided"
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

    def _render_view_contents(self, julian_date: float) -> None:
        """Render the 3D contents of the view (bodies, stars, orbits).

        Args:
            julian_date: The current simulation time.
        """
        assert julian_date is not None, "julian_date must be provided"
        if not self.renderer:  # type: ignore
            return
        renderer = self.renderer  # type: ignore

        renderer.render_stars()
        renderer.render_grid()
        renderer.render_axes()

        if self.view_state.show_orbits:  # type: ignore
            for planet in self.planets.values():  # type: ignore
                if self._should_render_body(planet):
                    renderer.render_orbit(planet, julian_date)

        self._render_sun(renderer, julian_date)
        self._render_planets(renderer, julian_date)
        self._render_minor_bodies(renderer, julian_date)
        self._render_moons(renderer, julian_date)
        self._render_trajectories(renderer, julian_date)
        self._render_spacecraft(renderer, julian_date)

    def _render_sun(self, renderer: Any, julian_date: float) -> None:
        """Render the Sun body and label."""
        assert julian_date is not None, "julian_date must be provided"
        if self.sun:  # type: ignore
            renderer.render_body(self.sun, julian_date, self.selected_body == self.sun)  # type: ignore
        if self.view_state.show_labels:  # type: ignore
            sun_pos = np.array([0, 0, 0])
            renderer.render_label("Sun", sun_pos, priority=3)

    def _render_planets(self, renderer: Any, julian_date: float) -> None:
        """Render all visible planets with labels."""
        for planet in self.planets.values():  # type: ignore
            if not self._should_render_body(planet):
                continue
            is_selected = self.selected_body == planet  # type: ignore
            renderer.render_body(planet, julian_date, is_selected)
            if self.view_state.show_labels:  # type: ignore
                state = planet.get_state_at_time(julian_date)
                pos = state.position * renderer.distance_scale
                renderer.render_label(planet.name, pos, priority=3)

    def _render_minor_bodies(self, renderer: Any, julian_date: float) -> None:
        """Render asteroids, comets, and the asteroid belt."""
        assert julian_date is not None, "julian_date must be provided"
        if not self.view_state.show_minor_bodies:  # type: ignore
            return

        renderer.render_asteroid_belt(self.asteroid_belt_points)  # type: ignore

        for asteroid in self.asteroids.values():  # type: ignore
            is_selected = self.selected_body == asteroid  # type: ignore
            renderer.render_body(asteroid, julian_date, is_selected)
            state = asteroid.get_state_at_time(julian_date)
            pos = state.position * renderer.distance_scale
            dist_to_cam = np.linalg.norm(pos - np.array(renderer.camera.position))
            if self.view_state.show_labels and (is_selected or dist_to_cam < 1.0):  # type: ignore
                renderer.render_label(asteroid.name, pos, priority=1)

        for comet in self.comets.values():  # type: ignore
            is_selected = self.selected_body == comet  # type: ignore
            renderer.render_body(comet, julian_date, is_selected)
            if self.view_state.show_orbits:  # type: ignore
                renderer.render_orbit(comet, julian_date, color=(0.6, 0.8, 1.0, 0.7))
            state = comet.get_state_at_time(julian_date)
            pos = state.position * renderer.distance_scale
            dist_to_cam = np.linalg.norm(pos - np.array(renderer.camera.position))
            if self.view_state.show_labels and (is_selected or dist_to_cam < 3.0):  # type: ignore
                renderer.render_label(comet.name, pos, priority=2)

    def _render_moons(self, renderer: Any, julian_date: float) -> None:
        """Render all moons with proximity-based labels."""
        for moon in self.moons.values():  # type: ignore
            is_selected = self.selected_body == moon  # type: ignore
            renderer.render_body(moon, julian_date, is_selected)
            if self.view_state.show_labels:  # type: ignore
                state = moon.get_state_at_time(julian_date)
                pos = state.position * renderer.distance_scale
                dist_to_cam = np.linalg.norm(pos - np.array(renderer.camera.position))
                if is_selected or dist_to_cam < 0.5:
                    renderer.render_label(moon.name, pos, priority=1)

    def _render_trajectories(self, renderer: Any, julian_date: float) -> None:
        """Render active transfer trajectories and famous mission paths."""
        assert julian_date is not None, "julian_date must be provided"
        if not self.view_state.show_trajectories:  # type: ignore
            return
        for trajectory in self.trajectories:  # type: ignore
            renderer.render_trajectory(trajectory.trajectory_points)
        for spacecraft in self.spacecraft.values():  # type: ignore
            if spacecraft.trajectory:
                renderer.render_trajectory(
                    spacecraft.trajectory,
                    color=(0.1, 0.8, 1.0, 0.4),
                    line_width=1.5,
                )

    def _render_spacecraft(self, renderer: Any, julian_date: float) -> None:
        """Render spacecraft markers and labels based on mission timeline."""
        for spacecraft in self.spacecraft.values():  # type: ignore
            if not spacecraft.trajectory or len(spacecraft.trajectory) < 2:
                continue
            start_time = spacecraft.trajectory[0].time
            end_time = spacecraft.trajectory[-1].time

            if start_time <= julian_date <= end_time:
                state = spacecraft.get_state_at_time(julian_date)
                pos = state.position * renderer.distance_scale
                renderer.render_label(
                    "\U0001f680 " + spacecraft.name, pos, (0, 255, 128), priority=2
                )
            elif julian_date > end_time:
                state = spacecraft.get_state_at_time(julian_date)
                pos = state.position * renderer.distance_scale
                renderer.render_label(
                    "\u2b50 " + spacecraft.name, pos, (150, 150, 150), priority=1
                )

    def _render_overlays(self, julian_date: float) -> None:
        """Render 2D UI overlays (sidebar, controls, HUD).

        Args:
            julian_date: The current simulation time.
        """
        assert julian_date is not None, "julian_date must be provided"
        if not self.renderer:  # type: ignore
            return
        renderer = self.renderer  # type: ignore

        self._render_sidebar(renderer, julian_date)
        self._render_unified_controls(renderer)
        self._render_floating_overlays(renderer)
        self._render_hud(renderer)

    def _render_sidebar(self, renderer: Any, julian_date: float) -> None:
        """Render the sidebar panel with active-tab content."""
        assert julian_date is not None, "julian_date must be provided"
        if not self.sidebar_panel:  # type: ignore
            return

        content_key = self.sidebar_panel.tabs[  # type: ignore
            self.sidebar_panel.current_tab_index  # type: ignore
        ].content_renderer_key
        content_data = self._get_sidebar_content_data(content_key, julian_date)
        renderer.render_sidebar(self.sidebar_panel.get_render_data(), content_data)  # type: ignore

    def _get_sidebar_content_data(
        self, content_key: str, julian_date: float
    ) -> dict[str, Any] | None:
        """Build the content data dict for the active sidebar tab."""
        assert content_key is not None, "content_key must be provided"
        if content_key == "educational" and self.educational_panel:  # type: ignore
            if self.selected_body:  # type: ignore
                info = self.selected_body.get_info_dict_at_time(julian_date)  # type: ignore
                self.educational_panel.set_body(self.selected_body.name, info)  # type: ignore
            return self.educational_panel.get_render_data()  # type: ignore

        if content_key == "checklist" and self.immersion_checklist:  # type: ignore
            return self.immersion_checklist.get_render_data()  # type: ignore

        if content_key == "history" and self.historical_events:  # type: ignore
            return self.historical_events.get_render_data()  # type: ignore

        if content_key == "missions" and self.missions_panel:  # type: ignore
            return self.missions_panel.get_render_data(FAMOUS_MISSIONS)  # type: ignore

        if content_key == "planets":
            bodies: list[dict[str, Any]] = []
            bodies.append({"name": "Sun", "selected": self.selected_body == self.sun})  # type: ignore
            for name in PLANET_ORDER:
                if name in self.planets:  # type: ignore
                    bodies.append(
                        {
                            "name": name,
                            "selected": self.selected_body == self.planets[name],  # type: ignore
                        }
                    )
            return {"visible": True, "bodies": bodies}

        return None

    def _render_unified_controls(self, renderer: Any) -> None:
        """Render the unified control panel."""
        if not self.unified_controls:  # type: ignore
            return
        time_data = self.time_nav_panel.get_render_data() if self.time_nav_panel else {}  # type: ignore
        renderer.render_unified_controls(
            self.unified_controls.get_render_data(), time_data  # type: ignore
        )

    def _render_floating_overlays(self, renderer: Any) -> None:
        """Render floating overlays (status bar, help, date picker)."""
        status = f"FPS: {renderer.get_fps():.0f}"
        if self.selected_body:  # type: ignore
            status += f"  |  Selected: {self.selected_body.name}"  # type: ignore
        renderer.render_status_bar(status)

        if (
            self.view_state.show_help  # type: ignore
            and hasattr(self, "help_overlay")
            and self.help_overlay
        ):
            renderer.render_help_overlay(self.help_overlay.get_render_data())

        if self.date_picker and self.date_picker.visible:  # type: ignore
            renderer.render_date_picker(self.date_picker.get_render_data())  # type: ignore

    def _render_hud(self, renderer: Any) -> None:
        """Render HUD elements (speed indicator, compass)."""
        renderer.render_speed_indicator(self.time_manager.time_warp)  # type: ignore
        renderer.render_compass(renderer.camera.yaw)

    def _should_render_body(self, body: Any) -> bool:
        """Determine if a celestial body should be rendered based on view settings.

        Args:
            body: The body to check.

        Returns:
            True if the body is visible, False otherwise.
        """
        if body.name in INNER_PLANETS:
            return self.view_state.show_inner_planets  # type: ignore
        elif body.name in OUTER_PLANETS:
            return self.view_state.show_outer_planets  # type: ignore
        elif body.name in DWARF_PLANETS:
            return self.view_state.show_dwarf_planets  # type: ignore
        elif body.body_type == BodyType.MOON:
            return self.view_state.show_minor_bodies  # type: ignore
        elif body.body_type in {BodyType.ASTEROID, BodyType.COMET}:
            return self.view_state.show_minor_bodies  # type: ignore
        return True

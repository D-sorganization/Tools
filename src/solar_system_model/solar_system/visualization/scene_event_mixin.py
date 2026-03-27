from numba import jit

"""Event handling mixin for SolarSystemScene.

Extracts mouse, keyboard, and UI click handling from the main scene class
to reduce class size and improve single-responsibility adherence.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from ..core.constants import PLANET_ORDER
from ..data.famous_missions import FAMOUS_MISSIONS

if TYPE_CHECKING:
    pass

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

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from .camera import CameraMode


class SceneEventMixin:
    """Mixin providing event-handling logic for SolarSystemScene."""

    def _handle_events(self) -> bool:
        """Process all pending Pygame events.

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
        """Handle keyboard input.

        Returns:
            False if should quit, True otherwise.
        """
        if not (key is not None):
            raise ValueError("key must be provided")
        if key == K_ESCAPE:
            return False
        if not self.renderer:
            return True

        if key == K_SPACE:
            self.time_manager.toggle_pause()
        elif key in (K_EQUALS, K_PLUS, K_KP_PLUS):
            self.time_manager.increase_time_warp()
        elif key in (K_MINUS, K_KP_MINUS):
            self.time_manager.decrease_time_warp()
        elif key == K_r:
            self.time_manager.reverse_time()
        elif key in (K_d, K_n, K_e):
            self._handle_panel_toggle(key)
        elif key in (K_LEFTBRACKET, K_RIGHTBRACKET, K_PAGEUP, K_PAGEDOWN):
            self._handle_time_jump(key)
        elif key == K_HOME:
            self.renderer.camera.reset()
            self.renderer.camera.mode = CameraMode.FREE
        elif key in (K_o, K_l, K_i, K_g, K_h, K_v, K_c, K_f, K_m):
            self._handle_view_toggle(key)
        elif key == K_t:
            self._handle_trajectory_plan()
        elif key == K_PERIOD:
            if self.educational_panel and self.educational_panel.visible:
                self.educational_panel.cycle_fact()
        elif key == K_0:
            if self.sun:
                self.select_body(self.sun)
                self._update_educational_panel()
        elif K_1 <= key <= K_9:
            self._select_planet_by_number(key)
        return True

    def _handle_panel_toggle(self, key: int) -> None:
        """Toggle date-picker, time-nav, or historical-events panels."""
        if key == K_d and self.date_picker:
            self.date_picker.toggle()
            if self.date_picker.visible:
                self.date_picker.set_date(self.time_manager.current_time.datetime_utc)
                self._mark_immersion_task("navigate_time")
        elif key == K_n and self.time_nav_panel:
            self.time_nav_panel.toggle()
            self._mark_immersion_task("navigate_time")
        elif key == K_e and self.historical_events:
            self.historical_events.toggle()
            if self.historical_events.visible:
                self._mark_immersion_task("historical_events")

    def _handle_time_jump(self, key: int) -> None:
        """Jump forward/backward by 1 day or 1 month."""
        if key == K_LEFTBRACKET:
            self._jump_time(-1)
        elif key == K_RIGHTBRACKET:
            self._jump_time(1)
        elif key == K_PAGEUP:
            self._jump_month(-1)
        elif key == K_PAGEDOWN:
            self._jump_month(1)

    def _handle_view_toggle(self, key: int) -> None:
        """Toggle visibility / overlay options."""
        if key == K_o:
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
        elif key == K_m:
            if self.immersion_checklist:
                self.immersion_checklist.toggle()
            self.view_state.show_immersion_checklist = not self.view_state.show_immersion_checklist

    def _handle_trajectory_plan(self) -> None:
        """Plan an Earth-to-Mars trajectory and display the result."""
        trajectory = self.plan_trajectory("Earth", "Mars")
        if trajectory:
            self._mark_immersion_task("plan_transfer")
            self._action_message = (
                "Earth\u2192Mars transfer: \u0394V "
                f"{trajectory.total_delta_v / 1000:.2f} km/s, "
                f"flight {trajectory.time_of_flight:.1f} days"
            )
        else:
            self._action_message = "Earth\u2192Mars transfer could not be created"

    def _select_planet_by_number(self, key: int) -> None:
        """Select a planet using the number key (1-9)."""
        if not (key is not None):
            raise ValueError("key must be provided")
        planet_index = key - K_1
        if planet_index < len(PLANET_ORDER):
            planet_name = PLANET_ORDER[planet_index]
            self.select_body(self.planets[planet_name])
            self._update_educational_panel()

    def _handle_mouse_button(self, button: int, pressed: bool) -> None:
        """Handle mouse button events."""
        if button == 1:  # Left button
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
        if not (pos is not None):
            raise ValueError("pos must be provided")
        if not self.renderer:
            return
        if self._mouse_dragging:
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
        if not (y_offset is not None):
            raise ValueError("y_offset must be provided")
        if not self.renderer:
            return
        mode = "Orbit"
        if self.unified_controls:
            mode = self.unified_controls.get_current_mode()
        elif self.nav_mode_panel:
            mode = self.nav_mode_panel.get_current_mode()

        if mode in ("Zoom", "Orbit"):
            mx, my = pygame.mouse.get_pos()
            width = self.renderer.settings.window_width
            height = self.renderer.settings.window_height
            aspect = width / height
            ndc_x = (mx / width) * 2.0 - 1.0
            ndc_y = -((my / height) * 2.0 - 1.0)
            self.renderer.camera.zoom_at(y_offset, (ndc_x, ndc_y), aspect)
        elif mode == "Pan":
            width = self.renderer.settings.window_width
            height = self.renderer.settings.window_height
            aspect = width / height
            self.renderer.camera.zoom_at(y_offset, (0, 0), aspect)

        self._mark_immersion_task("toggle_overlays")

    def _handle_ui_click(self, pos: tuple[int, int]) -> bool:
        """Handle clicks on UI overlays."""
        if not (pos is not None):
            raise ValueError("pos must be provided")
        x, y = pos

        # 1. Check Date Picker
        if self.date_picker and self.date_picker.visible:
            pass  # Placeholder for future hit testing logic

        # 2. Check Sidebar
        if self._handle_sidebar_click(x, y):
            return True

        # 3. Check Unified Controls
        if self._handle_controls_click(x, y):
            return True

        return False

    def _handle_sidebar_click(self, x: int, y: int) -> bool:
        """Handle click within the sidebar panel.

        Returns:
            True if the click was consumed by the sidebar.
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        if not self.sidebar_panel:
            return False

        sx, sy = self.sidebar_panel.position
        if not (
            sx <= x <= sx + self.sidebar_panel.width and sy <= y <= sy + self.sidebar_panel.height
        ):
            return False

        rel_x, rel_y = x - sx, y - sy
        action = self.sidebar_panel.handle_click(rel_x, rel_y)
        if action == "tab_changed":
            return True

        current_tab = self.sidebar_panel.tabs[self.sidebar_panel.current_tab_index]
        if current_tab.content_renderer_key == "planets":
            list_start_y = 75
            if rel_y > list_start_y:
                idx = (rel_y - list_start_y) // 25
                bodies = ["Sun"] + [p for p in PLANET_ORDER if p in self.planets]
                if 0 <= idx < len(bodies):
                    name = bodies[idx]
                    body = self.get_body_by_name(name)
                    if body:
                        self.select_body(body)
                        self._focus_on_selected()
        elif current_tab.content_renderer_key == "missions":
            list_start_y = 75
            if rel_y > list_start_y:
                # Approximate 90 pixels per mission entry in the list
                idx = (rel_y - list_start_y) // 90
                mission_names = list(FAMOUS_MISSIONS.keys())
                if 0 <= idx < len(mission_names):
                    name = mission_names[idx]
                    data = FAMOUS_MISSIONS[name]
                    launch_str = data.get("launch_date", "")
                    if launch_str:
                        launch_dt = datetime.strptime(launch_str, "%Y-%m-%d")
                        self.time_manager.set_datetime(launch_dt)
                        self._action_message = f"Simulating {name} launch..."

                    # Focus on spacecraft if available
                    if name in self.spacecraft:
                        self.select_body(self.spacecraft[name])
                        self._focus_on_selected()
        return True

    @jit(nopython=True, fastmath=True)
    def _handle_controls_click(self, x: int, y: int) -> bool:
        """Handle click within the unified control panel.

        Returns:
            True if the click was consumed by the control panel.
        """
        if not (x is not None):
            raise ValueError("x must be provided")
        if not self.unified_controls:
            return False

        cx, cy = self.unified_controls.position
        cw = self.unified_controls.width
        ch = self.unified_controls.height
        if not (cx <= x <= cx + cw and cy <= y <= cy + ch):
            return False

        rel_x = x - cx
        rel_y = y - cy

        # A. Navigation Modes (Left)
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

    def _handle_setting_action(self, action: str) -> None:
        """Handle settings actions triggered by UI controls.

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
        elif action == "toggle_inner":
            self.view_state.show_inner_planets = not self.view_state.show_inner_planets
        elif action == "toggle_outer":
            self.view_state.show_outer_planets = not self.view_state.show_outer_planets
        elif action == "toggle_dwarf":
            self.view_state.show_dwarf_planets = not self.view_state.show_dwarf_planets
        elif action == "toggle_moons":
            self.view_state.show_minor_bodies = not self.view_state.show_minor_bodies

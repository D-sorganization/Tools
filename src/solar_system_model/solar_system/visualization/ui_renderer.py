"""
UI Renderer
===========

Handles all 2D user interface rendering for the solar system visualization.
Separates UI logic from the main 3D renderer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import pygame
except ImportError:
    pass

try:
    from OpenGL.GL import (
        GL_BLEND,
        GL_DEPTH_TEST,
        GL_LIGHTING,
        GL_LINE_LOOP,
        GL_LINES,
        GL_MODELVIEW,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGBA,
        GL_SRC_ALPHA,
        GL_TEXTURE_2D,
        GL_TRIANGLE_FAN,
        GL_UNSIGNED_BYTE,
        glBegin,
        glBlendFunc,
        glColor4f,
        glDisable,
        glDrawPixels,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glPopMatrix,
        glPushMatrix,
        glRasterPos2i,
        glVertex2f,
    )
except ImportError:
    pass


@dataclass
class UITheme:
    """UI Color and Style definitions."""

    text_color: tuple[int, int, int] = (255, 255, 255)
    text_highlight: tuple[int, int, int] = (100, 200, 255)
    text_dim: tuple[int, int, int] = (180, 180, 180)

    bg_color: tuple[float, float, float, float] = (0.05, 0.08, 0.12, 0.95)
    border_color: tuple[float, float, float, float] = (0.3, 0.5, 0.7, 0.6)

    font_size_large: int = 28
    font_size_small: int = 20
    font_size_title: int = 32


class TextCache:
    """Caches rendered text surfaces to improve performance."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, str, tuple[int, int, int]], tuple[bytes, int, int]
        ] = {}
        self._fonts: dict[str, pygame.font.Font] = {}

        # Initialize fonts
        try:
            if "pygame" not in globals():
                raise ImportError
            pygame.font.init()
            self._fonts["default"] = pygame.font.SysFont("segoeui", 28, bold=True)
            self._fonts["small"] = pygame.font.SysFont("segoeui", 20)
            self._fonts["title"] = pygame.font.SysFont("segoeui", 32, bold=True)
        except (KeyError, ValueError, TypeError):
            self._fonts["default"] = pygame.font.Font(None, 28)
            self._fonts["small"] = pygame.font.Font(None, 20)
            self._fonts["title"] = pygame.font.Font(None, 32)

    def get_text_data(
        self, text: str, font_name: str, color: tuple[int, int, int]
    ) -> tuple[bytes, int, int]:
        """
        Get cached text data or render it.

        Returns:
            Tuple of (pixel_data, width, height)
        """
        if not (text is not None):
            raise ValueError("text must be provided")
        key = (text, font_name, color)
        if key in self._cache:
            return self._cache[key]

        font = self._fonts.get(font_name, self._fonts["default"])
        surface = font.render(text, True, color)
        data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()

        # Simple cache eviction if too large (naive)
        if len(self._cache) > 1000:
            self._cache.clear()

        self._cache[key] = (data, width, height)
        return data, width, height

    def render(
        self,
        text: str,
        x: int,
        y: int,
        font_name: str = "default",
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> tuple[int, int]:
        """Render text at position."""
        if not (text is not None):
            raise ValueError("text must be provided")
        data, width, height = self.get_text_data(text, font_name, color)
        glRasterPos2i(x, y + height)  # OpenGL draws from bottom-left
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return width, height


class UIRenderer:
    """
    Handles 2D UI rendering.
    """

    def __init__(self, window_width: int, window_height: int) -> None:
        if not (window_width is not None):
            raise ValueError("window_width must be provided")
        self.window_width = window_width
        self.window_height = window_height
        self.theme = UITheme()
        self.text_cache = TextCache()
        self.drawn_labels: list[pygame.Rect] = []

    def update_dimensions(self, width: int, height: int) -> None:
        """Update window dimensions for UI rendering."""
        if not (width is not None):
            raise ValueError("width must be provided")
        self.window_width = width
        self.window_height = height

    def begin_2d(self) -> None:
        """Setup 2D orthographic projection."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_width, self.window_height, 0, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def end_2d(self) -> None:
        """Restore 3D projection."""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def draw_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[float, float, float, float],
        filled: bool = True,
    ) -> None:
        """Draw a rectangle."""
        if not (x is not None):
            raise ValueError("x must be provided")
        glColor4f(*color)
        if filled:
            glBegin(GL_QUADS)
        else:
            glBegin(GL_LINE_LOOP)

        glVertex2f(x, y)
        glVertex2f(x + w, y)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

    def render_label_2d(
        self,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = (255, 255, 255),
        font_name: str = "default",
    ) -> None:
        """Render a text label ensuring no overlap via vertical offset.

        Args:
            text: The label string to render.
            position: Target (x, y) screen coordinates.
            color: RGB text colour.
            font_name: Font size key -- "default" (large), "small", or "title".
        """
        # Get dimensions without drawing
        if not (text is not None):
            raise ValueError("text must be provided")
        _, width, height = self.text_cache.get_text_data(text, font_name, color)

        rect = pygame.Rect(position[0], position[1], width, height)
        original_y = rect.y
        # Try original position, then offset vertically to dodge collisions
        offsets = [0, height + 2, -(height + 2), (height + 2) * 2, -(height + 2) * 2]

        final_pos = position
        found_spot = False

        for offset in offsets:
            test_rect = rect.copy()
            test_rect.y = original_y + offset

            if test_rect.top < 0 or test_rect.bottom > self.window_height:
                continue

            collision = False
            for other_rect in self.drawn_labels:
                if test_rect.colliderect(other_rect):
                    collision = True
                    break

            if not collision:
                rect = test_rect
                final_pos = (rect.x, rect.y)
                found_spot = True
                break

        if not found_spot:
            # All offsets collide -- skip this label entirely to reduce clutter
            return

        self.drawn_labels.append(rect)
        self.text_cache.render(text, final_pos[0], final_pos[1], font_name, color)

    def render_status_bar(self, text: str) -> None:
        """Render a status bar at the bottom of the screen."""
        if not (text is not None):
            raise ValueError("text must be provided")
        self.begin_2d()
        y = self.window_height - 30
        self.draw_rect(0, y - 5, self.window_width, 35, (0.0, 0.0, 0.0, 0.7))
        self.text_cache.render(text, 10, y, "default", (200, 200, 200))
        self.end_2d()

    def render_help_overlay(self, help_data: dict[str, Any]) -> None:
        """Render the help overlay with controls list."""
        if not (help_data is not None):
            raise ValueError("help_data must be provided")
        if not help_data.get("visible", False):
            return

        controls = help_data.get("controls", [])
        if not controls:
            return

        x, y = help_data.get("position") or (self.window_width - 350, 20)
        line_height = 20
        height = len(controls) * line_height + 40
        width = 335

        self.begin_2d()

        # Background and Border
        self.draw_rect(x - 15, y - 15, width, height, (0.0, 0.0, 0.0, 0.85))
        glLineWidth(2.0)
        self.draw_rect(
            x - 15, y - 15, width, height, (0.3, 0.5, 0.7, 0.8), filled=False
        )

        # Title
        self.text_cache.render(
            "CONTROLS (Press H to hide)", x, y, "default", (100, 200, 255)
        )

        current_y = y + 30
        for key, action in controls:
            if action == "":
                if key:
                    self.text_cache.render(
                        key, x, current_y, "default", (255, 200, 100)
                    )
                else:
                    current_y += line_height // 2
                    continue
            else:
                if key.strip():
                    text = f"{key}: {action}"
                    self.text_cache.render(text, x, current_y, "small", (220, 220, 220))
                else:
                    self.text_cache.render(
                        action, x, current_y, "small", (180, 180, 180)
                    )
            current_y += line_height

        self.end_2d()

    def render_date_picker(self, picker_data: dict[str, Any]) -> None:
        """Render the date picker widget."""
        if not (picker_data is not None):
            raise ValueError("picker_data must be provided")
        if not picker_data.get("visible", False):
            return

        x, y = picker_data.get("position", (20, 100))
        date = picker_data.get("date")
        if not date:
            return

        width = 300
        height = 80

        self.begin_2d()
        self.draw_rect(x - 5, y - 5, width, height, (0.1, 0.1, 0.15, 0.9))

        self.text_cache.render("Jump to Date", x, y, "default", (255, 255, 100))

        date_str = date.strftime("%Y-%m-%d %H:%M UTC")
        self.text_cache.render(date_str, x, y + 28, "default", (200, 240, 255))

        self.text_cache.render(
            "Press [ / ] to jump by day, E for events",
            x,
            y + 50,
            "small",
            (150, 150, 150),
        )
        self.end_2d()

    def render_sidebar(
        self, sidebar_data: dict[str, Any], content_data: dict[str, Any] | None
    ) -> None:
        """Render the sidebar with tabs and content."""
        if not (sidebar_data is not None):
            raise ValueError("sidebar_data must be provided")
        if not sidebar_data.get("visible", False):
            return

        x, y = sidebar_data.get("position", (0, 0))
        width = sidebar_data.get("width", 380)
        height = sidebar_data.get("height", 600)
        tabs = sidebar_data.get("tabs", [])
        current_tab = sidebar_data.get("current_tab_index", 0)
        content_key = sidebar_data.get("current_content_key", "")

        self.begin_2d()

        # Main BG
        self.draw_rect(x, y, width, height, self.theme.bg_color)
        glLineWidth(2)
        self.draw_rect(x, y, width, height, self.theme.border_color, filled=False)

        # Tabs
        header_height = 35
        tab_width = width / len(tabs) if tabs else 10

        for i, tab_name in enumerate(tabs):
            tab_x = x + i * tab_width
            is_active = i == current_tab

            bg_color = (0.2, 0.3, 0.4, 0.9) if is_active else (0.1, 0.15, 0.2, 0.8)
            self.draw_rect(tab_x, y, tab_width, header_height, bg_color)

            if is_active:
                glLineWidth(3)
                glColor4f(0.4, 0.8, 1.0, 1.0)
                glBegin(GL_LINES)
                glVertex2f(tab_x, y + header_height)
                glVertex2f(tab_x + tab_width, y + header_height)
                glEnd()

            color = (255, 255, 255) if is_active else (150, 150, 150)

            # Center text
            _, w, h = self.text_cache.get_text_data(tab_name, "default", color)
            text_x = tab_x + (tab_width - w) // 2
            text_y = y + (header_height - h) // 2
            self.text_cache.render(tab_name, int(text_x), int(text_y), "default", color)

        self.end_2d()

        # Render Content
        if content_data:
            content_data["position"] = (x + 10, y + header_height + 10)
            content_data["width"] = width - 20
            content_data["visible"] = True

            if content_key == "educational":
                self.render_educational_panel(content_data)
            elif content_key == "checklist":
                self.render_immersion_checklist(content_data)
            elif content_key == "history":
                self.render_historical_events(content_data)
            elif content_key == "planets":  # New Planet Selector
                self.render_planet_selector(content_data)
            elif content_key == "missions":
                self.render_mission_list(content_data)

    def render_mission_list(self, data: dict[str, Any]) -> None:
        """Render the list of famous space missions."""
        if not (data is not None):
            raise ValueError("data must be provided")
        if not data.get("visible", False):
            return

        x, y = data.get("position", (0, 0))
        missions = data.get("missions", [])

        self.begin_2d()
        current_y = y
        self.text_cache.render(
            "NASA Famous Missions", x, current_y, "default", self.theme.text_highlight
        )
        current_y += 35

        for mission in missions:
            name = mission.get("name", "Unknown")
            launch = mission.get("launch_date", "")
            desc = mission.get("description", "")
            mission_type = mission.get("mission_type", "")
            destinations = mission.get("destinations", "")
            highlights = mission.get("science_highlights", "")

            # Mission Title
            self.text_cache.render(name, x, current_y, "default", (255, 255, 100))
            current_y += 28

            # Launch Date
            self.text_cache.render(
                f"Launched: {launch}", x + 10, current_y, "small", (150, 200, 255)
            )
            current_y += 20

            if mission_type:
                self.text_cache.render(
                    mission_type, x + 10, current_y, "small", (160, 255, 180)
                )
                current_y += 18

            # Description (wrapped)
            words = desc.split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                if len(test_line) > 40:
                    self.text_cache.render(
                        line, x + 15, current_y, "small", (220, 220, 220)
                    )
                    current_y += 18
                    line = word
                else:
                    line = test_line
            if line:
                self.text_cache.render(
                    line, x + 15, current_y, "small", (220, 220, 220)
                )
                current_y += 25

            if destinations:
                self.text_cache.render(
                    f"Targets: {destinations}",
                    x + 10,
                    current_y,
                    "small",
                    (255, 210, 120),
                )
                current_y += 18

            if highlights:
                self.text_cache.render(
                    f"Science: {highlights}",
                    x + 10,
                    current_y,
                    "small",
                    (190, 220, 255),
                )
                current_y += 24

        self.end_2d()

    def render_educational_panel(self, edu_data: dict[str, Any]) -> None:
        """Render educational information about selected body."""
        if not (edu_data is not None):
            raise ValueError("edu_data must be provided")
        if not edu_data.get("visible", False):
            return

        x, y = edu_data.get("position", (20, 20))
        # width = edu_data.get("width", 350) # unused
        body_name = edu_data.get("body_name")
        properties = edu_data.get("properties", {})
        current_fact = edu_data.get("current_fact")

        if not body_name:
            return

        self.begin_2d()

        line_height = 18
        # num_lines = 2 + len(properties) + (3 if current_fact else 0) # unused
        # height = num_lines * line_height + 20 # unused

        current_y = y
        self.text_cache.render(
            body_name, x, current_y, "default", self.theme.text_highlight
        )
        current_y += line_height + 5

        for key, value in properties.items():
            text = f"{key}: {value}"
            if len(text) > 45:
                text = text[:42] + "..."
            self.text_cache.render(text, x, current_y, "small", (220, 220, 220))
            current_y += line_height

        if current_fact:
            current_y += 5
            self.text_cache.render(
                "Did you know?", x, current_y, "small", (255, 255, 100)
            )
            current_y += line_height

            words = current_fact.split()
            line = ""
            for word in words:
                test_line = f"{line} {word}".strip()
                if len(test_line) > 45:
                    self.text_cache.render(line, x, current_y, "small", (180, 220, 180))
                    current_y += line_height
                    line = word
                else:
                    line = test_line
            if line:
                self.text_cache.render(line, x, current_y, "small", (180, 220, 180))

        self.end_2d()

    def render_historical_events(self, events_data: dict[str, Any]) -> None:
        """Render list of historical events."""
        if not (events_data is not None):
            raise ValueError("events_data must be provided")
        if not events_data.get("visible", False):
            return

        x, y = events_data.get("position", (20, 450))
        events = events_data.get("events", [])

        self.begin_2d()
        line_height = 18

        current_y = y
        self.text_cache.render(
            "Historical Events", x, current_y, "default", (255, 200, 100)
        )
        current_y += line_height + 5

        for event in events[:5]:
            event_title = f"{event.get('year', '')}: {event.get('title', 'Unknown')}"
            if len(event_title) > 50:
                event_title = event_title[:47] + "..."
            self.text_cache.render(event_title, x, current_y, "small", (255, 255, 100))
            current_y += line_height

            description = event.get("description", "")
            if len(description) > 55:
                description = description[:52] + "..."
            self.text_cache.render(
                description, x + 10, current_y, "small", (200, 200, 200)
            )
            current_y += line_height + 3

        self.end_2d()

    def render_immersion_checklist(self, checklist_data: dict[str, Any]) -> None:
        """Render the immersion checklist."""
        if not (checklist_data is not None):
            raise ValueError("checklist_data must be provided")
        if not checklist_data.get("visible", False):
            return

        x, y = checklist_data.get("position", (20, 240))
        tasks = checklist_data.get("tasks", [])
        completed, total = checklist_data.get("progress", (0, len(tasks)))

        self.begin_2d()
        line_height = 18
        current_y = y

        title = f"Immersion Guide ({completed}/{total})"
        self.text_cache.render(title, x, current_y, "default", (160, 230, 255))
        current_y += line_height + 4

        for task in tasks:
            is_done = task.get("completed")
            marker = "✓" if is_done else "•"
            color = (140, 220, 170) if is_done else (240, 210, 160)

            self.text_cache.render(
                f"{marker} {task.get('title', '')}", x, current_y, "small", color
            )
            current_y += line_height

            text_color = (200, 220, 240) if is_done else (230, 230, 230)
            self.text_cache.render(
                task.get("description", ""), x + 14, current_y, "small", text_color
            )
            current_y += line_height

        self.end_2d()

    def render_unified_controls(
        self, ctrl_data: dict[str, Any], time_data: dict[str, Any]
    ) -> None:
        """Render the unified control panel."""
        if not (ctrl_data is not None):
            raise ValueError("ctrl_data must be provided")
        if not ctrl_data.get("visible", False):
            return

        x, y = ctrl_data.get("position", (0, 0))
        width = ctrl_data.get("width", 800)
        height = ctrl_data.get("height", 100)
        checkboxes = ctrl_data.get("checkboxes", [])
        modes = ctrl_data.get("modes", [])
        curr_mode = ctrl_data.get("current_mode_index", 0)

        self.begin_2d()

        # BG
        self.draw_rect(x, y, width, height, self.theme.bg_color)
        glColor4f(0.4, 0.8, 1.0, 0.6)
        glLineWidth(2)
        glBegin(GL_LINES)
        glVertex2f(x, y)
        glVertex2f(x + width, y)
        glEnd()

        # 1. Navigation Modes
        mode_x, mode_y = x + 20, y + 20
        self.text_cache.render(
            "NAVIGATION", mode_x, mode_y, "small", self.theme.text_highlight
        )
        mode_y += 25
        for i, mode in enumerate(modes):
            color = (100, 255, 100) if i == curr_mode else self.theme.text_dim
            prefix = "● " if i == curr_mode else "○ "
            self.text_cache.render(f"{prefix}{mode}", mode_x, mode_y, "small", color)
            mode_x += 80

        # 2. View Settings
        set_x = x + width - 350
        set_y = y + 20
        self.text_cache.render(
            "VIEW SETTINGS", set_x, set_y, "small", self.theme.text_highlight
        )
        set_y += 35
        col1, col2 = set_x, set_x + 160

        for i, cb in enumerate(checkboxes):
            cx = col1 if i % 2 == 0 else col2
            cy = set_y + (i // 2) * 30
            color = (255, 255, 255) if cb.checked else self.theme.text_dim
            marker = "☑" if cb.checked else "☐"
            self.text_cache.render(f"{marker} {cb.label}", cx, cy, "small", color)

        # 3. Action Buttons
        btn_x = x + 20
        btn_y = y + 80
        buttons = ctrl_data.get("buttons", [])
        for btn in buttons:
            self.draw_rect(btn_x, btn_y, btn.width, 30, (0.2, 0.4, 0.6, 0.8))
            _, w, h = self.text_cache.get_text_data(btn.label, "small", (255, 255, 255))
            tx = btn_x + (btn.width - w) // 2
            ty = btn_y + (30 - h) // 2
            self.text_cache.render(
                btn.label, int(tx), int(ty), "small", (255, 255, 255)
            )
            btn_x += btn.width + 10

        self.end_2d()

    def render_planet_selector(self, data: dict[str, Any]) -> None:
        """Render a clickable list of planets."""
        if not (data is not None):
            raise ValueError("data must be provided")
        if not data.get("visible", False):
            return

        x, y = data.get("position", (0, 0))

        bodies = data.get("bodies", [])

        self.begin_2d()
        current_y = y
        self.text_cache.render(
            "Select Body", x, current_y, "default", self.theme.text_highlight
        )
        current_y += 30

        for body in bodies:
            name = body.get("name", "Unknown")
            is_selected = body.get("selected", False)
            color = (100, 255, 100) if is_selected else (200, 200, 200)
            prefix = ">> " if is_selected else "   "

            self.text_cache.render(f"{prefix}{name}", x, current_y, "small", color)
            current_y += 25

        self.end_2d()

    def render_speed_indicator(self, time_warp: float) -> None:
        """Render a visual bar indicating time speed."""
        if not (time_warp is not None):
            raise ValueError("time_warp must be provided")
        self.begin_2d()

        w, h = 200, 10
        x = (self.window_width - w) // 2
        y = self.window_height - 120

        # Background
        self.draw_rect(x, y, w, h, (0.2, 0.2, 0.2, 0.8))

        # Bar
        if time_warp == 0:
            fill = 0.0
        else:
            # Approx 86400 is 1 day/sec. Max speed usually higher.
            # Log10(86400) ~ 5. Let's say range 0 to 7.
            fill = min(
                1.0,
                max(
                    0.0,
                    (math.log10(abs(time_warp)) if abs(time_warp) > 1 else 0) / 7.0,
                ),
            )

        fill_w = w * fill
        color = (0.2, 0.8, 0.2, 0.8) if time_warp > 0 else (0.8, 0.2, 0.2, 0.8)

        self.draw_rect(x, y, fill_w, h, color)

        text = f"{time_warp:.0f}x"
        _, tw, th = self.text_cache.get_text_data(text, "small", (255, 255, 255))
        self.text_cache.render(
            text, x + (w - tw) // 2, y - 20, "small", (255, 255, 255)
        )

        self.end_2d()

    def render_compass(self, camera_yaw: float) -> None:
        """Render a small N compass."""
        if not (camera_yaw is not None):
            raise ValueError("camera_yaw must be provided")
        self.begin_2d()

        cx, cy = self.window_width - 50, 50
        radius = 30

        # Circle
        glDisable(GL_TEXTURE_2D)
        glColor4f(0.3, 0.3, 0.3, 0.5)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        for i in range(33):
            angle = i * 2 * math.pi / 32
            glVertex2f(cx + math.cos(angle) * radius, cy + math.sin(angle) * radius)
        glEnd()

        # Needle (North)
        angle = -camera_yaw  # assuming radians

        nx = cx + math.sin(angle) * (radius - 5)
        ny = cy - math.cos(angle) * (radius - 5)

        glLineWidth(2)
        glColor4f(1.0, 0.2, 0.2, 0.9)
        glBegin(GL_LINES)
        glVertex2f(cx, cy)
        glVertex2f(nx, ny)
        glEnd()

        self.text_cache.render("N", int(nx), int(ny), "small", (255, 100, 100))

        self.end_2d()

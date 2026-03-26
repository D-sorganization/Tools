"""
Input Handler
=============

Handles keyboard and mouse input for the simulation.
Provides a clean interface between raw input events and simulation actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    import pygame
    from pygame.locals import (
        K_0,
        K_1,
        K_2,
        K_3,
        K_4,
        K_5,
        K_6,
        K_7,
        K_8,
        K_9,
        K_EQUALS,
        K_ESCAPE,
        K_HOME,
        K_KP_MINUS,
        K_KP_PLUS,
        K_MINUS,
        K_PLUS,
        K_SPACE,
        KEYDOWN,
        MOUSEBUTTONDOWN,
        MOUSEBUTTONUP,
        MOUSEMOTION,
        MOUSEWHEEL,
        QUIT,
        K_c,
        K_f,
        K_g,
        K_h,
        K_i,
        K_l,
        K_o,
        K_r,
        K_t,
    )

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class InputAction(Enum):
    """Actions that can be triggered by input."""

    QUIT = "quit"
    PAUSE = "pause"
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"
    REVERSE_TIME = "reverse_time"
    RESET_VIEW = "reset_view"
    TOGGLE_ORBITS = "toggle_orbits"
    TOGGLE_LABELS = "toggle_labels"
    TOGGLE_INFO = "toggle_info"
    TOGGLE_GRID = "toggle_grid"
    TOGGLE_HELP = "toggle_help"
    CYCLE_CAMERA = "cycle_camera"
    FOCUS_SELECTED = "focus_selected"
    PLAN_TRAJECTORY = "plan_trajectory"
    SELECT_SUN = "select_sun"
    SELECT_PLANET_1 = "select_planet_1"
    SELECT_PLANET_2 = "select_planet_2"
    SELECT_PLANET_3 = "select_planet_3"
    SELECT_PLANET_4 = "select_planet_4"
    SELECT_PLANET_5 = "select_planet_5"
    SELECT_PLANET_6 = "select_planet_6"
    SELECT_PLANET_7 = "select_planet_7"
    SELECT_PLANET_8 = "select_planet_8"
    SELECT_PLANET_9 = "select_planet_9"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    ORBIT_CAMERA = "orbit_camera"
    PAN_CAMERA = "pan_camera"


@dataclass
class MouseState:
    """Current state of the mouse."""

    position: tuple[int, int] = (0, 0)
    left_pressed: bool = False
    right_pressed: bool = False
    middle_pressed: bool = False
    scroll_delta: int = 0
    drag_delta: tuple[int, int] = (0, 0)


@dataclass
class KeyBinding:
    """A keyboard binding configuration."""

    key: int
    action: InputAction
    modifiers: int = 0  # KMOD_* flags
    description: str = ""


class InputHandler:
    """
    Handles all user input and converts to simulation actions.

    Provides customizable key bindings and mouse controls.
    """

    def __init__(self) -> None:
        """Initialize the input handler."""
        self.mouse_state = MouseState()
        self._action_callbacks: dict[InputAction, list[Callable[..., Any]]] = {}
        self._key_bindings: list[KeyBinding] = []
        self._last_mouse_pos: tuple[int, int] = (0, 0)

        # Set up default bindings
        self._setup_default_bindings()

    def _setup_default_bindings(self) -> None:
        """Set up default keyboard bindings."""
        if not PYGAME_AVAILABLE:
            return

        default_bindings = [
            KeyBinding(K_ESCAPE, InputAction.QUIT, description="Quit"),
            KeyBinding(K_SPACE, InputAction.PAUSE, description="Pause/Resume"),
            KeyBinding(K_EQUALS, InputAction.SPEED_UP, description="Speed up time"),
            KeyBinding(K_PLUS, InputAction.SPEED_UP),
            KeyBinding(K_KP_PLUS, InputAction.SPEED_UP),
            KeyBinding(K_MINUS, InputAction.SLOW_DOWN, description="Slow down time"),
            KeyBinding(K_KP_MINUS, InputAction.SLOW_DOWN),
            KeyBinding(K_r, InputAction.REVERSE_TIME, description="Reverse time"),
            KeyBinding(K_HOME, InputAction.RESET_VIEW, description="Reset view"),
            KeyBinding(K_o, InputAction.TOGGLE_ORBITS, description="Toggle orbits"),
            KeyBinding(K_l, InputAction.TOGGLE_LABELS, description="Toggle labels"),
            KeyBinding(K_i, InputAction.TOGGLE_INFO, description="Toggle info panel"),
            KeyBinding(K_g, InputAction.TOGGLE_GRID, description="Toggle grid"),
            KeyBinding(K_h, InputAction.TOGGLE_HELP, description="Toggle help"),
            KeyBinding(K_c, InputAction.CYCLE_CAMERA, description="Cycle camera mode"),
            KeyBinding(
                K_f, InputAction.FOCUS_SELECTED, description="Focus on selected"
            ),
            KeyBinding(K_t, InputAction.PLAN_TRAJECTORY, description="Plan trajectory"),
            KeyBinding(K_0, InputAction.SELECT_SUN, description="Select Sun"),
            KeyBinding(K_1, InputAction.SELECT_PLANET_1, description="Select Mercury"),
            KeyBinding(K_2, InputAction.SELECT_PLANET_2, description="Select Venus"),
            KeyBinding(K_3, InputAction.SELECT_PLANET_3, description="Select Earth"),
            KeyBinding(K_4, InputAction.SELECT_PLANET_4, description="Select Mars"),
            KeyBinding(K_5, InputAction.SELECT_PLANET_5, description="Select Jupiter"),
            KeyBinding(K_6, InputAction.SELECT_PLANET_6, description="Select Saturn"),
            KeyBinding(K_7, InputAction.SELECT_PLANET_7, description="Select Uranus"),
            KeyBinding(K_8, InputAction.SELECT_PLANET_8, description="Select Neptune"),
            KeyBinding(K_9, InputAction.SELECT_PLANET_9, description="Select Pluto"),
        ]

        self._key_bindings = default_bindings

    def register_callback(
        self, action: InputAction, callback: Callable[..., Any]
    ) -> None:
        """
        Register a callback for an input action.

        Args:
            action: The action to listen for
            callback: Function to call when action is triggered
        """
        if not (action is not None):
            raise ValueError("action must be provided")
        if action not in self._action_callbacks:
            self._action_callbacks[action] = []
        self._action_callbacks[action].append(callback)

    def unregister_callback(
        self, action: InputAction, callback: Callable[..., Any]
    ) -> None:
        """Remove a callback for an action."""
        if (
            action in self._action_callbacks
            and callback in self._action_callbacks[action]
        ):
            self._action_callbacks[action].remove(callback)

    def _trigger_action(self, action: InputAction, **kwargs: Any) -> None:
        """Trigger all callbacks for an action."""
        if action in self._action_callbacks:
            for callback in self._action_callbacks[action]:
                callback(**kwargs)

    def process_events(self) -> bool:
        """
        Process all pending pygame events.

        Returns:
            False if should quit, True otherwise
        """
        if not PYGAME_AVAILABLE:
            return True

        # Reset per-frame state
        self.mouse_state.scroll_delta = 0
        self.mouse_state.drag_delta = (0, 0)

        for event in pygame.event.get():
            if event.type == QUIT:
                self._trigger_action(InputAction.QUIT)
                return False

            elif event.type == KEYDOWN:
                if not self._handle_key_down(event.key, event.mod):
                    return False

            elif event.type == MOUSEBUTTONDOWN:
                self._handle_mouse_button_down(event.button, event.pos)

            elif event.type == MOUSEBUTTONUP:
                self._handle_mouse_button_up(event.button, event.pos)

            elif event.type == MOUSEMOTION:
                self._handle_mouse_motion(event.pos, event.rel)

            elif event.type == MOUSEWHEEL:
                self._handle_mouse_wheel(event.y)

        return True

    def _handle_key_down(self, key: int, modifiers: int) -> bool:
        """
        Handle a key press.

        Returns:
            False if should quit
        """
        if not (key is not None):
            raise ValueError("key must be provided")
        for binding in self._key_bindings:
            if binding.key == key and (
                binding.modifiers == 0 or (modifiers & binding.modifiers)
            ):
                self._trigger_action(binding.action)

                if binding.action == InputAction.QUIT:
                    return False

        return True

    def _handle_mouse_button_down(self, button: int, pos: tuple[int, int]) -> None:
        """Handle mouse button press."""
        if not (button is not None):
            raise ValueError("button must be provided")
        self.mouse_state.position = pos

        if button == 1:  # Left
            self.mouse_state.left_pressed = True
        elif button == 2:  # Middle
            self.mouse_state.middle_pressed = True
        elif button == 3:  # Right
            self.mouse_state.right_pressed = True

        self._last_mouse_pos = pos

    def _handle_mouse_button_up(self, button: int, pos: tuple[int, int]) -> None:
        """Handle mouse button release."""
        if not (button is not None):
            raise ValueError("button must be provided")
        self.mouse_state.position = pos

        if button == 1:
            self.mouse_state.left_pressed = False
        elif button == 2:
            self.mouse_state.middle_pressed = False
        elif button == 3:
            self.mouse_state.right_pressed = False

    def _handle_mouse_motion(self, pos: tuple[int, int], rel: tuple[int, int]) -> None:
        """Handle mouse movement."""
        if not (pos is not None):
            raise ValueError("pos must be provided")
        self.mouse_state.position = pos
        self.mouse_state.drag_delta = rel

        if self.mouse_state.left_pressed:
            self._trigger_action(InputAction.ORBIT_CAMERA, delta=rel)
        elif self.mouse_state.right_pressed:
            self._trigger_action(InputAction.PAN_CAMERA, delta=rel)

    def _handle_mouse_wheel(self, delta: int) -> None:
        """Handle mouse wheel scroll."""
        if not (delta is not None):
            raise ValueError("delta must be provided")
        self.mouse_state.scroll_delta = delta

        if delta > 0:
            self._trigger_action(InputAction.ZOOM_IN, amount=delta)
        elif delta < 0:
            self._trigger_action(InputAction.ZOOM_OUT, amount=-delta)

    def get_bindings_for_display(self) -> list[tuple[str, str]]:
        """
        Get list of key bindings formatted for display.

        Returns:
            List of (key_name, description) tuples
        """
        result = []

        if not PYGAME_AVAILABLE:
            return []

        for binding in self._key_bindings:
            if binding.description:
                key_name = pygame.key.name(binding.key).upper()
                result.append((key_name, binding.description))

        return result

    def is_dragging(self) -> bool:
        """Check if user is currently dragging."""
        return self.mouse_state.left_pressed or self.mouse_state.right_pressed

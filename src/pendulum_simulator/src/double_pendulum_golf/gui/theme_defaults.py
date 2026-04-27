"""First-launch theme seeding for the pendulum simulator.

Background
----------
The pendulum simulator uses the fleet ``ThemeManager`` from the shared
``theme`` package (``Tools/src/shared/python/theme``). That manager
falls back to the ``"Light"`` theme for any application that has no
saved preference. We override that single behaviour for *this* app by
seeding ``"Dark"`` into ``QSettings`` on the very first launch — but
only if the user has never expressed a preference.

Why a separate module?
----------------------
- DRY: the seeding logic lives in exactly one place; both the
  ``MainWindow`` startup path and any future entry point (e.g. a CLI
  subcommand) call ``ensure_default_theme_seeded()`` and get the same
  result.
- DbC: idempotent + non-destructive. After the first call the
  ``first_launch_initialized`` flag is set, so subsequent calls never
  overwrite a user choice. Even if the flag goes missing, an existing
  ``theme`` value is still respected.
- Law of Demeter: the rest of the codebase doesn't need to know which
  QSettings keys are involved or how the fleet ThemeManager resolves
  the effective theme.

Public API
----------
- ``DEFAULT_THEME_NAME``: the constant ``"Dark"``.
- ``ensure_default_theme_seeded() -> str``: seeds the default theme
  if and only if no user preference exists. Returns the theme name
  the application should now use.
"""

from __future__ import annotations

import logging
from typing import Final

from PyQt6.QtCore import QSettings

logger = logging.getLogger(__name__)

DEFAULT_THEME_NAME: Final[str] = "Dark"

_SETTINGS_ORG: Final[str] = "D-sorganization"
_SETTINGS_APP: Final[str] = "PendulumSimulator"
_THEME_KEY: Final[str] = "theme"
_INITIAL_FLAG_KEY: Final[str] = "first_launch_initialized"


def ensure_default_theme_seeded() -> str:
    """Seed ``DEFAULT_THEME_NAME`` into QSettings on first launch only.

    Returns
    -------
    str
        The theme name that the application should use after this call.
        On the very first launch this is ``DEFAULT_THEME_NAME``. After
        the user has chosen any theme it is whatever they chose.

    Postconditions
    --------------
    - The QSettings key ``"first_launch_initialized"`` is always set
      after this call.
    - The ``"theme"`` key is set if and only if it was unset *and*
      the first-launch flag was unset on entry.
    - An existing ``"theme"`` value is never overwritten.
    """
    settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    has_initial_flag = settings.value(_INITIAL_FLAG_KEY) is not None
    existing_theme = settings.value(_THEME_KEY)

    if has_initial_flag or existing_theme is not None:
        # Already initialised or user has a preference — leave it alone
        # but make sure the flag is present so we don't try again next launch.
        if not has_initial_flag:
            settings.setValue(_INITIAL_FLAG_KEY, "1")
            settings.sync()
        active = str(existing_theme) if existing_theme is not None else DEFAULT_THEME_NAME
        logger.debug(
            "Theme already initialised (theme=%s, flag=%s); not seeding",
            active,
            has_initial_flag,
        )
        return active

    # First-ever launch: write the dark default and the flag together.
    settings.setValue(_THEME_KEY, DEFAULT_THEME_NAME)
    settings.setValue(_INITIAL_FLAG_KEY, "1")
    settings.sync()
    logger.info(
        "First-launch theme seeded: %s (settings=%s/%s)",
        DEFAULT_THEME_NAME,
        _SETTINGS_ORG,
        _SETTINGS_APP,
    )
    return DEFAULT_THEME_NAME

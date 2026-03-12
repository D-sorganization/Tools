"""
standalone/base.py
==================
Base class for standalone single-reactor flowsheet models.

All three standalone models (Gasifier, PEM, TRC) share identical __init__,
setup_thermo, and calculate logic.  Subclasses only need to implement
build_flowsheet() which is specific to each reactor stage.
"""

from __future__ import annotations

import logging
from abc import abstractmethod

from dwsim_model.constants import COMPOUNDS_STANDARD, DEFAULT_PROPERTY_PACKAGE
from dwsim_model.core import FlowsheetBuilder, get_automation

logger = logging.getLogger(__name__)


class StandaloneBase:
    """
    Shared base class for Gasifier, PEM, and TRC standalone flowsheet models.

    Handles common setup: automation binding, builder construction, compound
    loading, property package assignment, and the calculate/save interface.

    Subclasses must implement :meth:`build_flowsheet`.
    """

    def __init__(self, compound_set: list | None = None) -> None:
        self.automation = get_automation()
        self.builder = FlowsheetBuilder()
        self._is_built = False
        self._compounds = compound_set or COMPOUNDS_STANDARD

    def setup_thermo(self) -> None:
        """Configure standard PR properties and required components."""
        for c in self._compounds:
            self.builder.add_compound(c)
        self.builder.add_property_package(DEFAULT_PROPERTY_PACKAGE)

    @abstractmethod
    def build_flowsheet(self) -> None:
        """Construct the reactor stage and its immediate streams.

        Subclasses must call ``self._is_built = True`` at the end.
        """

    def _safe_connect(self, src, tgt, p1: int = 0, p2: int = 0) -> None:
        """Connect two DWSIM objects, logging a warning on failure."""
        try:
            self.builder.connect(src, tgt, p1, p2)
        except Exception as e:  # noqa: BLE001 — DWSIM COM errors are opaque
            logger.warning(f"Failed to connect isolated node: {src} -> {tgt} ({e})")

    def calculate(self) -> None:
        """Run the flowsheet solver.  Raises if build_flowsheet was not called."""
        if not self._is_built:
            raise RuntimeError("Flowsheet must be built before calculating.")
        self.builder.calculate()

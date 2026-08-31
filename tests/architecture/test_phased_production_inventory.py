"""Unit tests for module inventory classification and phased engineering.

Issue: #4722.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_tools_module_inventory import (
    _classification,
    _domain,
)
from scripts.tools_module_inventory_contract import (
    AUTHORITY,
    MODULE_INVENTORY_SCHEMA_VERSION,
    RELEASE_STATUS,
    load_inventory,
)
from scripts.tools_module_inventory_storage import read_inventory

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "manuals" / "tools" / "manifests" / "module-inventory.json"


@pytest.mark.unit
class TestModuleInventoryClassification:
    """Tests for deterministic module inventory classification and domain mapping."""

    @pytest.mark.parametrize(
        ("path_str", "expected_family_prefix", "expected_owner"),
        [
            (
                "src/shared/python/plotting/identity.py",
                "shared-plotting",
                "Plotting and visualization maintainers",
            ),
            (
                "src/shared/python/plot_engine/pyqt6_widget.py",
                "plot-engine",
                "Plot engine maintainers",
            ),
            (
                "src/shared/python/plot_theme/themes.py",
                "plot-theme",
                "Plot theme maintainers",
            ),
            (
                "src/rate_of_closure/application/camera.py",
                "rate-of-closure",
                "Rate of Closure maintainers",
            ),
            (
                "src/shared/python/swing_sim/flight/inverse_contract.py",
                "swing-simulation",
                "Swing simulation maintainers",
            ),
        ],
    )
    def test_domain_mapping_rules(
        self, path_str: str, expected_family_prefix: str, expected_owner: str
    ) -> None:
        family, owner = _domain(Path(path_str))
        assert family == expected_family_prefix
        assert owner == expected_owner

    def test_calculation_classification_markers(self) -> None:
        path = Path("src/shared/python/signal_toolkit/filter_operations.py")
        text = "def apply_lowpass_filter(data): return data"
        classification, basis = _classification(path, text)
        assert classification == "calculation"
        assert "filter" in basis

    def test_non_calculation_classification(self) -> None:
        path = Path("src/shared/python/plotting/identity.py")
        text = "class PlotIdentity: pass"
        classification, basis = _classification(path, text)
        assert classification == "non-calculation"
        assert basis == "no-conservative-calculation-signal"


@pytest.mark.unit
class TestPhasedProductionManifest:
    """Tests for phased engineering production manifest integrity and boundaries."""

    def test_production_manifest_contract_envelope(self) -> None:
        payload = read_inventory(ROOT, MANIFEST)
        view = load_inventory(payload)
        assert view.schema_version == MODULE_INVENTORY_SCHEMA_VERSION
        assert payload["authority"] == AUTHORITY
        assert payload["release_status"] == RELEASE_STATUS
        assert len(payload["blockers"]) > 0
        assert len(payload["families"]) > 0

    def test_phased_production_batches_and_ownership(self) -> None:
        payload = read_inventory(ROOT, MANIFEST)
        entries = payload["entries"]
        assert len(entries) > 0

        # Every entry has maintainer, classification, and states
        for entry in entries:
            assert entry["maintainer"]
            assert entry["classification"] in {"calculation", "non-calculation"}
            assert "states" in entry
            assert "traceability" in entry
            assert "risk_tags" in entry

        # Verify families have explicit classification and rationale
        for fam in payload["families"]:
            assert fam["id"]
            assert fam["classification"] in {"calculation", "non-calculation"}
            assert fam["maintainer"]
            assert fam["rationale"]

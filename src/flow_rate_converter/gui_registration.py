"""
Flow Rate Converter - GUI Registration
======================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Flow Rate Converter",
    "description": "Convert between mass, molar, and volumetric flow rate units",
    "category": "utilities",
    "version": "1.0.0",
    "entry_point": "flow_rate_converter.ui.pyqt6.main_window:FlowRateConverterWindow",
    "web_entry_point": "web/src/components/FlowRateConverter.tsx",
    "icon": "exchange",
    "keywords": [
        "flow rate",
        "unit conversion",
        "mass flow",
        "molar flow",
        "volumetric flow",
        "SCFM",
        "ACFM",
    ],
    "dependencies": {
        "required": ["PyQt6"],
        "optional": [],
    },
    "conversion_types": [
        "Mass to Mass",
        "Molar to Molar",
        "Mass to Molar",
        "Molar to Mass",
        "Volumetric (Actual)",
        "Standard Volumetric (SCFM/Nm3)",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA

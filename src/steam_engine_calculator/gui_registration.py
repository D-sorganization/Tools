"""
Steam Engine Calculator - GUI Registration
===========================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    'name': 'Steam Engine Calculator',
    'description': 'Calculate thermodynamic properties of steam/water using CoolProp, Cantera, or simplified correlations',
    'category': 'thermodynamics',
    'version': '1.0.0',
    'entry_point': 'steam_engine_calculator.ui.pyqt6.main_window:SteamEngineCalculatorWindow',
    'web_entry_point': 'web/src/components/SteamEngineCalculator.tsx',
    'icon': 'steam',
    'keywords': ['steam', 'water', 'thermodynamics', 'properties', 'enthalpy', 'entropy'],
    'dependencies': {
        'required': ['PyQt6'],
        'optional': ['CoolProp', 'cantera', 'numpy'],
    },
    'calculation_modes': [
        'Temperature & Pressure',
        'Saturated from Temperature',
        'Saturated from Pressure',
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA

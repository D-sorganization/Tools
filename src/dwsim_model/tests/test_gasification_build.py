import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from dwsim_model.core import FlowsheetBuilder


def test_add_compounds():
    builder = FlowsheetBuilder()
    builder.add_compound("Methane")
    assert "Methane" in [c.Name for c in builder.sim.SelectedCompounds.Values]


def test_add_property_package():
    builder = FlowsheetBuilder()
    pkg = builder.add_property_package("Peng-Robinson (PR)")
    assert pkg is not None
    # PropertyPackages on sim itself are usually dictionaries in older versions,
    # but might just be in the solver
    assert len(builder.sim.PropertyPackages) > 0


def test_create_operations():
    builder = FlowsheetBuilder()

    # Material Streams
    _ms = builder.add_object("MaterialStream", "FeedStream")
    assert builder.materials["FeedStream"] is not None

    # 1. Downdraft Gasifier -> RCT_Conversion or similar
    _gd = builder.add_object("RCT_Conversion", "Downdraft_Gasifier")
    assert builder.operations["Downdraft_Gasifier"] is not None

    # 2. PEM -> Equilibrium Reactor
    _pem = builder.add_object("RCT_Equilibrium", "PEM")
    assert builder.operations["PEM"] is not None

    # 3. TRC -> PFR Reactor
    _trc = builder.add_object("RCT_PFR", "TRC")
    assert builder.operations["TRC"] is not None

    # 4. Quench vessel -> Vessel or Cooler
    _quench = builder.add_object("Vessel", "Quench_Vessel")
    assert builder.operations["Quench_Vessel"] is not None

    # 5. Baghouse -> SolidSeparator
    _baghouse = builder.add_object("SolidSeparator", "Baghouse")
    assert builder.operations["Baghouse"] is not None

    # 6. Scrubber -> ComponentSeparator
    _scrubber = builder.add_object("ComponentSeparator", "Scrubber")
    assert builder.operations["Scrubber"] is not None

    # 7. Blower / Compressor -> Compressor
    _blower = builder.add_object("Compressor", "Blower")
    assert builder.operations["Blower"] is not None

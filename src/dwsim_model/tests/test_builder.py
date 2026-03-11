import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dwsim_model.core import FlowsheetBuilder


def test_builder_init():
    builder = FlowsheetBuilder()
    assert builder.sim is not None


def test_builder_add_compound():
    builder = FlowsheetBuilder()
    builder.add_compound("Methane")
    builder.add_compound("Oxygen")
    assert len(list(builder.sim.SelectedCompounds.Values)) == 2


def test_builder_property_package():
    builder = FlowsheetBuilder()
    _prop_pack = builder.add_property_package("Peng-Robinson (PR)")
    packages = list(builder.sim.PropertyPackages.Values)
    assert len(packages) > 0

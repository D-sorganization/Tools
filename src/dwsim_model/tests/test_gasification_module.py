import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from dwsim_model.core import FlowsheetBuilder
from dwsim_model.gasification import GasificationFlowsheet, ReactorMode


@pytest.fixture(scope="module")
def flowsheet():
    """Module-scoped flowsheet instance to prevent multiple Initialization issues"""
    b = FlowsheetBuilder()
    gf = GasificationFlowsheet(builder=b, mode=ReactorMode.MIXED)
    return gf


def test_thermodynamics_setup(flowsheet):
    """Test correctly configuring property packages and compounds."""
    flowsheet.setup_thermo()
    compounds = {c.Name for c in flowsheet.builder.sim.SelectedCompounds.Values}
    assert "Carbon monoxide" in compounds
    assert "Methane" in compounds


def test_flowsheet_builder(flowsheet):
    """Test all required gasification operations are built"""
    flowsheet.build_flowsheet()

    ops = flowsheet.builder.operations
    assert "Downdraft_Gasifier" in ops
    assert "PEM_Reactor" in ops
    assert "TRC_Reactor" in ops
    assert "Quench_Vessel" in ops
    assert "Baghouse" in ops
    assert "Scrubber" in ops
    assert "Blower" in ops

    mats = flowsheet.builder.materials
    assert "Quench_Water_Injection" in mats


def test_reactor_configuration(flowsheet):
    """Ensure private _configure_reactors handles setup gracefully."""
    flowsheet._configure_reactors()
    # Just ensure it doesn't crash prior to strict parameters being locked down
    assert True


def test_flowsheet_run_does_not_crash(flowsheet):
    """Test running the sequence executes safely (calculations will be incomplete)."""
    # Simply assert run completes without fatal program error.
    flowsheet.run()
    assert flowsheet._is_built


def test_custom_reactor_modes():
    """Test logic for alternative reactor modes."""
    # Test Equilibrium Mode
    gf_eq = GasificationFlowsheet(mode=ReactorMode.EQUILIBRIUM)
    rtypes_eq = gf_eq._get_reactor_types()
    assert rtypes_eq["gasifier"] == "RCT_Equilibrium"
    assert rtypes_eq["pem"] == "RCT_Equilibrium"
    assert rtypes_eq["trc"] == "RCT_Equilibrium"

    # Test Custom mode
    custom_dict = {"gasifier": "RCT_Conversion", "pem": "RCT_Gibbs", "trc": "RCT_PFR"}
    gf_custom = GasificationFlowsheet(
        mode=ReactorMode.CUSTOM, custom_reactors=custom_dict
    )
    rtypes_custom = gf_custom._get_reactor_types()
    assert rtypes_custom["pem"] == "RCT_Gibbs"
    assert rtypes_custom["gasifier"] == "RCT_Conversion"


def test_build_flowsheet_fails_when_critical_connection_fails(monkeypatch):
    gf = GasificationFlowsheet(builder=FlowsheetBuilder(), mode=ReactorMode.MIXED)

    def fail_connect(*_args, **_kwargs):
        raise RuntimeError("connect failed")

    monkeypatch.setattr(gf.builder, "connect", fail_connect)

    with pytest.raises(RuntimeError, match="Critical flowsheet connections failed"):
        gf.build_flowsheet()


def test_build_flowsheet_fails_when_reactor_configuration_fails(monkeypatch):
    from dwsim_model.chemistry import reactions

    gf = GasificationFlowsheet(builder=FlowsheetBuilder(), mode=ReactorMode.MIXED)

    def fail_gasifier(*_args, **_kwargs):
        raise RuntimeError("gasifier setup failed")

    monkeypatch.setattr(reactions, "configure_gasifier", fail_gasifier)

    with pytest.raises(RuntimeError, match="Gasifier reactor configuration failed"):
        gf.build_flowsheet()


def test_build_flowsheet_fails_when_config_application_fails(monkeypatch):
    from dwsim_model.config_loader import ConfigLoader

    gf = GasificationFlowsheet(builder=FlowsheetBuilder(), mode=ReactorMode.MIXED)

    def fail_apply(self, *_args, **_kwargs):
        raise RuntimeError("config apply failed")

    monkeypatch.setattr(ConfigLoader, "apply_to_flowsheet", fail_apply)

    with pytest.raises(RuntimeError, match="Failed to apply external config"):
        gf.build_flowsheet()


def test_load_config_prefers_injected_runtime_config(monkeypatch):
    observed = {}

    class RecordingLoader:
        def __init__(self, config_path=None, config_data=None):
            observed["config_path"] = config_path
            observed["config_data"] = config_data

        def load(self):
            observed["loaded"] = True
            return {"feeds": {}}

        def apply_to_flowsheet(self, builder, materials, energy_streams):
            observed["applied"] = (builder, materials, energy_streams)

    gf = GasificationFlowsheet(builder=FlowsheetBuilder(), mode=ReactorMode.MIXED)
    gf._injected_config = {"feeds": {"Gasifier_Biomass_Feed": {"mass_flow_kg_s": 7.5}}}

    monkeypatch.setattr("dwsim_model.gasification.ConfigLoader", RecordingLoader)

    gf._load_config()

    assert observed["config_data"] == gf._injected_config
    assert observed["loaded"] is True
    assert observed["applied"] == (
        gf.builder,
        gf.builder.materials,
        gf.builder.energy_streams,
    )

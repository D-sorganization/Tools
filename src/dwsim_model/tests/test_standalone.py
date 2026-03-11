import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from dwsim_model.standalone.gasifier_model import GasifierStandaloneFlowsheet
from dwsim_model.standalone.pem_model import PEMStandaloneFlowsheet
from dwsim_model.standalone.trc_model import TRCStandaloneFlowsheet


def test_gasifier_builds():
    m = GasifierStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    assert m._is_built
    # Verify core blocks exist
    ops = m.builder.operations
    assert "Downdraft_Gasifier" in ops
    assert "Gasifier_Heat_Loss_Block" in ops


def test_pem_builds():
    m = PEMStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    assert m._is_built
    ops = m.builder.operations
    assert "PEM_Reactor" in ops
    assert "PEM_DC_Block" in ops


def test_trc_builds():
    m = TRCStandaloneFlowsheet()
    m.setup_thermo()
    m.build_flowsheet()
    assert m._is_built
    ops = m.builder.operations
    assert "TRC_Reactor" in ops
    assert "TRC_Heat_Loss_Block" in ops

from __future__ import annotations

from dataclasses import dataclass

from dwsim_model.topology import (
    build_gasifier_stage,
    build_pem_stage,
    build_trc_stage,
)


@dataclass
class FakeNode:
    Name: str


class FakeBuilder:
    def __init__(self):
        self.materials: dict[str, FakeNode] = {}
        self.energy_streams: dict[str, FakeNode] = {}
        self.operations: dict[str, FakeNode] = {}
        self.connections: list[tuple[str, str, int, int]] = []

    def add_object(self, obj_type_name: str, name: str, _x: int = 0, _y: int = 0):
        node = FakeNode(name)
        if obj_type_name == "MaterialStream":
            self.materials[name] = node
        elif obj_type_name == "EnergyStream":
            self.energy_streams[name] = node
        else:
            self.operations[name] = node
        return node


def test_gasifier_stage_topology_shape():
    builder = FakeBuilder()

    def connect(source, target, source_port: int = 0, target_port: int = 0) -> None:
        builder.connections.append((source.Name, target.Name, source_port, target_port))

    stage = build_gasifier_stage(builder, "RCT_Conversion", connect)

    assert stage["reactor"].Name == "Downdraft_Gasifier"
    assert "Gasifier_Inlet_Mixer" in builder.operations
    assert "Gasifier_Cooling_Jacket" in builder.operations
    assert "Gasifier_Biomass_Feed" in builder.materials
    assert "Syngas_Pre_PEM" in builder.materials
    assert ("Downdraft_Gasifier", "Syngas_Pre_PEM", 0, 0) in builder.connections


def test_pem_stage_reuses_upstream_syngas_stream():
    builder = FakeBuilder()
    upstream = builder.add_object("MaterialStream", "Syngas_Pre_PEM")

    def connect(source, target, source_port: int = 0, target_port: int = 0) -> None:
        builder.connections.append((source.Name, target.Name, source_port, target_port))

    build_pem_stage(builder, "RCT_Equilibrium", connect, syngas_inlet=upstream)

    assert "PEM_Inlet_Mixer" in builder.operations
    assert "PEM_Syngas_Inlet" not in builder.materials
    assert ("Syngas_Pre_PEM", "PEM_Inlet_Mixer", 0, 0) in builder.connections


def test_trc_stage_creates_standalone_syngas_inlet_when_needed():
    builder = FakeBuilder()

    def connect(source, target, source_port: int = 0, target_port: int = 0) -> None:
        builder.connections.append((source.Name, target.Name, source_port, target_port))

    build_trc_stage(builder, "RCT_PFR", connect)

    assert "TRC_Syngas_Inlet" in builder.materials
    assert "TRC_Reactor" in builder.operations
    assert ("TRC_Reactor", "Syngas_Pre_Quench", 0, 0) in builder.connections

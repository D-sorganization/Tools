"""Fixture gates for the body-chain wire and engine parsers (H2, #4564)."""

from __future__ import annotations

import json

import pytest

from shared.python.contracts import PreconditionError
from shared.python.swing_sim.model_interchange import (
    BODY_CHAIN_FORMAT,
    BodyChain,
    ChainBody,
    ChainJoint,
    body_chain_from_json,
    body_chain_to_json,
    chain_from_mjcf,
    chain_from_osim,
    chain_from_urdf,
    grip_boundary_reduction,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]

_MJCF = """
<mujoco model="golfer-arm">
  <worldbody>
    <body name="torso">
      <inertial mass="40.0" diaginertia="1.2 1.1 0.4"/>
      <body name="forearm">
        <inertial mass="1.5" diaginertia="0.01 0.01 0.002"/>
        <joint name="elbow" type="hinge" stiffness="30.0" damping="2.0"/>
        <body name="hand">
          <inertial mass="0.5" diaginertia="0.001 0.001 0.0005"/>
          <joint name="wrist" type="hinge" stiffness="50000.0" damping="50.0"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_URDF = """
<robot name="golfer-arm">
  <link name="torso">
    <inertial>
      <mass value="40.0"/>
      <inertia ixx="1.2" iyy="1.1" izz="0.4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <link name="hand">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="wrist" type="revolute">
    <parent link="torso"/>
    <child link="hand"/>
    <dynamics damping="50.0"/>
  </joint>
</robot>
"""

_OSIM = """
<OpenSimDocument Version="40000">
  <Model name="golfer-arm">
    <BodySet>
      <objects>
        <Body name="torso">
          <mass>40.0</mass>
          <inertia>1.2 1.1 0.4 0 0 0</inertia>
        </Body>
        <Body name="hand">
          <mass>0.5</mass>
          <inertia>0.001 0.001 0.0005 0 0 0</inertia>
        </Body>
      </objects>
    </BodySet>
  </Model>
</OpenSimDocument>
"""


class TestMjcf:
    def test_parses_the_nested_tree_with_native_stiffness(self) -> None:
        chain = chain_from_mjcf(_MJCF)
        assert chain.source_id == "mjcf:golfer-arm"
        hand = chain.body("hand")
        assert hand.parent == "forearm"
        assert hand.mass_kg == pytest.approx(0.5)
        assert hand.joint is not None
        assert hand.joint.stiffness == pytest.approx(5.0e4)
        assert hand.joint.damping == pytest.approx(50.0)

    def test_missing_inertial_is_refused(self) -> None:
        broken = _MJCF.replace(
            '<inertial mass="0.5" diaginertia="0.001 0.001 0.0005"/>', ""
        )
        with pytest.raises(ValueError, match="inertial"):
            chain_from_mjcf(broken)


class TestUrdf:
    def test_parses_links_joints_and_zero_stiffness(self) -> None:
        """URDF carries no joint stiffness — it must parse as 0, and the
        explicit override in the reduction is the sanctioned supply path."""
        chain = chain_from_urdf(_URDF)
        assert chain.source_id == "urdf:golfer-arm"
        hand = chain.body("hand")
        assert hand.parent == "torso"
        assert hand.joint is not None
        assert hand.joint.stiffness == 0.0
        assert hand.joint.damping == pytest.approx(50.0)

    def test_malformed_xml_is_refused(self) -> None:
        with pytest.raises(ValueError, match="well-formed"):
            chain_from_urdf("<robot name='x'>")


class TestOsim:
    def test_parses_bodyset_masses_and_diagonal_inertia(self) -> None:
        chain = chain_from_osim(_OSIM)
        assert chain.source_id == "osim:golfer-arm"
        hand = chain.body("hand")
        assert hand.mass_kg == pytest.approx(0.5)
        assert hand.inertia_diag_kg_m2 == pytest.approx((0.001, 0.001, 0.0005))
        assert hand.joint is not None and hand.joint.stiffness == 0.0


class TestWire:
    def test_round_trip_is_deterministic_and_lossless(self) -> None:
        chain = chain_from_mjcf(_MJCF)
        first = body_chain_to_json(chain)
        assert body_chain_to_json(chain) == first
        restored = body_chain_from_json(first)
        assert body_chain_to_json(restored) == first
        assert json.loads(first)["format"] == BODY_CHAIN_FORMAT

    def test_unknown_fields_and_dangling_parents_are_refused(self) -> None:
        payload = json.loads(body_chain_to_json(chain_from_mjcf(_MJCF)))
        payload["extra"] = 1
        with pytest.raises(PreconditionError):
            body_chain_from_json(json.dumps(payload))
        with pytest.raises(PreconditionError, match="not in the chain"):
            BodyChain(
                source_id="bad",
                bodies=(
                    ChainBody(
                        name="orphan",
                        mass_kg=1.0,
                        inertia_diag_kg_m2=(0.1, 0.1, 0.1),
                        parent="ghost",
                        joint=ChainJoint(name="j", type="fixed"),
                    ),
                ),
            )


class TestGripBoundaryReduction:
    def test_reduces_named_selection_with_native_stiffness(self) -> None:
        chain = chain_from_mjcf(_MJCF)
        boundary = grip_boundary_reduction(
            chain,
            hand_bodies=("hand", "forearm"),
            boundary_joint_of="hand",
        )
        assert boundary["effective_mass_kg"] == pytest.approx(2.0)
        assert boundary["stiffness_n_m"] == pytest.approx(5.0e4)
        assert boundary["damping_n_s_m"] == pytest.approx(50.0)
        assert "mjcf:golfer-arm" in str(boundary["provenance"])

    def test_urdf_reduction_requires_the_explicit_override(self) -> None:
        chain = chain_from_urdf(_URDF)
        boundary = grip_boundary_reduction(
            chain,
            hand_bodies=("hand",),
            boundary_joint_of="hand",
            stiffness_override_n_m=5.0e4,
        )
        assert boundary["stiffness_n_m"] == pytest.approx(5.0e4)
        assert "overridden" in str(boundary["provenance"])

    def test_feeds_the_coupled_impact_model_end_to_end(self) -> None:
        """The epic's promise: an engine model file drives the coupled
        impact analysis through GripBoundary(**reduction)."""
        from shared.python.golf_club.impact_coupling import (
            CoupledImpactConfig,
            GripBoundary,
            simulate_coupled_impact,
        )

        chain = chain_from_mjcf(_MJCF)
        boundary = grip_boundary_reduction(
            chain, hand_bodies=("hand", "forearm"), boundary_joint_of="hand"
        )
        result = simulate_coupled_impact(
            CoupledImpactConfig(
                head_mass_kg=0.2005,
                head_speed_mps=44.0,
                shaft_stiffness_n_m=200.0,
                grip=GripBoundary(**boundary),  # type: ignore[arg-type]
            )
        )
        assert result.decoupling_fraction > 0.99
        assert "mjcf:golfer-arm" in result.grip_provenance

    def test_unknown_bodies_and_missing_joints_are_refused(self) -> None:
        chain = chain_from_mjcf(_MJCF)
        with pytest.raises(PreconditionError, match="unknown body"):
            grip_boundary_reduction(
                chain, hand_bodies=("ghost",), boundary_joint_of="hand"
            )
        with pytest.raises(PreconditionError, match="no parent joint"):
            grip_boundary_reduction(
                chain, hand_bodies=("hand",), boundary_joint_of="torso"
            )

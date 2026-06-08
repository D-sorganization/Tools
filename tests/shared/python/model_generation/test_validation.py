"""Tests for the centralized model Validator.

These assert the validation *contract* downstream model generation relies
on: mass/inertia physical checks, joint-axis and limit checks, and
hierarchy checks (duplicate names, missing roots, circular dependencies).
"""

from __future__ import annotations

import pytest
from model_generation.core.types import (
    Inertia,
    Joint,
    JointLimits,
    JointType,
    Link,
)
from model_generation.core.validation import (
    ValidationError,
    ValidationResult,
    ValidationWarning,
    Validator,
)


def _good_inertia() -> Inertia:
    return Inertia(ixx=0.1, iyy=0.1, izz=0.1, mass=1.0)


def _link(name: str, inertia: Inertia | None = None) -> Link:
    return Link(name=name, inertia=inertia or _good_inertia())


def _joint(
    name: str,
    parent: str,
    child: str,
    jtype: JointType = JointType.FIXED,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    limits: JointLimits | None = None,
) -> Joint:
    return Joint(
        name=name,
        joint_type=jtype,
        parent=parent,
        child=child,
        axis=axis,
        limits=limits,
    )


class TestValidationResult:
    def test_truthiness_follows_is_valid(self) -> None:
        assert bool(ValidationResult(is_valid=True)) is True
        assert bool(ValidationResult(is_valid=False)) is False

    def test_add_error_flips_validity(self) -> None:
        result = ValidationResult(is_valid=True)
        result.add_error("CODE", "boom", component="link0")
        assert result.is_valid is False
        assert result.get_error_messages() == ["[link0] CODE: boom"]

    def test_add_error_requires_code(self) -> None:
        result = ValidationResult(is_valid=True)
        with pytest.raises(ValueError):
            result.add_error(None, "msg")  # type: ignore[arg-type]

    def test_add_warning_keeps_validity(self) -> None:
        result = ValidationResult(is_valid=True)
        result.add_warning("W", "careful")
        assert result.is_valid is True
        assert result.get_warning_messages() == ["Warning: careful"]

    def test_merge_combines_and_invalidates(self) -> None:
        a = ValidationResult(is_valid=True)
        b = ValidationResult(is_valid=True)
        b.add_error("X", "bad")
        a.merge(b)
        assert a.is_valid is False
        assert len(a.errors) == 1

    def test_merge_none_raises(self) -> None:
        with pytest.raises(ValueError):
            ValidationResult(is_valid=True).merge(None)  # type: ignore[arg-type]

    def test_error_str_without_component(self) -> None:
        assert str(ValidationError("C", "m")) == "C: m"

    def test_warning_str_without_component(self) -> None:
        assert str(ValidationWarning("C", "m")) == "Warning: m"


class TestValidateMass:
    def test_positive_mass_is_valid(self) -> None:
        assert Validator.validate_mass(1.0).is_valid

    def test_zero_mass_errors(self) -> None:
        result = Validator.validate_mass(0.0, component="link")
        assert not result.is_valid
        assert any("MASS_001" in m for m in result.get_error_messages())

    def test_negative_mass_errors(self) -> None:
        assert not Validator.validate_mass(-5.0).is_valid

    def test_tiny_mass_warns_but_valid(self) -> None:
        result = Validator.validate_mass(1e-9)
        assert result.is_valid
        assert result.warnings

    def test_none_mass_raises(self) -> None:
        with pytest.raises(ValueError):
            Validator.validate_mass(None)  # type: ignore[arg-type]


class TestValidateInertia:
    def test_good_inertia_valid(self) -> None:
        assert Validator.validate_inertia(_good_inertia()).is_valid

    def test_negative_diagonal_errors(self) -> None:
        bad = Inertia(ixx=-0.1, iyy=0.1, izz=0.1, mass=1.0)
        result = Validator.validate_inertia(bad)
        assert not result.is_valid
        assert any("INERTIA_002" in m for m in result.get_error_messages())

    def test_triangle_inequality_strict_is_error(self) -> None:
        # izz far exceeds ixx + iyy -> violates triangle inequality.
        bad = Inertia(ixx=0.1, iyy=0.1, izz=10.0, mass=1.0)
        strict = Validator.validate_inertia(bad, strict=True)
        lax = Validator.validate_inertia(bad, strict=False)
        assert not strict.is_valid
        # In lax mode the same condition becomes a warning, not an error.
        assert lax.is_valid or any(
            "INERTIA_003" in m for m in lax.get_warning_messages()
        )

    def test_small_inertia_warns(self) -> None:
        # Below MIN_INERTIA_KG_M2 (1e-12) but still positive-definite.
        small = Inertia(ixx=1e-13, iyy=1e-13, izz=1e-13, mass=1.0)
        result = Validator.validate_inertia(small)
        assert any(w.code == "INERTIA_SMALL" for w in result.warnings)


class TestValidateLink:
    def test_valid_link(self) -> None:
        assert Validator.validate_link(_link("base")).is_valid

    def test_empty_name_errors(self) -> None:
        result = Validator.validate_link(_link("   "))
        assert not result.is_valid
        assert any("LINK_NAME_EMPTY" in m for m in result.get_error_messages())


class TestValidateJoint:
    def test_valid_revolute_joint(self) -> None:
        j = _joint("j", "a", "b", JointType.REVOLUTE, axis=(0.0, 0.0, 1.0))
        assert Validator.validate_joint(j, {"a", "b"}).is_valid

    def test_missing_parent_errors(self) -> None:
        j = _joint("j", "ghost", "b")
        result = Validator.validate_joint(j, {"b"})
        assert any("JOINT_003" in m for m in result.get_error_messages())

    def test_missing_child_errors(self) -> None:
        j = _joint("j", "a", "ghost")
        result = Validator.validate_joint(j, {"a"})
        assert any("JOINT_004" in m for m in result.get_error_messages())

    def test_zero_axis_errors(self) -> None:
        j = _joint("j", "a", "b", JointType.REVOLUTE, axis=(0.0, 0.0, 0.0))
        result = Validator.validate_joint(j, {"a", "b"})
        assert any("JOINT_001" in m for m in result.get_error_messages())

    def test_unnormalized_axis_warns(self) -> None:
        j = _joint("j", "a", "b", JointType.PRISMATIC, axis=(0.0, 0.0, 2.0))
        result = Validator.validate_joint(j, {"a", "b"})
        # Non-unit (but non-zero) axis is a warning, not a hard error.
        assert result.is_valid
        assert any(w.code == "JOINT_001" for w in result.warnings)

    def test_inverted_limits_error(self) -> None:
        limits = JointLimits(lower=1.0, upper=-1.0)
        j = _joint("j", "a", "b", JointType.REVOLUTE, axis=(0, 0, 1), limits=limits)
        result = Validator.validate_joint(j, {"a", "b"})
        assert any("JOINT_002" in m for m in result.get_error_messages())

    def test_fixed_joint_skips_axis_check(self) -> None:
        # A fixed joint with a degenerate axis is still valid.
        j = _joint("j", "a", "b", JointType.FIXED, axis=(0.0, 0.0, 0.0))
        assert Validator.validate_joint(j, {"a", "b"}).is_valid


class TestValidateHierarchy:
    def test_simple_chain_valid(self) -> None:
        links = [_link("a"), _link("b")]
        joints = [_joint("j", "a", "b")]
        assert Validator.validate_hierarchy(links, joints).is_valid

    def test_duplicate_link_names_error(self) -> None:
        links = [_link("a"), _link("a")]
        result = Validator.validate_hierarchy(links, [])
        assert any("HIERARCHY_003" in m for m in result.get_error_messages())

    def test_no_root_is_circular_error(self) -> None:
        # Every link is a child -> no root.
        links = [_link("a"), _link("b")]
        joints = [_joint("j1", "a", "b"), _joint("j2", "b", "a")]
        result = Validator.validate_hierarchy(links, joints)
        assert any("HIERARCHY_001" in m for m in result.get_error_messages())

    def test_multiple_roots_warns(self) -> None:
        links = [_link("a"), _link("b"), _link("c")]
        joints = [_joint("j", "a", "b")]  # c is a second root
        result = Validator.validate_hierarchy(links, joints)
        assert any(w.code == "HIERARCHY_MULTIPLE_ROOTS" for w in result.warnings)

    def test_circular_dependency_detected(self) -> None:
        links = [_link("a"), _link("b"), _link("c")]
        joints = [
            _joint("j1", "a", "b"),
            _joint("j2", "b", "c"),
            _joint("j3", "c", "b"),
        ]
        result = Validator.validate_hierarchy(links, joints)
        assert not result.is_valid


class TestValidateModel:
    def test_valid_two_link_model(self) -> None:
        links = [_link("base"), _link("tip")]
        joints = [_joint("j", "base", "tip", JointType.REVOLUTE, axis=(0.0, 0.0, 1.0))]
        assert Validator.validate_model(links, joints).is_valid

    def test_model_aggregates_link_and_joint_errors(self) -> None:
        # Bad inertia on a link AND a joint referencing a missing link.
        bad_link = _link("base", Inertia(ixx=-1, iyy=0.1, izz=0.1, mass=1.0))
        links = [bad_link]
        joints = [_joint("j", "base", "ghost")]
        result = Validator.validate_model(links, joints)
        assert not result.is_valid
        assert len(result.errors) >= 2

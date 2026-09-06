"""The interlock limit domain: percent of span, enforced at the API boundary.

Issue #4032 -- the firmware broker used to clamp every tag into [0, 100]
percent of span, so a high limit typed above that ceiling (an operator
entering ``high_limit = 900`` intending 900 degC on the type-K channel) could
never be exceeded: its trip silently never fired. The firmware clamp is gone;
the register contract now defines every actionable limit as either ``None``
(the firmware's disabled sentinels) or a percent-of-span float in
``[hardware.INTERLOCK_LIMIT_MIN, INTERLOCK_LIMIT_MAX]`` = [0, 100].

These tests pin that boundary: a limit the firmware cannot act on is rejected
loudly (DbC: ValueError at the model, HTTP 422 at ``POST /api/routing``)
instead of being accepted into a configuration that cannot trip.
"""

from __future__ import annotations

import os
import sys

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import hardware  # noqa: E402
import pytest  # noqa: E402
from defaults import default_routing_config  # noqa: E402
from models import InterlockConfig, RoutingConfig  # noqa: E402

_LIMIT_FIELDS = ("lolo_limit", "low_limit", "high_limit", "hihi_limit")

pytest.importorskip("httpx")
pytest.importorskip("fastapi.testclient")

os.environ["PLC_DRIVER"] = "modbus"

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app, headers={"X-Requested-With": "p1am-hmi"})


class TestLimitDomainContract:
    """Every actionable limit is a percent of span in [0, 100] (#4032)."""

    @pytest.mark.parametrize("field", _LIMIT_FIELDS)
    @pytest.mark.parametrize("value", [0.0, 1.8, 95.0, 100.0])
    def test_in_domain_limits_are_accepted(self, field: str, value: float) -> None:
        config = InterlockConfig(**{field: value})
        assert getattr(config, field) == value

    @pytest.mark.parametrize("field", _LIMIT_FIELDS)
    @pytest.mark.parametrize("bad", [900.0, 101.0, 1400.0, -0.5, -2.5])
    def test_out_of_domain_limit_is_rejected(self, field: str, bad: float) -> None:
        """DbC: a limit the firmware cannot act on must fail loudly.

        900 is the issue's operator scenario: a degC limit typed into the
        percent-domain field. Under the old clamp it was accepted, encoded to
        the PLC, displayed as configured, and never tripped.
        """
        with pytest.raises(ValueError, match="percent"):
            InterlockConfig(**{field: bad})

    @pytest.mark.parametrize("field", _LIMIT_FIELDS)
    def test_domain_bounds_are_inclusive(self, field: str) -> None:
        """0 % and 100 % (the shipped default hihi) are actionable limits."""
        low = InterlockConfig(**{field: hardware.INTERLOCK_LIMIT_MIN})
        high = InterlockConfig(**{field: hardware.INTERLOCK_LIMIT_MAX})
        assert getattr(low, field) == hardware.INTERLOCK_LIMIT_MIN
        assert getattr(high, field) == hardware.INTERLOCK_LIMIT_MAX

    @pytest.mark.parametrize("field", _LIMIT_FIELDS)
    def test_none_still_disables_the_side(self, field: str) -> None:
        config = InterlockConfig(**{field: None})
        assert getattr(config, field) is None

    @pytest.mark.parametrize("field", _LIMIT_FIELDS)
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_still_rejected(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError):
            InterlockConfig(**{field: bad})

    def test_domain_constants_match_the_documented_contract(self) -> None:
        """The [0, 100] percent-of-span domain is the register contract."""
        assert hardware.INTERLOCK_LIMIT_MIN == 0.0
        assert hardware.INTERLOCK_LIMIT_MAX == 100.0

    def test_default_config_is_inside_the_domain(self) -> None:
        """The shipped defaults must survive the boundary they deploy through."""
        config = default_routing_config()
        for interlock in config.interlocks.values():
            for field in _LIMIT_FIELDS:
                value = getattr(interlock, field)
                if value is not None:
                    assert hardware.INTERLOCK_LIMIT_MIN <= value
                    assert value <= hardware.INTERLOCK_LIMIT_MAX


class TestRoutingBoundary:
    """The operator-facing /api/routing boundary rejects nonsense limits."""

    def _payload(self) -> dict:
        return default_routing_config().model_dump()

    def test_post_routing_rejects_engineering_unit_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The #4032 operator scenario end to end: 900 must be a 422.

        An operator reading a field labelled "high limit" on a thermocouple
        channel naturally types 900 meaning 900 degC. The boundary must refuse
        it with the expected unit named, not encode it into a trip that can
        never fire.
        """
        monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
        payload = self._payload()
        payload["interlocks"]["TAG_0"]["high_limit"] = 900.0
        response = client.post("/api/routing", json=payload)
        assert response.status_code == 422
        assert "percent" in response.text

    def test_post_routing_accepts_in_domain_percent_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The degC-equivalent of the same intent is valid: 64.29 % = 900 degC."""
        monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
        payload = self._payload()
        payload["interlocks"]["TAG_0"]["high_limit"] = hardware.celsius_to_percent(
            900.0
        )
        config = RoutingConfig.model_validate(payload)
        assert config.interlocks["TAG_0"].high_limit == pytest.approx(64.2857)


class TestRegisterImageDecoding:
    """A limit the firmware cannot act on decodes as disabled, not live."""

    def test_legacy_out_of_domain_register_reads_as_disabled(self) -> None:
        """A pre-contract register image (hihi 900) is not an actionable limit.

        Decoding it as a live number would hand the alarm engine a threshold
        no reading can ever cross; ``None`` matches what the firmware can do
        with it. The API boundary prevents writing one in the first place.
        """
        from modbus_codec import (
            INTERLOCK_REGISTER_WIDTH,
            decode_interlocks,
            encode_interlocks,
            float_to_registers,
        )

        registers = [0] * (hardware.TAG_COUNT * INTERLOCK_REGISTER_WIDTH)
        base = 0 * INTERLOCK_REGISTER_WIDTH
        registers[base + 4 : base + 6] = float_to_registers(900.0)  # high slot
        decoded = decode_interlocks(registers)
        assert decoded["TAG_0"].high_limit is None
        # The in-domain encode/decode path is untouched.
        config = {"TAG_1": InterlockConfig(high_limit=95.0, hihi_limit=100.0)}
        re_decoded = decode_interlocks(encode_interlocks(config))
        assert re_decoded["TAG_1"].high_limit == pytest.approx(95.0)
        assert re_decoded["TAG_1"].hihi_limit == pytest.approx(100.0)

    def test_disabled_sentinels_still_decode_to_none(self) -> None:
        from modbus_codec import decode_high_limit, encode_high_limit

        low, high = encode_high_limit(None)
        assert decode_high_limit(low, high) is None

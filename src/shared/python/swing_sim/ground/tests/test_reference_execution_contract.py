"""Golden-byte and import-order contracts for reference ground execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import shared.python.swing_sim.ground.reference_execution as reference_execution
from shared.python.swing_sim.ground import (
    GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION,
    BounceModelSettings,
    GroundReferenceExecution,
    GroundSimulationRequest,
    PlanarSurfaceDomain,
    SkidRollSettings,
    SurfaceResolver,
    run_ground_reference,
)

from ._support import _surface_run_request

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "ground_reference_pipeline_golden_v1.json"
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_reference_golden_fixture_pins_request_result_and_digests() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = GroundSimulationRequest.from_dict(fixture["request"])
    controls = fixture["execution"]
    execution = GroundReferenceExecution(
        bounce_settings=BounceModelSettings(**controls["bounce_settings"]),
        skid_roll_settings=SkidRollSettings(**controls["skid_roll_settings"]),
    )

    result = run_ground_reference(request, execution)

    assert fixture["fixture_schema_version"] == ("ground-reference-execution-golden/v1")
    assert controls["is_cancelled"] is None
    assert controls["resolver"] is None
    assert fixture["execution_schema_version"] == (
        GROUND_REFERENCE_EXECUTION_SCHEMA_VERSION
    )
    assert result.to_dict() == fixture["result"]
    assert _digest(request.to_json()) == fixture["request_sha256"]
    assert _digest(result.to_json()) == fixture["result_sha256"]
    expected_fixture_sha256 = "a5d96d4686c589c1d5c6daac1feebaac0d7fdb69ac5020166f04e27308dd5345"  # pragma: allowlist secret  # noqa: E501
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == expected_fixture_sha256


def test_reference_executor_rejects_nonexact_request() -> None:
    with pytest.raises(ValueError, match="exact GroundSimulationRequest"):
        run_ground_reference(object())  # type: ignore[arg-type]


def test_resolver_mismatch_fails_before_bounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _surface_run_request()
    foreign = replace(request.surface, surface_id="different-surface")
    resolver = SurfaceResolver(PlanarSurfaceDomain(foreign))

    def forbidden_bounce(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("resolver mismatch must fail before physics")

    monkeypatch.setattr(
        reference_execution,
        "simulate_repeated_bounce",
        forbidden_bounce,
    )

    with pytest.raises(ValueError, match="provider identity must match"):
        run_ground_reference(
            request,
            GroundReferenceExecution(resolver=resolver),
        )


@pytest.mark.parametrize(
    "imports",
    [
        "import shared.python.swing_sim.flight; "
        "import shared.python.swing_sim.ground as ground; "
        "assert callable(ground.run_ground_reference)",
        "import shared.python.swing_sim.ground as ground; "
        "import shared.python.swing_sim.flight; "
        "assert callable(ground.run_ground_reference)",
    ],
)
def test_reference_exports_preserve_both_import_orders(imports: str) -> None:
    repository = Path(__file__).parents[6]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(repository / "src"),
            str(repository / "src" / "shared" / "python"),
            environment.get("PYTHONPATH", ""),
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", imports],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

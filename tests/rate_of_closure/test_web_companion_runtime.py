"""Real-process lifecycle qualification for the source production companion."""

# These tests each spawn a real authority subprocess (uvicorn) and wait for it
# to report its listener within _PORT_REPORT_TIMEOUT_S (15 s). pytest runs with
# `-n auto --dist loadscope`, so this module gets one worker, but other
# subprocess-spawning modules run on other workers at the same time. On a box
# already saturated -- two full suites at once, for instance -- the children can
# starve and fail with "authority child did not report its listener". That is
# oversubscription, not a defect: the same tests pass in isolation and in CI.
# Do not raise _PORT_REPORT_TIMEOUT_S to chase it; that constant guards real
# production hangs.

from __future__ import annotations

import json
import socket
import time
from http.client import HTTPConnection
from types import MappingProxyType

import pytest

import rate_of_closure.web_companion.runtime as runtime_module
from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_execution_job import (
    regional_ground_execution_job_to_json,
)
from rate_of_closure.web_authority.api import CAPABILITY_PATH
from rate_of_closure.web_companion.bundle import CompanionWebBundle
from rate_of_closure.web_companion.runtime import start_companion
from rate_of_closure.web_distribution.asset_resolver import ResolvedWebAsset
from tests.rate_of_closure.test_regional_ground_authority_jobs import _job

_REVISION = "a" * 40


def _bundle() -> CompanionWebBundle:
    runtime = json.dumps(
        {
            "schema_version": "rate-of-closure/web-runtime/v1",
            "mode": "local_companion",
            "release_revision": _REVISION,
            "authority_path": "/api/rate-of-closure/v1",
        },
        separators=(",", ":"),
    )
    index = (
        '<!doctype html><script id="rate-of-closure-web-runtime" '
        f'type="application/json">{runtime}</script><div id="root"></div>'
    ).encode()
    return CompanionWebBundle(
        _REVISION,
        ResolvedWebAsset(index, "text/html; charset=utf-8"),
        MappingProxyType(
            {
                "assets/index-AbCd_123.js": ResolvedWebAsset(
                    b"export {};", "text/javascript; charset=utf-8"
                )
            }
        ),
    )


def _request(port: int, path: str) -> tuple[int, bytes, dict[str, str]]:
    connection = HTTPConnection("127.0.0.1", port, timeout=10.0)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, response.read(), headers
    finally:
        connection.close()


def _api_request(
    runtime,
    method: str,
    path: str,
    body: bytes | None = None,
) -> tuple[int, dict[str, object]]:
    headers = {}
    if method == "POST":
        headers = {
            "Origin": runtime.url.rstrip("/"),
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
    connection = HTTPConnection("127.0.0.1", runtime.port, timeout=10.0)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_real_companion_is_ready_token_free_and_reaps_authority(tmp_path) -> None:
    runtime = start_companion(bundle=_bundle(), state_root=tmp_path, open_browser=False)
    child_port = runtime.authority.port
    token = runtime.authority.token
    try:
        status, index, headers = _request(runtime.port, "/")
        capability_status, capability, _ = _request(runtime.port, CAPABILITY_PATH)
        assert status == 200
        assert capability_status == 200
        assert json.loads(capability)["regional_ground_execution"] is True
        assert token.encode() not in index
        assert str(child_port).encode() not in index
        assert headers["cache-control"] == "no-store"
        assert runtime.port != child_port
        assert runtime.authority.process.poll() is None
        gateway_port = runtime.port
    finally:
        runtime.close()
    assert runtime.authority.process.poll() is not None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        assert probe.connect_ex(("127.0.0.1", gateway_port)) != 0


def test_companion_rejects_nonrelease_bundle_before_starting_authority() -> None:
    bundle = CompanionWebBundle(
        "development",
        ResolvedWebAsset(b"index", "text/html; charset=utf-8"),
        MappingProxyType(
            {"assets/index.js": ResolvedWebAsset(b"x", "text/javascript")}
        ),
    )
    try:
        start_companion(bundle=bundle, open_browser=False)
    except ValueError as error:
        assert str(error) == "production companion requires an exact release revision"
    else:
        raise AssertionError("development bundle was admitted")


def test_companion_restarts_dead_authority_before_next_explicit_request(
    tmp_path,
) -> None:
    runtime = start_companion(bundle=_bundle(), state_root=tmp_path, open_browser=False)
    first = runtime.authority
    try:
        first.process.kill()
        first.process.wait(timeout=5.0)
        status, capability, _ = _request(runtime.port, CAPABILITY_PATH)
        second = runtime.authority
        assert status == 200
        assert json.loads(capability)["regional_ground_execution"] is True
        assert second.process.poll() is None
        assert second.port != first.port
        assert second.token != first.token
    finally:
        runtime.close()


def test_companion_cleans_partial_startup_and_releases_state_lock(
    tmp_path, monkeypatch
) -> None:
    captured = []
    real_supervisor = runtime_module.AuthoritySupervisor

    def supervisor_factory(**kwargs):
        supervisor = real_supervisor(**kwargs)
        captured.append(supervisor)
        return supervisor

    monkeypatch.setattr(runtime_module, "AuthoritySupervisor", supervisor_factory)
    monkeypatch.setattr(
        runtime_module,
        "_listener",
        lambda: (_ for _ in ()).throw(OSError("injected listener failure")),
    )
    with pytest.raises(OSError, match="injected listener failure"):
        start_companion(bundle=_bundle(), state_root=tmp_path, open_browser=False)
    assert len(captured) == 1
    assert captured[0].authority.process.poll() is not None
    replacement = real_supervisor(
        source_root=runtime_module._source_root(),
        state_root=tmp_path,
        timeout_s=1.0,
    )
    replacement.close()


def test_companion_closes_listener_when_app_construction_fails(
    tmp_path, monkeypatch
) -> None:
    listeners = []
    supervisors = []
    real_listener = runtime_module._listener
    real_supervisor = runtime_module.AuthoritySupervisor

    def tracked_listener():
        listener = real_listener()
        listeners.append(listener)
        return listener

    def supervisor_factory(**kwargs):
        supervisor = real_supervisor(**kwargs)
        supervisors.append(supervisor)
        return supervisor

    monkeypatch.setattr(runtime_module, "_listener", tracked_listener)
    monkeypatch.setattr(runtime_module, "AuthoritySupervisor", supervisor_factory)
    monkeypatch.setattr(
        runtime_module,
        "create_companion_app",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected app failure")),
    )
    with pytest.raises(RuntimeError, match="injected app failure"):
        start_companion(bundle=_bundle(), state_root=tmp_path, open_browser=False)
    assert listeners[0].fileno() == -1
    assert supervisors[0].authority.process.poll() is not None


def test_completed_result_survives_hard_loss_through_companion_gateway(
    tmp_path,
) -> None:
    factory = (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_durable_test_authority_app"
    )
    runtime = start_companion(
        bundle=_bundle(),
        state_root=tmp_path,
        open_browser=False,
        authority_app_factory=factory,
    )
    job = _job()
    path = f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}"
    try:
        submitted, _ = _api_request(
            runtime,
            "POST",
            "/api/rate-of-closure/v1/regional-ground/jobs",
            regional_ground_execution_job_to_json(job).encode(),
        )
        assert submitted == 202
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, status = _api_request(runtime, "GET", path)
            if status["status"] == AuthorityJobStatus.SUCCEEDED:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("companion job did not complete")
        first = runtime.authority
        first.process.kill()
        first.process.wait(timeout=5.0)
        status_code, recovered = _api_request(runtime, "GET", path)
        result_code, result = _api_request(runtime, "GET", f"{path}/result")
        assert status_code == 200
        assert recovered["status"] == AuthorityJobStatus.SUCCEEDED
        assert result_code == 200
        assert result["job_id"] == job.job_id
        assert runtime.authority.process.poll() is None
        assert runtime.authority.port != first.port
    finally:
        runtime.close()


def test_companion_gateway_cancels_running_job(tmp_path) -> None:
    factory = (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_cancellable_authority_app"
    )
    runtime = start_companion(
        bundle=_bundle(),
        state_root=tmp_path,
        open_browser=False,
        authority_app_factory=factory,
    )
    job = _job()
    path = f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}"
    try:
        submitted, _ = _api_request(
            runtime,
            "POST",
            "/api/rate-of-closure/v1/regional-ground/jobs",
            regional_ground_execution_job_to_json(job).encode(),
        )
        assert submitted == 202
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, status = _api_request(runtime, "GET", path)
            if status["status"] == AuthorityJobStatus.RUNNING:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("companion job did not start")
        cancelled, _ = _api_request(runtime, "POST", f"{path}/cancel")
        assert cancelled == 202
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, status = _api_request(runtime, "GET", path)
            if status["status"] == AuthorityJobStatus.CANCELLED:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("companion job did not cancel")
    finally:
        runtime.close()


def test_running_job_hard_loss_is_failed_without_gateway_replay(tmp_path) -> None:
    factory = (
        "tests.rate_of_closure.test_regional_ground_real_loopback:"
        "create_durable_blocking_authority_app"
    )
    runtime = start_companion(
        bundle=_bundle(),
        state_root=tmp_path,
        open_browser=False,
        authority_app_factory=factory,
    )
    job = _job()
    path = f"/api/rate-of-closure/v1/regional-ground/jobs/{job.job_id}"
    try:
        submitted, _ = _api_request(
            runtime,
            "POST",
            "/api/rate-of-closure/v1/regional-ground/jobs",
            regional_ground_execution_job_to_json(job).encode(),
        )
        assert submitted == 202
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            _, status = _api_request(runtime, "GET", path)
            if status["status"] == AuthorityJobStatus.RUNNING:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("companion job did not reach running")
        first = runtime.authority
        first.process.kill()
        first.process.wait(timeout=5.0)
        recovered_code, recovered = _api_request(runtime, "GET", path)
        assert recovered_code == 200
        assert recovered["status"] == AuthorityJobStatus.FAILED
        assert recovered["failure"] == {
            "code": "execution_failed",
            "stage": "authority_restart",
        }
    finally:
        runtime.close()

"""Transport-boundary tests for the UI-neutral Morris authority client."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rate_of_closure.application.morris.client import (
    MorrisAuthorityClient,
    MorrisAuthorityHttpError,
    _bounded_body,
    _public_error,
    _strict_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_lazy_packages_preserve_existing_public_exports() -> None:
    from rate_of_closure.application import APP_COMMAND_IDS, WORKSPACE_SCHEMA
    from rate_of_closure.application.morris import MORRIS_REQUEST_SCHEMA_ID

    assert APP_COMMAND_IDS
    assert WORKSPACE_SCHEMA
    assert MORRIS_REQUEST_SCHEMA_ID == "rate-of-closure/morris-request"


def test_ui_contract_submodules_import_without_optional_servers_or_scipy() -> None:
    source = textwrap.dedent(
        """
        import builtins
        real_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in {'scipy', 'fastapi', 'uvicorn'}:
                raise ImportError(f'blocked optional dependency: {name}')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        for name in (
            'rate_of_closure.application.morris.request_document',
            'rate_of_closure.application.morris.client',
            'rate_of_closure.application.morris.presentation',
            'rate_of_closure.application.morris.response_contract',
        ):
            __import__(name)
        """
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited_path
        else os.pathsep.join((source_root, inherited_path))
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_durable_ui_contract_imports_without_optional_server() -> None:
    source = textwrap.dedent(
        """
        import builtins
        real_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in {'fastapi', 'uvicorn'}:
                raise ImportError(f'blocked optional dependency: {name}')
            return real_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        __import__('rate_of_closure.application.durable_ensemble.contracts')
        """
    )
    environment = os.environ.copy()
    source_root = str(Path(__file__).parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr


class _Handler(BaseHTTPRequestHandler):
    received_authorization = ""

    def do_GET(self) -> None:  # noqa: N802
        type(self).received_authorization = self.headers.get("Authorization", "")
        if self.path != "/api/rate-of-closure/v1/morris/capabilities":
            body = (
                b'{"error":"Bearer private-token reflected"}'
                if self.path.endswith("/reflected")
                else b'{"error":"unknown Morris job"}'
            )
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps(
            {
                "schema_id": "rate-of-closure/morris-authority-capability",
                "schema_version": 1,
                "available": True,
                "api_prefix": "/api/rate-of-closure/v1",
                "request_schema_id": "rate-of-closure/morris-request",
                "job_schema_id": "rate-of-closure/morris-job",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def test_python_client_is_authenticated_direct_loopback_and_sanitizes_errors() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client = MorrisAuthorityClient(
            f"http://127.0.0.1:{server.server_port}",
            {"Authorization": "Bearer private-token"},
        )
        assert client.capability().available
        assert _Handler.received_authorization == "Bearer private-token"
        assert "private-token" not in repr(client)
        with pytest.raises(ValueError, match="numeric IPv4 loopback"):
            MorrisAuthorityClient(
                "http://localhost:1234", {"Authorization": "Bearer private-token"}
            )
        with pytest.raises(ValueError, match="numeric IPv4 loopback"):
            MorrisAuthorityClient(
                f"http://user@127.0.0.1:{server.server_port}",
                {"Authorization": "Bearer private-token"},
            )
        with pytest.raises(ValueError, match="portable"):
            client.status("míssing")
        with pytest.raises(MorrisAuthorityHttpError) as caught:
            client.status("missing")
        assert "private-token" not in str(caught.value)
        with pytest.raises(MorrisAuthorityHttpError) as reflected:
            client.status("reflected")
        assert "private-token" not in str(reflected.value)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class _FakeResponse:
    status = 200

    def __init__(self, length: str | None, body: bytes) -> None:
        self._length = length
        self._body = body

    def getheader(self, name: str) -> str | None:
        return self._length if name == "Content-Length" else None

    def read(self, amount: int) -> bytes:
        return self._body[:amount]


def test_python_client_bounded_reader_and_strict_json_fail_closed() -> None:
    with pytest.raises(MorrisAuthorityHttpError, match="exceeds"):
        _bounded_body(_FakeResponse("17", b""), 16)  # type: ignore[arg-type]
    with pytest.raises(MorrisAuthorityHttpError, match="exceeds"):
        _bounded_body(_FakeResponse(None, b"x" * 17), 16)  # type: ignore[arg-type]
    for body in (b"\xff", b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises((UnicodeDecodeError, ValueError)):
            _strict_json(body)


def test_python_client_rejects_unsafe_public_error_text() -> None:
    generic = "authority rejected the request"
    for message in ("", " ", " padded", "line\r\nbreak", "nul\0byte", "c1\x85"):
        assert _public_error({"error": message}) == generic
    assert _public_error({"error": "bounded public error"}) == "bounded public error"


def test_python_client_sanitizes_malformed_success_without_masking_request_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MorrisAuthorityClient(
        "http://127.0.0.1:34001", {"Authorization": "Bearer private-token"}
    )
    monkeypatch.setattr(MorrisAuthorityClient, "_request", lambda *_args: {})
    with pytest.raises(MorrisAuthorityHttpError, match="success response failed"):
        client.capability()
    with pytest.raises(ValueError):
        client.create({})

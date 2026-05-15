"""Tests for calculator exception handling and error recovery."""

from collections.abc import Generator
from unittest.mock import patch

import pytest
from flask.testing import FlaskClient

from web_applications.calculator.webapp import create_app


@pytest.fixture()
def client() -> Generator[FlaskClient, None, None]:
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_exception_info_leak(client: FlaskClient) -> None:
    """
    Test that internal exceptions do not leak sensitive info.
    We patch the internal dispatch to raise a specific exception
    and verify if that exception's message appears in the response.
    """
    secret_message = "CRITICAL_DATABASE_PASSWORD_LEAK"

    # Patch _dispatch_calculation to raise an exception with a sensitive message
    with patch(
        "web_applications.calculator.webapp._dispatch_calculation",
        side_effect=Exception(secret_message),
    ):
        payload = {"operation": "evaluate", "expression": "1+1"}
        response = client.post("/api/calculate", json=payload)

        assert response.status_code == 500
        json_data = response.get_json()
        assert "error" in json_data

        # Verify fix: The secret message should NOT be in the response
        assert (
            secret_message not in json_data["error"]
        ), "Vulnerability present: Secret message found in response"
        assert json_data["error"] == "An internal error occurred."

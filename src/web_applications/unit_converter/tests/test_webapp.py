"""Tests for the Unit Converter Flask web application."""

from __future__ import annotations

import json

import pytest

from web_applications.unit_converter.webapp import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestIndexRoute:
    def test_index_returns_html(self, client) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Unit Converter" in response.data

    def test_index_has_category_options(self, client) -> None:
        response = client.get("/")
        assert b"length" in response.data.lower()
        assert b"temperature" in response.data.lower()


class TestConvertAPI:
    def test_basic_length_conversion(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "m", "to_unit": "ft"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert abs(data["result"] - 3.28084) < 0.001

    def test_temperature_conversion(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 100, "from_unit": "C", "to_unit": "F"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert abs(data["result"] - 212.0) < 0.01

    def test_mass_conversion(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "kg", "to_unit": "lb"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert abs(data["result"] - 2.20462) < 0.001

    def test_missing_units_returns_400(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "", "to_unit": "ft"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_unknown_unit_returns_400(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "xyz", "to_unit": "ft"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_incompatible_categories_returns_400(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "m", "to_unit": "kg"}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_response_has_formatted_field(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "m", "to_unit": "cm"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert "formatted" in data
        assert data["formatted"] == "100"

    def test_pressure_conversion(self, client) -> None:
        response = client.post(
            "/api/convert",
            data=json.dumps({"value": 1, "from_unit": "atm", "to_unit": "psi"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert abs(data["result"] - 14.696) < 0.01


class TestCategoriesAPI:
    def test_returns_all_categories(self, client) -> None:
        response = client.get("/api/categories")
        assert response.status_code == 200
        data = response.get_json()
        assert "length" in data
        assert "temperature" in data
        assert "pressure" in data

    def test_each_category_has_units(self, client) -> None:
        response = client.get("/api/categories")
        data = response.get_json()
        for cat, info in data.items():
            assert "units" in info
            assert "label" in info
            assert len(info["units"]) > 0, f"Category '{cat}' has no units"


class TestUnitsAPI:
    def test_length_units(self, client) -> None:
        response = client.get("/api/units/length")
        assert response.status_code == 200
        data = response.get_json()
        assert "m" in data["units"]
        assert "ft" in data["units"]

    def test_unknown_category(self, client) -> None:
        response = client.get("/api/units/nonexistent")
        assert response.status_code == 404


class TestSecurityHeaders:
    def test_csp_header(self, client) -> None:
        response = client.get("/")
        assert "Content-Security-Policy" in response.headers

    def test_x_content_type_options(self, client) -> None:
        response = client.get("/")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self, client) -> None:
        response = client.get("/")
        assert response.headers.get("X-Frame-Options") == "DENY"

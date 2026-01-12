import os
import sys
import unittest

# Add the repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from web_applications.calculator.webapp import create_app


class TestWebApp(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_index(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Verify ARIA labels for accessibility
        self.assertIn(b'aria-label="Move cursor left"', response.data)
        self.assertIn(b'aria-label="Matrix exponential"', response.data)
        self.assertIn(b'aria-label="SE3 Hat operator"', response.data)

    def test_calculate_evaluate(self) -> None:
        payload = {"operation": "evaluate", "expression": "2 + 2"}
        response = self.client.post("/api/calculate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["result"], "4")

    def test_calculate_derivative(self) -> None:
        payload = {"operation": "derivative", "expression": "x**2", "variable": "x"}
        response = self.client.post("/api/calculate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["result"], "2*x")

    def test_invalid_operation(self) -> None:
        payload = {"operation": "invalid", "expression": "2 + 2"}
        response = self.client.post("/api/calculate", json=payload)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

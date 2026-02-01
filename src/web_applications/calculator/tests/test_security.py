import os
import sys
import unittest
from pathlib import Path

# Add the repo root to sys.path
sys.path.append(os.path.abspath(Path(Path(__file__).parent, "../..")))


from web_applications.calculator.webapp import create_app


class TestSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_input_too_large(self) -> None:
        # Create a very large expression (> 1000 characters)
        large_expression = "1+" * 1000 + "1"
        payload = {"operation": "evaluate", "expression": large_expression}
        response = self.client.post("/api/calculate", json=payload)

        if response.status_code == 200:
            print("VULNERABILITY CONFIRMED: Large input accepted")

        self.assertEqual(
            response.status_code, 400, "Should reject excessively large input"
        )


if __name__ == "__main__":
    unittest.main()

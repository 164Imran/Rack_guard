import unittest

from fastapi.testclient import TestClient

from backend.main import app


class HealthRouteTests(unittest.TestCase):
    def test_health_reports_modular_backend(self) -> None:
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["modules"]["roi_and_audit"], "ready")
        self.assertTrue(payload["mock_fallback"])


if __name__ == "__main__":
    unittest.main()

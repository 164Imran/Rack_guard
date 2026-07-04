import unittest

from fastapi.testclient import TestClient

from backend.main import app


class RecommendationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_final_recommendation_composes_modules(self) -> None:
        response = self.client.post(
            "/api/recommendations/final",
            json={"gpu_id": "rack-1/gpu-1", "seed": 42},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("thermal_context", payload)
        self.assertIn("roi_result", payload)
        self.assertIn("hidden_cost_audit", payload)
        self.assertIn("llm_report", payload)
        self.assertEqual(payload["module_status"]["minh_thermal"], "used")
        self.assertEqual(payload["module_status"]["roi_and_audit"], "used")
        self.assertEqual(payload["module_status"]["imran"], "pending")
        self.assertEqual(payload["assumptions"]["source"], "demo_defaults")
        self.assertEqual(len(payload["ranked_actions"]), 4)

    def test_unknown_gpu_returns_404(self) -> None:
        response = self.client.post(
            "/api/recommendations/final",
            json={"gpu_id": "rack-99/gpu-99"},
        )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

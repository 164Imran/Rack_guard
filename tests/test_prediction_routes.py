import unittest

from fastapi.testclient import TestClient

from backend.main import app


class PredictionRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_temperature_prediction(self) -> None:
        response = self.client.post(
            "/api/predictions/temperature",
            json={"gpu_id": "rack-1/gpu-1", "seed": 42},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["gpu_id"], "gpu-1")
        self.assertIn(payload["risk"], {"safe", "warning", "critical"})
        self.assertIn(payload["source"], {"imran_node", "minh", "fallback"})
        self.assertGreater(payload["predicted_peak_temp_c"], 0)
        self.assertLessEqual(len(payload["trajectory"]), 22)

    def test_unknown_gpu_returns_404(self) -> None:
        response = self.client.post(
            "/api/predictions/temperature",
            json={"gpu_id": "rack-99/gpu-99"},
        )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()

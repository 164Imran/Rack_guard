import unittest

from fastapi.testclient import TestClient

from backend.main import app


class GpuRouteTests(unittest.TestCase):
    def test_current_gpu_state_is_normalized(self) -> None:
        response = TestClient(app).get(
            "/api/gpus/current?rack_count=1&gpus_per_rack=2&seed=42"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["source"], {"minh_simulator", "mock_fallback"})
        self.assertEqual(len(payload["racks"]), 1)
        self.assertEqual(len(payload["racks"][0]["gpus"]), 2)
        self.assertIn("predicted_temp_c", payload["racks"][0]["gpus"][0])


if __name__ == "__main__":
    unittest.main()

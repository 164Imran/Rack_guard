import unittest

from fastapi.testclient import TestClient

from backend.main import app


class ROIRouteTests(unittest.TestCase):
    def test_roi_endpoint_uses_deterministic_engine(self) -> None:
        response = TestClient(app).post(
            "/api/roi",
            json={
                "action_type": "increase_ventilation",
                "risk_no_action": 0.82,
                "risk_after_action": 0.18,
                "gpu_hour_value_eur": 10,
                "job_remaining_hours": 4,
                "expected_throttle_loss_percent": 0.3,
                "electricity_price_eur_kwh": 0.25,
                "extra_power_kw": 0.15,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["decision"], "recommended")
        self.assertGreater(payload["net_gain_eur"], 0)
        self.assertIsNotNone(payload["roi_percent"])


if __name__ == "__main__":
    unittest.main()

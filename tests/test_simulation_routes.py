import unittest

from fastapi.testclient import TestClient

from backend.main import app


class SimulationRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_all_action_types_are_supported(self) -> None:
        for action_type in (
            "increase_ventilation",
            "reduce_power_cap",
            "migrate_job",
            "no_action",
        ):
            with self.subTest(action_type=action_type):
                response = self.client.post(
                    "/api/simulations/actions",
                    json={
                        "gpu_id": "rack-1/gpu-1",
                        "action_type": action_type,
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["action_type"], action_type)

    def test_no_action_keeps_same_peak(self) -> None:
        response = self.client.post(
            "/api/simulations/actions",
            json={"gpu_id": "rack-1/gpu-1", "action_type": "no_action"},
        )

        payload = response.json()
        self.assertEqual(
            payload["before"]["peak_temp_c"],
            payload["after"]["peak_temp_c"],
        )
        self.assertEqual(payload["cooling_gain_c"], 0)


if __name__ == "__main__":
    unittest.main()

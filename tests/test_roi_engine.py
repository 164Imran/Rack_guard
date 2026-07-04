import unittest

from backend.services.roi_engine import (
    ROIConfig,
    calculate_roi,
    rank_actions_by_net_gain,
)


BASE_INPUT = {
    "risk_no_action": 0.84,
    "risk_after_action": 0.20,
    "gpu_hour_value_eur": 10.0,
    "job_remaining_hours": 4.0,
    "expected_throttle_loss_percent": 0.30,
}


class ROIEngineTests(unittest.TestCase):
    def test_increase_ventilation(self) -> None:
        result = calculate_roi(
            {
                **BASE_INPUT,
                "action_type": "increase_ventilation",
                "extra_power_kw": 1.5,
                "electricity_price_eur_kwh": 0.25,
            }
        )

        self.assertAlmostEqual(result["risk_reduction"], 0.64)
        self.assertAlmostEqual(result["avoided_loss_eur"], 7.68)
        self.assertAlmostEqual(result["action_cost_eur"], 1.5)
        self.assertAlmostEqual(result["net_gain_eur"], 6.18)
        self.assertAlmostEqual(result["roi_percent"] or 0, 412.0)
        self.assertEqual(result["decision"], "recommended")

    def test_reduce_power_cap_partial(self) -> None:
        result = calculate_roi(
            {
                **BASE_INPUT,
                "action_type": "reduce_power_cap",
                "risk_after_action": 0.35,
                "performance_loss_percent": 0.05,
            }
        )

        self.assertAlmostEqual(result["action_cost_eur"], 2.0)
        self.assertGreater(result["net_gain_eur"], 0)
        self.assertEqual(result["decision"], "partial")

    def test_migrate_job_cost(self) -> None:
        result = calculate_roi(
            {
                **BASE_INPUT,
                "action_type": "migrate_job",
                "migration_time_hours": 0.1,
                "transfer_cost_eur": 0.5,
                "interruption_penalty_eur": 0.25,
            }
        )

        self.assertAlmostEqual(result["action_cost_eur"], 1.75)
        self.assertEqual(result["decision"], "recommended")

    def test_no_action_has_null_roi(self) -> None:
        result = calculate_roi(
            {
                **BASE_INPUT,
                "action_type": "no_action",
                "risk_after_action": BASE_INPUT["risk_no_action"],
            }
        )

        self.assertEqual(result["action_cost_eur"], 0)
        self.assertIsNone(result["roi_percent"])
        self.assertEqual(result["decision"], "not_recommended")

    def test_custom_recommended_threshold(self) -> None:
        result = calculate_roi(
            {
                **BASE_INPUT,
                "action_type": "increase_ventilation",
                "risk_after_action": 0.35,
                "extra_power_kw": 0.1,
                "electricity_price_eur_kwh": 0.25,
            },
            config=ROIConfig(recommended_risk_threshold=0.40),
        )

        self.assertEqual(result["decision"], "recommended")

    def test_rank_actions_by_net_gain(self) -> None:
        actions = [
            {
                **BASE_INPUT,
                "action_type": "migrate_job",
                "migration_time_hours": 0.3,
                "transfer_cost_eur": 1.0,
                "interruption_penalty_eur": 1.0,
            },
            {
                **BASE_INPUT,
                "action_type": "increase_ventilation",
                "extra_power_kw": 0.5,
                "electricity_price_eur_kwh": 0.25,
            },
            {
                **BASE_INPUT,
                "action_type": "reduce_power_cap",
                "performance_loss_percent": 0.20,
            },
        ]

        ranked = rank_actions_by_net_gain(actions)

        self.assertEqual(ranked[0]["action_type"], "increase_ventilation")
        self.assertEqual(ranked[-1]["action_type"], "reduce_power_cap")
        self.assertGreaterEqual(ranked[0]["net_gain_eur"], ranked[1]["net_gain_eur"])

    def test_rejects_invalid_probability(self) -> None:
        with self.assertRaises(ValueError):
            calculate_roi(
                {
                    **BASE_INPUT,
                    "action_type": "no_action",
                    "risk_no_action": 1.2,
                }
            )


if __name__ == "__main__":
    unittest.main()

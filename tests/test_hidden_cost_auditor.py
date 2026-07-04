import json
import unittest

from backend.skills.hidden_cost_auditor import audit_hidden_costs


ROI_RESULT = {
    "action_cost_eur": 1.0,
    "avoided_loss_eur": 5.0,
    "net_gain_eur": 4.0,
    "decision": "recommended",
}
THERMAL_CONTEXT = {
    "risk_no_action": 0.84,
    "risk_after_action": 0.20,
}


class HiddenCostAuditorTests(unittest.TestCase):
    def test_ventilation_audit_matches_expected_shape(self) -> None:
        result = audit_hidden_costs(
            {
                "action_type": "increase_ventilation",
                "extra_power_kw": 1.0,
                "electricity_price_eur_kwh": 0.2,
            },
            ROI_RESULT,
            THERMAL_CONTEXT,
        )

        self.assertIn("ambient_temp_c", result["missing_assumptions"])
        self.assertIn("cooling_headroom_percent", result["missing_assumptions"])
        self.assertIn("fan wear", result["hidden_costs"])
        self.assertTrue(result["should_recalculate_roi"])
        self.assertEqual(result["confidence_level"], "medium")
        json.dumps(result)

    def test_complete_power_cap_context_has_high_confidence(self) -> None:
        result = audit_hidden_costs(
            {
                "action_type": "reduce_power_cap",
                "performance_loss_percent": 0.05,
                "job_duration_extension_hours": 0.2,
                "throughput_reduction_percent": 0.04,
                "sla_delay_assessed": True,
                "training_or_inference_latency_impact": "validated",
                "sla_penalty_eur": 0.0,
            },
            ROI_RESULT,
            THERMAL_CONTEXT,
        )

        self.assertEqual(result["missing_assumptions"], [])
        self.assertEqual(result["confidence_level"], "high")
        self.assertFalse(result["should_recalculate_roi"])

    def test_migration_checks_target_and_cold_start(self) -> None:
        result = audit_hidden_costs(
            {
                "action_type": "migrate_job",
                "migration_time_hours": 0.1,
                "interruption_penalty_eur": 0.0,
                "transfer_cost_eur": 0.0,
                "target_gpu_available": False,
                "other_jobs_impact_assessed": False,
            },
            ROI_RESULT,
            THERMAL_CONTEXT,
        )

        self.assertIn("cold_start_or_reload_cost_eur", result["missing_assumptions"])
        self.assertGreaterEqual(len(result["feasibility_warnings"]), 2)
        self.assertTrue(result["should_recalculate_roi"])

    def test_no_action_checks_failure_and_wasted_time(self) -> None:
        result = audit_hidden_costs(
            {"action_type": "no_action"},
            ROI_RESULT,
            THERMAL_CONTEXT,
        )

        self.assertIn("wasted_gpu_hours", result["missing_assumptions"])
        self.assertIn("possible SLA penalty", result["hidden_costs"])
        self.assertIn("job failure or restart cost", result["hidden_costs"])

    def test_missing_roi_truth_lowers_confidence(self) -> None:
        result = audit_hidden_costs(
            {"action_type": "no_action"},
            {},
            THERMAL_CONTEXT,
        )

        self.assertEqual(result["confidence_level"], "low")
        self.assertTrue(result["should_recalculate_roi"])

    def test_unsupported_action_is_safe_and_serializable(self) -> None:
        result = audit_hidden_costs(
            {"action_type": "unknown"},
            ROI_RESULT,
            THERMAL_CONTEXT,
        )

        self.assertEqual(result["confidence_level"], "low")
        self.assertTrue(result["should_recalculate_roi"])
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()

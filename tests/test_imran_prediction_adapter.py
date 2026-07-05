import os
import unittest
from unittest.mock import patch

from backend.services.imran_prediction_adapter import (
    _build_history,
    predict_temperature_with_imran,
)


class ImranPredictionAdapterTests(unittest.TestCase):
    def test_disabled_adapter_returns_minh_fallback(self) -> None:
        fallback = {"source": "simulator_forecast_placeholder", "risk": "safe"}
        with patch.dict(os.environ, {}, clear=True):
            result = predict_temperature_with_imran({}, fallback)
        self.assertEqual(result["source"], "minh")
        self.assertEqual(result["risk"], "safe")

    def test_builds_explicit_synthetic_history(self) -> None:
        history, source = _build_history(
            {
                "power_draw_w": 310,
                "current_temp_c": 78,
                "predicted_temp_c": 86,
                "utilization_percent": 96,
            },
            31,
        )
        self.assertEqual(source, "synthetic")
        self.assertEqual(len(history or []), 31)
        self.assertEqual(history[-1][1], 78)

    def test_incomplete_input_returns_fallback(self) -> None:
        with patch.dict(os.environ, {"USE_IMRAN_PREDICTION": "true"}):
            result = predict_temperature_with_imran(
                {"gpu_id": "gpu-1"},
                {"source": "minh_simulator", "risk": "warning"},
            )
        self.assertEqual(result["source"], "minh")
        self.assertEqual(result["risk"], "warning")


if __name__ == "__main__":
    unittest.main()

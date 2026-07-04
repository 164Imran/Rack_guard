"""Presentation-only debug overrides for proving frontend/backend connectivity."""

from copy import deepcopy
from typing import Any, Mapping


def apply_extreme_debug(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy with unmistakable temperatures after all calculations."""

    result = deepcopy(dict(decision))
    thermal = result.setdefault("thermal_context", {})
    thermal.update(
        {
            "current_temp_c": 100_000.0,
            "predicted_temp_no_action_c": 120_000.0,
            "predicted_temp_after_action_c": 110_000.0,
            "cooling_gain_c": 10_000.0,
        }
    )
    simulation = result.get("simulation", {})
    simulation.get("before", {})["peak_temp_c"] = 120_000.0
    simulation.get("after", {})["peak_temp_c"] = 110_000.0
    result["debug_mode"] = "extreme"
    return result

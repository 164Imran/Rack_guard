"""Stable adapter around Minh's thermal module with a demo-safe fallback."""

from functools import lru_cache
from typing import Any

ACTION_TO_MINH = {
    "increase_ventilation": "increase_cooling_request",
    "reduce_power_cap": "apply_power_frequency_cap",
    "migrate_job": "migrate_inference_traffic",
}


@lru_cache(maxsize=8)
def _load_fleet(
    rack_count: int = 1,
    gpus_per_rack: int = 8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], str]:
    try:
        from rack_guardians.agent import build_gpu_fleet

        return build_gpu_fleet(rack_count, gpus_per_rack, seed), "minh_simulator"
    except Exception:
        return _mock_fleet(rack_count, gpus_per_rack), "mock_fallback"


def get_current_gpu_state(
    rack_count: int = 1,
    gpus_per_rack: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a compact, frontend-safe GPU snapshot."""

    racks, source = _load_fleet(rack_count, gpus_per_rack, seed)
    normalized_racks = []
    for rack in racks:
        normalized_racks.append(
            {
                "rack_id": rack["rack_id"],
                "status": _normalize_risk(rack.get("top_risk")),
                "current_temp_c": rack.get("rack_temp_c"),
                "gpu_count": rack.get("gpu_count"),
                "gpus": [_compact_gpu(gpu) for gpu in rack.get("gpus", [])],
            }
        )
    return {
        "source": source,
        "mock_fallback": source == "mock_fallback",
        "racks": normalized_racks,
    }


def get_gpu_evaluation(gpu_id: str, seed: int = 42) -> dict[str, Any] | None:
    """Return Minh's complete evaluation for one GPU."""

    racks, _ = _load_fleet(4, 8, seed)
    for rack in racks:
        for gpu in rack.get("gpus", []):
            telemetry = gpu.get("telemetry", {})
            compound_id = f"{telemetry.get('rack_id')}/{telemetry.get('gpu_id')}"
            if gpu_id in {telemetry.get("gpu_id"), compound_id}:
                return gpu
    return None


def get_temperature_prediction(
    gpu_id: str,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Return a compact forecast for one GPU without exposing model internals."""

    evaluation = get_gpu_evaluation(gpu_id, seed)
    if evaluation is None:
        return None

    telemetry = evaluation.get("telemetry", {})
    forecast = evaluation.get("forecast", {})
    trajectory = forecast.get("trajectory", [])
    sample_step = max(1, len(trajectory) // 20)
    sampled_trajectory = trajectory[::sample_step]
    if trajectory and sampled_trajectory[-1] != trajectory[-1]:
        sampled_trajectory.append(trajectory[-1])

    baseline = {
        "gpu_id": telemetry.get("gpu_id"),
        "rack_id": telemetry.get("rack_id"),
        "source": evaluation.get("prediction_source", "minh_simulator"),
        "current_temp_c": forecast.get(
            "current_temp_c",
            telemetry.get("gpu_temp_c"),
        ),
        "predicted_peak_temp_c": forecast.get("peak_temp_c"),
        "predicted_equilibrium_temp_c": forecast.get("convergence_temp_c"),
        "safe_limit_c": forecast.get("threshold_c") or 85.0,
        "time_to_threshold_s": forecast.get("time_to_threshold_s"),
        "risk": _normalize_risk(forecast.get("risk")),
        "confidence": forecast.get("confidence"),
        "trajectory": sampled_trajectory,
    }
    from .imran_prediction_adapter import predict_temperature_with_imran

    return predict_temperature_with_imran(
        {
            "gpu_id": telemetry.get("gpu_id"),
            "rack_id": telemetry.get("rack_id"),
            "power_draw_w": telemetry.get("gpu_power_w"),
            "current_temp_c": baseline["current_temp_c"],
            "predicted_temp_c": baseline["predicted_equilibrium_temp_c"],
            "utilization_percent": telemetry.get("gpu_util_pct"),
        },
        baseline,
    )


def simulate_action(
    gpu_id: str,
    action_type: str,
    seed: int = 42,
) -> dict[str, Any] | None:
    """Simulate one normalized action using Minh's mitigation model."""

    evaluation = get_gpu_evaluation(gpu_id, seed)
    if evaluation is None:
        return None
    forecast = evaluation.get("forecast", {})

    if action_type == "no_action":
        peak = forecast.get("peak_temp_c")
        result = {
            "action_id": "no_action",
            "before": {
                "risk": forecast.get("risk"),
                "peak_temp_c": peak,
                "time_to_threshold_s": forecast.get("time_to_threshold_s"),
            },
            "after": {
                "risk": forecast.get("risk"),
                "peak_temp_c": peak,
                "convergence_temp_c": forecast.get("convergence_temp_c"),
                "time_to_threshold_s": forecast.get("time_to_threshold_s"),
            },
        }
    else:
        minh_action = ACTION_TO_MINH.get(action_type)
        if minh_action is None:
            raise ValueError(f"Unsupported action_type: {action_type}")
        try:
            from rack_guardians.agent import simulate_mitigation

            result = simulate_mitigation(evaluation, minh_action)
        except Exception:
            result = _mock_simulation(evaluation, minh_action)

    before_peak = result["before"].get("peak_temp_c")
    after_peak = result["after"].get("peak_temp_c")
    cooling_gain = (
        round(float(before_peak) - float(after_peak), 2)
        if before_peak is not None and after_peak is not None
        else None
    )
    return {
        "gpu_id": evaluation.get("telemetry", {}).get("gpu_id"),
        "rack_id": evaluation.get("telemetry", {}).get("rack_id"),
        "action_type": action_type,
        "source_action_id": result.get("action_id"),
        "before": {
            **result["before"],
            "risk": _normalize_risk(result["before"].get("risk")),
        },
        "after": {
            **result["after"],
            "risk": _normalize_risk(result["after"].get("risk")),
        },
        "cooling_gain_c": cooling_gain,
    }


def thermal_source_status() -> str:
    """Cheap readiness probe used by /health."""

    try:
        import rack_guardians.agent  # noqa: F401

        return "ready"
    except Exception:
        return "fallback"


def _compact_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    telemetry = gpu.get("telemetry", {})
    forecast = gpu.get("forecast", {})
    return {
        "gpu_id": telemetry.get("gpu_id"),
        "rack_id": telemetry.get("rack_id"),
        "status": _normalize_risk(forecast.get("risk")),
        "current_temp_c": telemetry.get("gpu_temp_c"),
        "predicted_temp_c": forecast.get("convergence_temp_c"),
        "peak_temp_c": forecast.get("peak_temp_c"),
        "time_to_threshold_s": forecast.get("time_to_threshold_s"),
        "power_draw_w": telemetry.get("gpu_power_w"),
        "utilization_percent": telemetry.get("gpu_util_pct"),
    }


def _normalize_risk(risk: object) -> str:
    value = str(risk or "safe").lower()
    if value in {"critical", "high"}:
        return "critical"
    if value in {"watch", "warning", "medium"}:
        return "warning"
    return "safe"


def _mock_fleet(rack_count: int, gpus_per_rack: int) -> list[dict[str, Any]]:
    racks = []
    for rack_index in range(1, rack_count + 1):
        gpus = []
        for gpu_index in range(1, gpus_per_rack + 1):
            current = 84.0 if rack_index == 1 and gpu_index == 1 else 62.0
            predicted = 90.0 if current > 80 else 68.0
            risk = "CRITICAL" if predicted > 85 else "SAFE"
            gpus.append(
                {
                    "telemetry": {
                        "rack_id": f"rack-{rack_index}",
                        "gpu_id": f"gpu-{gpu_index}",
                        "gpu_temp_c": current,
                        "gpu_power_w": 310.0 if current > 80 else 180.0,
                        "gpu_util_pct": 96.0 if current > 80 else 55.0,
                    },
                    "forecast": {
                        "risk": risk,
                        "convergence_temp_c": predicted,
                        "peak_temp_c": predicted,
                        "time_to_threshold_s": 400.0 if current > 80 else None,
                    },
                    "recommendation": {
                        "primary_action": {"action_id": "increase_cooling_request"}
                    },
                }
            )
        racks.append(
            {
                "rack_id": f"rack-{rack_index}",
                "top_risk": gpus[0]["forecast"]["risk"],
                "rack_temp_c": max(gpu["telemetry"]["gpu_temp_c"] for gpu in gpus),
                "gpu_count": len(gpus),
                "gpus": gpus,
            }
        )
    return racks


def _mock_simulation(
    evaluation: dict[str, Any],
    action_id: str,
) -> dict[str, Any]:
    forecast = evaluation.get("forecast", {})
    before_peak = float(forecast.get("peak_temp_c", 90.0))
    reductions = {
        "increase_cooling_request": 8.0,
        "apply_power_frequency_cap": 10.0,
        "migrate_inference_traffic": 14.0,
    }
    after_peak = max(
        float(forecast.get("current_temp_c", 0.0)),
        before_peak - reductions.get(action_id, 0.0),
    )
    return {
        "action_id": action_id,
        "before": {
            "risk": forecast.get("risk", "CRITICAL"),
            "peak_temp_c": before_peak,
            "time_to_threshold_s": forecast.get("time_to_threshold_s"),
        },
        "after": {
            "risk": "SAFE" if after_peak < float(forecast.get("threshold_c", 85.0)) else "CRITICAL",
            "peak_temp_c": after_peak,
            "convergence_temp_c": after_peak,
            "time_to_threshold_s": None,
        },
    }

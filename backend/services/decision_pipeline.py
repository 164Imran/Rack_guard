"""Composable decision pipeline across Minh, ROI, audit, and Imran modules."""

from copy import deepcopy
from typing import Any, Mapping

from ..skills.hidden_cost_auditor import audit_hidden_costs
from .roi_engine import calculate_roi, rank_actions_by_net_gain
from .thermal_adapter import (
    get_gpu_evaluation,
    get_temperature_prediction,
    simulate_action,
)

ACTIONS = (
    "increase_ventilation",
    "reduce_power_cap",
    "migrate_job",
    "no_action",
)

ACTION_LABELS = {
    "increase_ventilation": "Increase ventilation",
    "reduce_power_cap": "Reduce GPU power cap",
    "migrate_job": "Migrate workload",
    "no_action": "No action",
}

# Explicit demo assumptions keep the end-to-end demo operational until business
# values arrive from Imran's module or an operator-provided request.
DEMO_FINANCIAL_INPUTS: dict[str, dict[str, float]] = {
    "increase_ventilation": {
        "gpu_hour_value_eur": 10.0,
        "job_remaining_hours": 4.0,
        "expected_throttle_loss_percent": 0.30,
        "electricity_price_eur_kwh": 0.25,
        "extra_power_kw": 0.15,
    },
    "reduce_power_cap": {
        "gpu_hour_value_eur": 10.0,
        "job_remaining_hours": 4.0,
        "expected_throttle_loss_percent": 0.30,
        "performance_loss_percent": 0.08,
    },
    "migrate_job": {
        "gpu_hour_value_eur": 10.0,
        "job_remaining_hours": 4.0,
        "expected_throttle_loss_percent": 0.30,
        "migration_time_hours": 0.10,
        "transfer_cost_eur": 0.10,
        "interruption_penalty_eur": 0.10,
    },
    "no_action": {
        "gpu_hour_value_eur": 10.0,
        "job_remaining_hours": 4.0,
        "expected_throttle_loss_percent": 0.30,
    },
}

DEFAULT_RISK_SCORES = {
    "safe": 0.12,
    "warning": 0.45,
    "critical": 0.82,
}


def build_final_recommendation(
    gpu_id: str,
    seed: int = 42,
    financial_inputs: Mapping[str, Mapping[str, float]] | None = None,
    risk_scores: Mapping[str, float] | None = None,
) -> dict[str, Any] | None:
    """Run the complete decision pipeline without coupling module internals."""

    prediction = get_temperature_prediction(gpu_id, seed)
    evaluation = get_gpu_evaluation(gpu_id, seed)
    if prediction is None or evaluation is None:
        return None

    configured_inputs = _merge_financial_inputs(financial_inputs)
    configured_risks = {**DEFAULT_RISK_SCORES, **(risk_scores or {})}
    risk_no_action = _risk_value(prediction["risk"], configured_risks)

    simulations: dict[str, dict[str, Any]] = {}
    roi_inputs: dict[str, dict[str, Any]] = {}
    for action_type in ACTIONS:
        simulation = simulate_action(gpu_id, action_type, seed)
        if simulation is None:
            continue
        simulations[action_type] = simulation
        roi_inputs[action_type] = {
            "action_type": action_type,
            "risk_no_action": risk_no_action,
            "risk_after_action": _risk_value(
                simulation["after"]["risk"],
                configured_risks,
            ),
            **configured_inputs[action_type],
        }

    ranked = rank_actions_by_net_gain(roi_inputs.values())
    best_roi = ranked[0]
    best_action = best_roi["action_type"]
    best_simulation = simulations[best_action]
    telemetry = evaluation.get("telemetry", {})
    forecast = evaluation.get("forecast", {})
    diagnosis = evaluation.get("diagnosis", {})
    minh_recommendation = evaluation.get("recommendation", {})

    thermal_context = {
        "current_temp_c": telemetry.get("gpu_temp_c"),
        "predicted_temp_no_action_c": forecast.get("peak_temp_c"),
        "predicted_temp_after_action_c": best_simulation["after"].get(
            "peak_temp_c"
        ),
        "cooling_gain_c": best_simulation.get("cooling_gain_c"),
        "risk_no_action": best_roi["risk_reduction"]
        + roi_inputs[best_action]["risk_after_action"],
        "risk_after_action": roi_inputs[best_action]["risk_after_action"],
        "fan_speed_percent": telemetry.get("fan_command_pct"),
        "ambient_temp_c": telemetry.get("ambient_temp_c"),
        "power_draw_w": telemetry.get("gpu_power_w"),
        "gpu_utilization_percent": telemetry.get("gpu_util_pct"),
    }
    action_context = {
        **roi_inputs[best_action],
        **thermal_context,
        "target_gpu_available": telemetry.get("alternative_capacity_pct") is not None,
    }
    audit = audit_hidden_costs(action_context, best_roi, thermal_context)
    llm_summary = _fallback_report(best_roi, audit)

    return {
        "gpu_id": telemetry.get("gpu_id"),
        "rack_id": telemetry.get("rack_id"),
        "action_label": ACTION_LABELS[best_action],
        "diagnosis": diagnosis,
        "why": diagnosis.get("reasons", []),
        "thermal_context": thermal_context,
        "simulation": best_simulation,
        "roi_result": {
            **best_roi,
            "gpu_hour_value_eur": configured_inputs[best_action][
                "gpu_hour_value_eur"
            ],
            "job_remaining_hours": configured_inputs[best_action][
                "job_remaining_hours"
            ],
        },
        "ranked_actions": ranked,
        "action_scores": minh_recommendation.get("candidates", []),
        "report": evaluation.get("report", {}),
        "migration_plan": None,
        "hidden_cost_audit": audit,
        "llm_report": {
            "summary": llm_summary,
            "provider": "deterministic_fallback",
            "imran_module": "pending",
        },
        "module_status": {
            "minh_thermal": "used",
            "roi_and_audit": "used",
            "imran": "pending",
        },
        "assumptions": {
            "source": "request"
            if financial_inputs is not None
            else "demo_defaults",
            "financial_inputs": configured_inputs,
            "risk_scores": configured_risks,
        },
    }


def _merge_financial_inputs(
    overrides: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    merged = deepcopy(DEMO_FINANCIAL_INPUTS)
    for action_type, values in (overrides or {}).items():
        if action_type in merged:
            merged[action_type].update(values)
    return merged


def _risk_value(risk: object, scores: Mapping[str, float]) -> float:
    normalized = str(risk or "safe").lower()
    return float(scores.get(normalized, scores["warning"]))


def _fallback_report(
    roi_result: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    action = ACTION_LABELS[str(roi_result["action_type"])]
    if audit["should_recalculate_roi"]:
        return (
            f"{action} currently has the highest deterministic net gain, "
            "but missing assumptions should be validated before execution."
        )
    return (
        f"{action} has the highest deterministic net gain and its supplied "
        "cost assumptions passed the hidden-cost checklist."
    )

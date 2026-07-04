"""Deterministic hidden-cost audit for thermal mitigation decisions.

This module never replaces or recalculates the deterministic ROI engine. It only
identifies assumptions and costs that should be validated before trusting a
numeric ROI result.
"""

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, TypedDict

ConfidenceLevel = Literal["high", "medium", "low"]


class HiddenCostAudit(TypedDict):
    missing_assumptions: list[str]
    hidden_costs: list[str]
    operational_risks: list[str]
    feasibility_warnings: list[str]
    confidence_level: ConfidenceLevel
    should_recalculate_roi: bool
    suggested_roi_inputs_to_add: list[str]
    operator_summary: str
    manager_summary: str


@dataclass(frozen=True, slots=True)
class AuditRule:
    required_assumptions: tuple[str, ...]
    roi_relevant_inputs: tuple[str, ...]
    hidden_cost_labels: Mapping[str, str]
    operational_risks: tuple[str, ...]
    feasibility_fields: Mapping[str, str]
    operator_action: str
    manager_risk: str


ACTION_RULES: dict[str, AuditRule] = {
    "increase_ventilation": AuditRule(
        required_assumptions=(
            "extra_power_kw",
            "electricity_price_eur_kwh",
            "fan_wear_cost_eur",
            "extra_hvac_power_kw",
            "hvac_infrastructure_cost_eur",
            "fan_speed_percent",
            "ambient_temp_c",
            "cooling_headroom_percent",
            "neighboring_rack_impact_assessed",
        ),
        roi_relevant_inputs=(
            "extra_power_kw",
            "electricity_price_eur_kwh",
            "fan_wear_cost_eur",
            "extra_hvac_power_kw",
            "hvac_infrastructure_cost_eur",
            "cooling_headroom_percent",
        ),
        hidden_cost_labels={
            "extra_power_kw": "extra electricity consumption",
            "fan_wear_cost_eur": "fan wear",
            "extra_hvac_power_kw": "additional HVAC energy consumption",
            "hvac_infrastructure_cost_eur": "HVAC or cooling infrastructure cost",
        },
        operational_risks=(
            "The cooling gain may be lower than simulated when fan capacity is constrained.",
            "Higher airflow may change thermal conditions for neighboring racks.",
        ),
        feasibility_fields={
            "fan_speed_percent": "Current fan speed is unknown; available fan capacity cannot be confirmed.",
            "cooling_headroom_percent": "Cooling headroom is unknown; the requested airflow increase may not be feasible.",
            "ambient_temp_c": "Ambient temperature is missing; expected cooling effectiveness cannot be validated.",
            "neighboring_rack_impact_assessed": "The effect on neighboring racks has not been assessed.",
        },
        operator_action="Validate fan capacity, ambient temperature, and cooling headroom before applying.",
        manager_risk="Cooling costs and physical cooling limits are not fully modeled.",
    ),
    "reduce_power_cap": AuditRule(
        required_assumptions=(
            "performance_loss_percent",
            "job_duration_extension_hours",
            "throughput_reduction_percent",
            "sla_delay_assessed",
            "training_or_inference_latency_impact",
        ),
        roi_relevant_inputs=(
            "performance_loss_percent",
            "job_duration_extension_hours",
            "throughput_reduction_percent",
            "sla_penalty_eur",
        ),
        hidden_cost_labels={
            "performance_loss_percent": "performance loss",
            "job_duration_extension_hours": "longer job duration",
            "throughput_reduction_percent": "lower throughput",
            "sla_penalty_eur": "possible SLA delay cost",
        },
        operational_risks=(
            "A lower power cap may extend the job and reduce cluster throughput.",
            "Training time or inference latency may increase.",
        ),
        feasibility_fields={
            "sla_delay_assessed": "The effect on the SLA deadline has not been assessed.",
            "training_or_inference_latency_impact": "Training or inference latency impact is unknown.",
        },
        operator_action="Validate runtime, throughput, and latency impact before reducing the power cap.",
        manager_risk="Delivery and SLA effects may be missing from the financial result.",
    ),
    "migrate_job": AuditRule(
        required_assumptions=(
            "migration_time_hours",
            "interruption_penalty_eur",
            "transfer_cost_eur",
            "target_gpu_available",
            "other_jobs_impact_assessed",
            "cold_start_or_reload_cost_eur",
        ),
        roi_relevant_inputs=(
            "migration_time_hours",
            "interruption_penalty_eur",
            "transfer_cost_eur",
            "cold_start_or_reload_cost_eur",
        ),
        hidden_cost_labels={
            "migration_time_hours": "migration time",
            "interruption_penalty_eur": "interruption risk",
            "transfer_cost_eur": "network transfer cost",
            "cold_start_or_reload_cost_eur": "cold start or model reload cost",
        },
        operational_risks=(
            "Migration may interrupt the workload or require a cold start.",
            "The destination rack may displace or slow other jobs.",
        ),
        feasibility_fields={
            "target_gpu_available": "Target GPU availability has not been confirmed.",
            "other_jobs_impact_assessed": "The impact on jobs already using the target rack is unknown.",
        },
        operator_action="Confirm target GPU capacity and migration readiness before moving the job.",
        manager_risk="Availability, interruption, and workload displacement costs may be incomplete.",
    ),
    "no_action": AuditRule(
        required_assumptions=(
            "throttling_risk_assessed",
            "job_slowdown_percent",
            "wasted_gpu_hours",
            "sla_penalty_eur",
            "job_failure_or_restart_risk_assessed",
        ),
        roi_relevant_inputs=(
            "job_slowdown_percent",
            "wasted_gpu_hours",
            "sla_penalty_eur",
            "job_restart_cost_eur",
        ),
        hidden_cost_labels={
            "job_slowdown_percent": "job slowdown",
            "wasted_gpu_hours": "wasted GPU time",
            "sla_penalty_eur": "possible SLA penalty",
            "job_restart_cost_eur": "job failure or restart cost",
        },
        operational_risks=(
            "Thermal throttling may slow the active job.",
            "Continued heating may cause a job failure or restart.",
        ),
        feasibility_fields={
            "throttling_risk_assessed": "The probability of thermal throttling has not been validated.",
            "job_failure_or_restart_risk_assessed": "Job failure and restart exposure has not been assessed.",
        },
        operator_action="Validate throttling and restart exposure before choosing no action.",
        manager_risk="The cost of delay, wasted GPU time, and possible failure may be understated.",
    ),
}

REQUIRED_ROI_RESULT_FIELDS = (
    "action_cost_eur",
    "avoided_loss_eur",
    "net_gain_eur",
    "decision",
)
REQUIRED_THERMAL_FIELDS = ("risk_no_action", "risk_after_action")


def audit_hidden_costs(
    action_context: Mapping[str, object],
    roi_result: Mapping[str, object],
    thermal_context: Mapping[str, object],
) -> HiddenCostAudit:
    """Audit one ROI decision without inventing or recalculating numeric values."""

    action_type = action_context.get("action_type")
    if not isinstance(action_type, str) or action_type not in ACTION_RULES:
        return _unsupported_action_audit(action_type)

    rule = ACTION_RULES[action_type]
    combined_context = {**thermal_context, **action_context}

    missing = [
        field
        for field in rule.required_assumptions
        if not _has_value(combined_context, field)
    ]
    missing.extend(
        field
        for field in REQUIRED_THERMAL_FIELDS
        if not _has_value(thermal_context, field)
    )
    missing.extend(
        field
        for field in REQUIRED_ROI_RESULT_FIELDS
        if not _has_value(roi_result, field)
    )
    missing = _unique(missing)

    hidden_costs = _unique(
        [
            label
            for field, label in rule.hidden_cost_labels.items()
            if not _has_value(combined_context, field)
        ]
    )
    feasibility_warnings = _unique(
        [
            warning
            for field, warning in rule.feasibility_fields.items()
            if not _is_confirmed(combined_context, field)
        ]
    )
    suggested_inputs = _unique(
        field
        for field in rule.roi_relevant_inputs
        if not _has_value(combined_context, field)
    )

    missing_numeric_truth = any(
        field in missing
        for field in (*REQUIRED_ROI_RESULT_FIELDS, *REQUIRED_THERMAL_FIELDS)
    )
    should_recalculate = bool(suggested_inputs or missing_numeric_truth)
    confidence = _confidence_level(missing, missing_numeric_truth)

    if missing:
        operator_summary = (
            f"{rule.operator_action} Missing checks: {', '.join(missing)}."
        )
        manager_summary = (
            f"Audit confidence is {confidence}. {rule.manager_risk} "
            f"Recalculate ROI: {'yes' if should_recalculate else 'no'}."
        )
    else:
        operator_summary = (
            "No hidden-cost assumptions are missing from the supplied context. "
            "Proceed using the deterministic ROI result."
        )
        manager_summary = (
            "Audit confidence is high. The supplied cost and feasibility assumptions "
            "cover the deterministic ROI decision."
        )

    return {
        "missing_assumptions": missing,
        "hidden_costs": hidden_costs,
        "operational_risks": list(rule.operational_risks),
        "feasibility_warnings": feasibility_warnings,
        "confidence_level": confidence,
        "should_recalculate_roi": should_recalculate,
        "suggested_roi_inputs_to_add": suggested_inputs,
        "operator_summary": operator_summary,
        "manager_summary": manager_summary,
    }


def _has_value(context: Mapping[str, object], field: str) -> bool:
    return field in context and context[field] is not None


def _is_confirmed(context: Mapping[str, object], field: str) -> bool:
    if not _has_value(context, field):
        return False
    value = context[field]
    if field.endswith("_assessed") or field == "target_gpu_available":
        return value is True
    return True


def _confidence_level(
    missing: list[str],
    missing_numeric_truth: bool,
) -> ConfidenceLevel:
    if missing_numeric_truth:
        return "low"
    if missing:
        return "medium"
    return "high"


def _unsupported_action_audit(action_type: object) -> HiddenCostAudit:
    action_label = action_type if isinstance(action_type, str) else "missing"
    return {
        "missing_assumptions": ["action_type"],
        "hidden_costs": [],
        "operational_risks": [
            f"Hidden-cost rules are not defined for action type: {action_label}."
        ],
        "feasibility_warnings": [
            "A supported action type is required before feasibility can be audited."
        ],
        "confidence_level": "low",
        "should_recalculate_roi": True,
        "suggested_roi_inputs_to_add": [],
        "operator_summary": "Select a supported mitigation action before proceeding.",
        "manager_summary": "The decision cannot be audited because its action type is unsupported.",
    }


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))

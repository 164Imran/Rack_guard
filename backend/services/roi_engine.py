"""Pure financial ROI calculations for thermal mitigation actions."""

from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, TypedDict, cast

ActionType = Literal[
    "increase_ventilation",
    "reduce_power_cap",
    "migrate_job",
    "no_action",
]
Decision = Literal["recommended", "partial", "not_recommended"]

SUPPORTED_ACTIONS: frozenset[str] = frozenset(
    {
        "increase_ventilation",
        "reduce_power_cap",
        "migrate_job",
        "no_action",
    }
)


@dataclass(frozen=True, slots=True)
class ROIConfig:
    """Configurable decision policy for Module C."""

    recommended_risk_threshold: float = 0.30

    def __post_init__(self) -> None:
        _validate_probability(
            "recommended_risk_threshold",
            self.recommended_risk_threshold,
        )


@dataclass(frozen=True, slots=True)
class ROIInput:
    """Thermal risk and business assumptions for one candidate action."""

    action_type: ActionType
    risk_no_action: float
    risk_after_action: float
    gpu_hour_value_eur: float
    job_remaining_hours: float
    expected_throttle_loss_percent: float
    electricity_price_eur_kwh: float = 0.0
    extra_power_kw: float = 0.0
    performance_loss_percent: float = 0.0
    migration_time_hours: float = 0.0
    transfer_cost_eur: float = 0.0
    interruption_penalty_eur: float = 0.0

    def __post_init__(self) -> None:
        if self.action_type not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        _validate_probability("risk_no_action", self.risk_no_action)
        _validate_probability("risk_after_action", self.risk_after_action)
        _validate_probability(
            "expected_throttle_loss_percent",
            self.expected_throttle_loss_percent,
        )
        _validate_probability("performance_loss_percent", self.performance_loss_percent)

        for field_name in (
            "gpu_hour_value_eur",
            "job_remaining_hours",
            "electricity_price_eur_kwh",
            "extra_power_kw",
            "migration_time_hours",
            "transfer_cost_eur",
            "interruption_penalty_eur",
        ):
            _validate_non_negative(field_name, getattr(self, field_name))


class ROIResult(TypedDict):
    """Serializable result returned by the ROI engine."""

    action_type: ActionType
    risk_reduction: float
    avoided_loss_eur: float
    action_cost_eur: float
    net_gain_eur: float
    roi_percent: float | None
    decision: Decision


def calculate_roi(
    action: ROIInput | Mapping[str, object],
    config: ROIConfig | None = None,
) -> ROIResult:
    """Calculate the financial outcome of one thermal mitigation action."""

    roi_input = _coerce_input(action)
    policy = config or ROIConfig()

    risk_reduction = roi_input.risk_no_action - roi_input.risk_after_action
    avoided_loss = (
        risk_reduction
        * roi_input.gpu_hour_value_eur
        * roi_input.job_remaining_hours
        * roi_input.expected_throttle_loss_percent
    )
    action_cost = _calculate_action_cost(roi_input)
    net_gain = avoided_loss - action_cost
    roi_percent = (net_gain / action_cost) * 100 if action_cost > 0 else None

    if net_gain <= 0:
        decision: Decision = "not_recommended"
    elif roi_input.risk_after_action < policy.recommended_risk_threshold:
        decision = "recommended"
    else:
        decision = "partial"

    return {
        "action_type": roi_input.action_type,
        "risk_reduction": risk_reduction,
        "avoided_loss_eur": avoided_loss,
        "action_cost_eur": action_cost,
        "net_gain_eur": net_gain,
        "roi_percent": roi_percent,
        "decision": decision,
    }


def rank_actions_by_net_gain(
    actions: Iterable[ROIInput | Mapping[str, object]],
    config: ROIConfig | None = None,
) -> list[ROIResult]:
    """Calculate and rank candidate actions from highest to lowest net gain."""

    results = [calculate_roi(action, config=config) for action in actions]
    return sorted(results, key=lambda result: result["net_gain_eur"], reverse=True)


def _calculate_action_cost(action: ROIInput) -> float:
    if action.action_type == "increase_ventilation":
        return (
            action.extra_power_kw
            * action.job_remaining_hours
            * action.electricity_price_eur_kwh
        )
    if action.action_type == "reduce_power_cap":
        return (
            action.gpu_hour_value_eur
            * action.job_remaining_hours
            * action.performance_loss_percent
        )
    if action.action_type == "migrate_job":
        return (
            action.migration_time_hours * action.gpu_hour_value_eur
            + action.transfer_cost_eur
            + action.interruption_penalty_eur
        )
    return 0.0


def _coerce_input(action: ROIInput | Mapping[str, object]) -> ROIInput:
    if isinstance(action, ROIInput):
        return action
    if not isinstance(action, Mapping):
        raise TypeError("action must be an ROIInput or a mapping")

    try:
        return ROIInput(**cast(dict, dict(action)))
    except TypeError as error:
        raise ValueError(f"Invalid ROI input: {error}") from error


def _validate_probability(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_non_negative(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")

"""
Bloc B for Rack Guardian: rule-first diagnosis, multimodal evidence handling,
and one-tap mitigation recommendations.

The module is intentionally independent from the current Bloc A model internals.
It can consume any forecast shaped as [power_w, gpu_temp_c, heatsink_temp_c], so
the simulator, NODE, DLinear, or a later production forecaster can feed it.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
import urllib.error
import urllib.request
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from . import simulator
except ImportError:  # pragma: no cover - useful when run from rack_guardians/
    import simulator

try:
    from . import bloc_a_predictor
except ImportError:  # pragma: no cover - useful when run from rack_guardians/
    try:
        import bloc_a_predictor
    except ImportError:  # pragma: no cover - NODE is optional at runtime
        bloc_a_predictor = None


CRUSOE_BASE_URL = "https://api.inference.crusoecloud.com/v1/"
CRUSOE_MODEL = "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B"
DEFAULT_THRESHOLD_C = float(getattr(simulator, "T_THRESH", 70.0))
GPU_MAX_TRAFFIC_PCT = 100.0
GPU_SAFE_TRAFFIC_PCT = 72.0
GPU_RACK_SIZE = 8


@dataclass
class RackTelemetry:
    rack_id: str = "rack-4"
    gpu_id: str = "gpu-1"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    gpu_temp_c: Optional[float] = None
    heatsink_temp_c: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    coolant_temp_c: Optional[float] = None
    fan_speed_rpm: Optional[float] = None
    fan_command_pct: Optional[float] = None
    cooling_flow_lpm: Optional[float] = None
    cooling_command_pct: Optional[float] = None
    gpu_power_w: Optional[float] = None
    rack_power_kw: Optional[float] = None
    psu_voltage_v: Optional[float] = None
    psu_voltage_std_v: Optional[float] = None
    power_spike_ratio: Optional[float] = None
    workload_type: str = "inference"
    gpu_util_pct: Optional[float] = None
    sm_util_pct: Optional[float] = None
    memory_util_pct: Optional[float] = None
    inference_queue_len: Optional[int] = None
    request_rate_rps: Optional[float] = None
    active_batches: Optional[int] = None
    assigned_traffic_pct: Optional[float] = None
    workload_demand_units: Optional[float] = None
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    network_latency_ms: Optional[float] = None
    packet_loss_pct: Optional[float] = None
    frequency_mhz: Optional[float] = None
    sensor_age_s: Optional[float] = None
    alternative_capacity_pct: Optional[float] = None
    latency_tolerance_ms: Optional[float] = None
    batch_jobs_present: bool = False
    missing_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RackTelemetry":
        allowed = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in (data or {}).items() if k in allowed}
        return cls(**clean)


@dataclass
class ForecastSummary:
    threshold_c: float
    horizon_s: float
    current_temp_c: float
    peak_temp_c: float
    convergence_temp_c: float
    time_to_threshold_s: Optional[float]
    risk: str
    confidence: float
    trajectory: list[dict[str, float]] = field(default_factory=list)


@dataclass
class Diagnosis:
    likely_cause: str
    confidence: float
    reasons: list[str]
    request_human_evidence: bool = False
    evidence_summary: str = ""


@dataclass
class ActionCandidate:
    action_id: str
    label: str
    score: float
    expected_impact: str
    rationale: str
    operational_cost: str


@dataclass
class Recommendation:
    primary_action: ActionCandidate
    candidates: list[ActionCandidate]
    secondary_actions: list[str] = field(default_factory=list)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    return float(value) if _is_number(value) else default


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classify_workload(telemetry: RackTelemetry) -> str:
    spike = _safe_float(telemetry.power_spike_ratio, 1.0) or 1.0
    sm = _safe_float(telemetry.sm_util_pct, 0.0) or 0.0
    mem = _safe_float(telemetry.memory_util_pct, 0.0) or 0.0
    gpu = _safe_float(telemetry.gpu_util_pct, 0.0) or 0.0

    if gpu < 20 and sm < 20:
        return "idle_or_light"
    if spike >= 1.25 and sm >= 70:
        return "high_spike_compute"
    if mem >= 70 and sm < 65:
        return "memory_bound"
    if sm >= 65 and mem >= 65:
        return "mixed_compute_memory"
    return "steady_inference"


def classify_risk(
    current_temp_c: float,
    peak_temp_c: float,
    convergence_temp_c: float,
    time_to_threshold_s: Optional[float],
    threshold_c: float,
) -> str:
    if time_to_threshold_s is not None and time_to_threshold_s <= 6 * 60:
        return "CRITICAL"
    if peak_temp_c >= threshold_c or convergence_temp_c >= threshold_c:
        return "HIGH"
    if peak_temp_c >= threshold_c - 5 or current_temp_c >= threshold_c - 8:
        return "WATCH"
    return "SAFE"


def forecast_from_trajectory(
    trajectory: Any,
    dt: float,
    threshold_c: float = DEFAULT_THRESHOLD_C,
    confidence: float = 0.86,
    max_points: int = 180,
) -> ForecastSummary:
    rows = trajectory.tolist() if hasattr(trajectory, "tolist") else list(trajectory)
    if not rows:
        raise ValueError("trajectory must contain at least one row")

    temps = [float(row[1]) for row in rows]
    current = temps[0]
    peak = max(temps)
    tail = temps[-min(len(temps), max(12, int(round(60.0 / max(dt, 1e-6))))):]
    convergence = sum(tail) / len(tail)

    crossing = None
    for idx, temp in enumerate(temps):
        if temp >= threshold_c:
            crossing = idx * dt
            break

    risk = classify_risk(current, peak, convergence, crossing, threshold_c)
    step = max(1, int(math.ceil(len(rows) / max_points)))
    points = []
    for idx in range(0, len(rows), step):
        row = rows[idx]
        points.append({
            "t_s": round(idx * dt, 2),
            "power_w": round(float(row[0]), 3),
            "gpu_temp_c": round(float(row[1]), 3),
            "heatsink_temp_c": round(float(row[2]), 3),
        })

    return ForecastSummary(
        threshold_c=threshold_c,
        horizon_s=(len(rows) - 1) * dt,
        current_temp_c=current,
        peak_temp_c=peak,
        convergence_temp_c=convergence,
        time_to_threshold_s=crossing,
        risk=risk,
        confidence=confidence,
        trajectory=points,
    )


def forecast_with_bloc_a(
    telemetry: RackTelemetry,
    observed_window: Any = None,
    context: Optional[dict[str, Any]] = None,
) -> Optional[ForecastSummary]:
    """Use Bloc A's Neural ODE checkpoint when its three-channel input is available.

    Bloc A currently consumes only ``power_w``, ``gpu_temp_c`` and
    ``heatsink_temp_c``. Bloc B still keeps richer telemetry for diagnosis.
    """

    if bloc_a_predictor is None:
        return None
    prediction = bloc_a_predictor.predict_gpu_temperature(
        observed_window=observed_window,
        telemetry=telemetry,
        context=context or {},
    )
    if not prediction.get("ok"):
        return None
    points = prediction.get("trajectory") or []
    if not points:
        return None
    forecast = _forecast_from_points(
        points,
        threshold_c=DEFAULT_THRESHOLD_C,
        confidence=float(prediction.get("confidence", 0.78)),
    )
    forecast.prediction_source = prediction.get("source", "bloc_a_neural_ode")
    forecast.bloc_a = {
        "source": prediction.get("source", "bloc_a_neural_ode"),
        "input_mode": prediction.get("input_mode"),
        "model_meta": prediction.get("model_meta", {}),
        "input_schema": prediction.get("input_schema", {}),
        "power_scaled_for_node": prediction.get("power_scaled_for_node", False),
        "context_adjustment_c": prediction.get("context_adjustment_c", 0.0),
        "observed_window": prediction.get("observed_window", []),
    }
    return forecast


def bloc_a_input_schema() -> dict[str, Any]:
    if bloc_a_predictor is None:
        return {
            "available": False,
            "reason": "bloc_a_predictor module is not importable",
            "required_channels": ["power_w", "gpu_temp_c", "heatsink_temp_c"],
        }
    schema = bloc_a_predictor.input_schema()
    schema["available"] = bloc_a_predictor.model_available()
    schema["note"] = (
        "Bloc A predicts temperature from three thermal channels. "
        "Bloc B keeps richer telemetry for diagnosis and recommendations."
    )
    return schema


def forecasts_many_with_bloc_a(items: list[dict[str, Any]]) -> list[Optional[ForecastSummary]]:
    if bloc_a_predictor is None:
        return [None for _ in items]
    predictions = bloc_a_predictor.predict_gpu_temperature_batch(items)
    forecasts: list[Optional[ForecastSummary]] = []
    for prediction in predictions:
        if not prediction.get("ok"):
            forecasts.append(None)
            continue
        forecast = _forecast_from_points(
            prediction.get("trajectory") or [],
            threshold_c=DEFAULT_THRESHOLD_C,
            confidence=float(prediction.get("confidence", 0.78)),
        )
        forecast.prediction_source = prediction.get("source", "bloc_a_neural_ode")
        forecast.bloc_a = {
            "source": prediction.get("source", "bloc_a_neural_ode"),
            "input_mode": prediction.get("input_mode"),
            "model_meta": prediction.get("model_meta", {}),
            "input_schema": prediction.get("input_schema", {}),
            "power_scaled_for_node": prediction.get("power_scaled_for_node", False),
            "context_adjustment_c": prediction.get("context_adjustment_c", 0.0),
            "observed_window": prediction.get("observed_window", []),
        }
        forecasts.append(forecast)
    return forecasts


def _forecast_from_points(
    points: list[dict[str, Any]],
    threshold_c: float = DEFAULT_THRESHOLD_C,
    confidence: float = 0.82,
    max_points: int = 180,
) -> ForecastSummary:
    temps = [float(point["gpu_temp_c"]) for point in points]
    current = temps[0]
    peak = max(temps)
    horizon_s = float(points[-1].get("t_s", 0.0)) if points else 0.0
    tail_start = max(0.0, horizon_s - 60.0)
    tail = [float(point["gpu_temp_c"]) for point in points if float(point.get("t_s", 0.0)) >= tail_start]
    if not tail:
        tail = temps[-min(len(temps), 12):]

    crossing = None
    for point in points:
        if float(point["gpu_temp_c"]) >= threshold_c:
            crossing = float(point.get("t_s", 0.0))
            break

    step = max(1, int(math.ceil(len(points) / max_points)))
    sampled = []
    for point in points[::step]:
        sampled.append({
            "t_s": round(float(point.get("t_s", 0.0)), 3),
            "power_w": round(float(point.get("power_w", 0.0)), 3),
            "gpu_temp_c": round(float(point.get("gpu_temp_c", 0.0)), 3),
            "heatsink_temp_c": round(float(point.get("heatsink_temp_c", 0.0)), 3),
        })
    if sampled and sampled[-1]["t_s"] != round(float(points[-1].get("t_s", 0.0)), 3):
        point = points[-1]
        sampled.append({
            "t_s": round(float(point.get("t_s", 0.0)), 3),
            "power_w": round(float(point.get("power_w", 0.0)), 3),
            "gpu_temp_c": round(float(point.get("gpu_temp_c", 0.0)), 3),
            "heatsink_temp_c": round(float(point.get("heatsink_temp_c", 0.0)), 3),
        })

    convergence = sum(tail) / len(tail)
    return ForecastSummary(
        threshold_c=threshold_c,
        horizon_s=horizon_s,
        current_temp_c=current,
        peak_temp_c=peak,
        convergence_temp_c=convergence,
        time_to_threshold_s=crossing,
        risk=classify_risk(current, peak, convergence, crossing, threshold_c),
        confidence=round(_clip(confidence, 0.0, 0.99), 3),
        trajectory=sampled,
    )


def diagnose(telemetry: RackTelemetry, forecast: ForecastSummary) -> Diagnosis:
    reasons: list[str] = []
    missing = set(telemetry.missing_fields)
    for field_name in ("gpu_temp_c", "gpu_power_w", "cooling_flow_lpm", "fan_speed_rpm", "psu_voltage_v"):
        if getattr(telemetry, field_name, None) is None:
            missing.add(field_name)

    temp_rising = forecast.peak_temp_c - forecast.current_temp_c >= 4.0
    if temp_rising:
        reasons.append("Forecast temperature is rising across the prediction horizon.")

    flow = _safe_float(telemetry.cooling_flow_lpm)
    cooling_command = _safe_float(telemetry.cooling_command_pct, 0.0) or 0.0
    fan_speed = _safe_float(telemetry.fan_speed_rpm)
    fan_command = _safe_float(telemetry.fan_command_pct, 0.0) or 0.0
    psu_voltage = _safe_float(telemetry.psu_voltage_v)
    psu_std = _safe_float(telemetry.psu_voltage_std_v, 0.0) or 0.0
    latency = _safe_float(telemetry.network_latency_ms, 0.0) or 0.0
    packet_loss = _safe_float(telemetry.packet_loss_pct, 0.0) or 0.0
    queue = telemetry.inference_queue_len if telemetry.inference_queue_len is not None else 0
    power = _safe_float(telemetry.gpu_power_w, 0.0) or 0.0
    sm = _safe_float(telemetry.sm_util_pct, 0.0) or 0.0

    contradictions = []
    if telemetry.gpu_temp_c is not None and abs(telemetry.gpu_temp_c - forecast.current_temp_c) > 18:
        contradictions.append("live GPU temperature conflicts with the forecast starting point")
    if telemetry.gpu_temp_c is not None and telemetry.ambient_temp_c is not None and telemetry.gpu_temp_c + 2 < telemetry.ambient_temp_c:
        contradictions.append("GPU temperature is below ambient by an implausible margin")
    if telemetry.heatsink_temp_c is not None and telemetry.gpu_temp_c is not None:
        if telemetry.heatsink_temp_c + 45 < telemetry.gpu_temp_c:
            contradictions.append("GPU and heatsink sensors disagree sharply")
    if telemetry.sensor_age_s is not None and telemetry.sensor_age_s > 120:
        contradictions.append("one or more telemetry streams are stale")

    if contradictions or len(missing) >= 3:
        reasons.extend(contradictions)
        if missing:
            reasons.append(f"Missing telemetry fields: {', '.join(sorted(missing))}.")
        return Diagnosis(
            likely_cause="telemetry_fault_or_insufficient_evidence",
            confidence=0.34 if contradictions else 0.45,
            reasons=reasons,
            request_human_evidence=True,
        )

    if temp_rising and flow is not None and flow < 0.75 and cooling_command >= 70:
        reasons.append("Cooling command is high but measured liquid flow is low.")
        return Diagnosis("liquid_cooling_degradation", 0.88, reasons)

    if temp_rising and fan_speed is not None and fan_command >= 70 and fan_speed < 3500:
        reasons.append("Fan command is high but fan RPM is low.")
        return Diagnosis("fan_failure", 0.86, reasons)

    if temp_rising and psu_voltage is not None and (psu_voltage < 11.5 or psu_voltage > 12.6 or psu_std > 0.25):
        reasons.append("PSU voltage is outside the stable operating band.")
        return Diagnosis("power_supply_instability", 0.82, reasons)

    if temp_rising and latency > 80 and queue >= 50:
        reasons.append("Network latency and inference queue depth are both elevated.")
        return Diagnosis("network_induced_queue_buildup", 0.78, reasons)

    if temp_rising and power >= 80 and sm >= 70:
        reasons.append("Power and SM utilization indicate high-compute heat load while cooling looks normal.")
        return Diagnosis("workload_induced_heating", 0.76, reasons)

    if forecast.risk in ("CRITICAL", "HIGH"):
        reasons.append("Thermal risk is high, but telemetry does not isolate a cause.")
        return Diagnosis("unknown_thermal_risk", 0.52, reasons, request_human_evidence=True)

    reasons.append("Forecast remains below the thermal threshold.")
    return Diagnosis("no_active_thermal_incident", 0.74, reasons)


def recommend_actions(
    telemetry: RackTelemetry,
    forecast: ForecastSummary,
    diagnosis: Diagnosis,
    workload_class: str,
) -> Recommendation:
    if forecast.risk == "SAFE" and diagnosis.likely_cause == "no_active_thermal_incident":
        monitor = ActionCandidate(
            "continue_monitoring",
            "No mitigation needed",
            99.0,
            "GPU remains below threshold with stable telemetry.",
            "The forecast and observed telemetry do not indicate an active GPU thermal incident.",
            "none",
        )
        return Recommendation(monitor, [monitor], ["Keep standard telemetry sampling active."])

    confidence_penalty = (1.0 - diagnosis.confidence) * 18.0
    risk_base = {"CRITICAL": 82.0, "HIGH": 66.0, "WATCH": 42.0, "SAFE": 12.0}.get(forecast.risk, 35.0)
    capacity = _safe_float(telemetry.alternative_capacity_pct, 0.0) or 0.0
    latency_tolerance = _safe_float(telemetry.latency_tolerance_ms, 0.0) or 0.0
    queue = float(telemetry.inference_queue_len or 0)

    def score(action_id: str, label: str, reduction: float, latency: float, migration: float,
              sla: float, preference: float, impact: str, rationale: str, cost: str) -> ActionCandidate:
        value = risk_base * reduction - latency - migration - sla - preference - confidence_penalty
        return ActionCandidate(action_id, label, round(value, 2), impact, rationale, cost)

    candidates = [
        score(
            "migrate_inference_traffic",
            "Migrate high-compute inference traffic",
            0.82 if capacity >= 15 else 0.42,
            8.0 if latency_tolerance >= 25 else 16.0,
            8.0,
            6.0,
            0.0,
            "Expected to reduce this GPU's heat load by roughly 20-35%.",
            "Best when thermal risk is high and nearby capacity exists.",
            "medium",
        ),
        score(
            "apply_power_frequency_cap",
            "Apply temporary power/frequency cap",
            0.58 if workload_class == "high_spike_compute" else 0.38,
            7.0,
            1.0,
            9.0,
            0.0,
            "Expected to flatten power spikes and lower peak temperature.",
            "Useful for power-spiky workloads when latency tolerance exists.",
            "low",
        ),
        score(
            "increase_cooling_request",
            "Increase cooling request",
            0.46 if diagnosis.likely_cause != "liquid_cooling_degradation" else 0.22,
            0.0,
            0.0,
            1.0,
            0.0,
            "Expected to help only if the cooling loop serving this GPU is responsive.",
            "Low operational cost, but limited if flow is already degraded.",
            "low",
        ),
        score(
            "move_batch_jobs",
            "Move non-urgent batch jobs",
            0.35 if telemetry.batch_jobs_present else 0.12,
            0.0,
            4.0,
            2.0,
            0.0,
            "Expected to remove background heat without touching live inference.",
            "Best when batch jobs are present on the hot rack.",
            "low",
        ),
        score(
            "add_gpu_rack_capacity",
            "Add GPU rack capacity",
            0.72 if capacity < 8 and queue >= 60 else 0.22,
            0.0,
            18.0,
            4.0,
            0.0,
            "Expected to reduce sustained queue pressure when all nearby GPUs are already near safe load.",
            "Best when thermal risk is driven by demand and there is not enough safe headroom to migrate traffic.",
            "high",
        ),
        score(
            "escalate_technician",
            "Escalate technician inspection",
            0.30 if diagnosis.likely_cause in {
                "liquid_cooling_degradation", "fan_failure", "power_supply_instability"
            } else 0.08,
            0.0,
            0.0,
            0.0,
            0.0,
            "Does not immediately reduce heat, but addresses likely GPU, cooling, or facility cause.",
            "Needed when telemetry points to cooling, fan, power, or physical evidence.",
            "high",
        ),
        score(
            "request_human_evidence",
            "Request human evidence",
            0.20,
            0.0,
            0.0,
            1.0,
            0.0,
            "Improves confidence before costly mitigation.",
            "Best when telemetry is missing, stale, contradictory, or diagnosis confidence is low.",
            "low",
        ),
    ]

    if diagnosis.request_human_evidence or diagnosis.confidence < 0.55:
        for candidate in candidates:
            if candidate.action_id == "request_human_evidence":
                candidate.score = max(candidate.score, 91.0)

    if diagnosis.likely_cause == "liquid_cooling_degradation":
        for candidate in candidates:
            if candidate.action_id == "escalate_technician":
                candidate.score += 18.0
            if candidate.action_id == "migrate_inference_traffic":
                candidate.score += 12.0

    if diagnosis.likely_cause == "fan_failure":
        for candidate in candidates:
            if candidate.action_id in ("migrate_inference_traffic", "escalate_technician"):
                candidate.score += 12.0

    if capacity < 8 and queue >= 60 and diagnosis.likely_cause in {
        "workload_induced_heating", "network_induced_queue_buildup", "unknown_thermal_risk"
    }:
        for candidate in candidates:
            if candidate.action_id == "add_gpu_rack_capacity":
                candidate.score += 28.0

    candidates.sort(key=lambda c: c.score, reverse=True)
    secondary = []
    if diagnosis.likely_cause in ("liquid_cooling_degradation", "fan_failure", "power_supply_instability"):
        secondary.append("Open a facility inspection ticket with the forecast and sensor evidence.")
    if candidates[0].action_id != "request_human_evidence" and diagnosis.confidence < 0.68:
        secondary.append("Ask the shift engineer for a short observation note or GPU/rack photo.")

    return Recommendation(candidates[0], candidates, secondary)


class CrusoeEvidenceClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = CRUSOE_BASE_URL,
        model: str = CRUSOE_MODEL,
        timeout_s: int = 60,
    ) -> None:
        self.api_key = api_key or os.getenv("CRUSOE_API_KEY")
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def analyze(
        self,
        prompt: str,
        text_note: str = "",
        image_data_url: str = "",
        audio_data_url: str = "",
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("CRUSOE_API_KEY is not set")

        content: list[dict[str, Any]] = []
        if image_data_url:
            content.append({"type": "image_url", "image_url": {"url": image_data_url}})
        if audio_data_url:
            content.append({"type": "audio_url", "audio_url": {"url": audio_data_url}})
        content.append({"type": "text", "text": prompt + ("\n\nOperator note:\n" + text_note if text_note else "")})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": (
                    "You are Rack Guardian, a datacenter thermal incident assistant. "
                    "Use telemetry and human evidence to produce concise engineering JSON. "
                    "Do not invent observations that are not in the evidence."
                )},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "max_tokens": 1400,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Crusoe HTTP {exc.code}: {body}") from exc

        body = json.loads(raw)
        message = body["choices"][0]["message"]
        text = _message_content_to_text(message.get("content") or message.get("reasoning_content") or "")
        return _extract_json_object(text)


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for block in content:
            if isinstance(block, dict):
                chunks.append(str(block.get("text", "")))
            else:
                chunks.append(str(block))
        return "".join(chunks)
    return str(content or "")


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"raw_response": text}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"raw_response": text}


def _build_evidence_prompt(
    telemetry: RackTelemetry,
    forecast: ForecastSummary,
    diagnosis: Diagnosis,
    recommendation: Recommendation,
    workload_class: str,
) -> str:
    context = {
        "telemetry": asdict(telemetry),
        "forecast": {
            "threshold_c": forecast.threshold_c,
            "horizon_s": forecast.horizon_s,
            "current_temp_c": round(forecast.current_temp_c, 2),
            "peak_temp_c": round(forecast.peak_temp_c, 2),
            "convergence_temp_c": round(forecast.convergence_temp_c, 2),
            "time_to_threshold_s": forecast.time_to_threshold_s,
            "risk": forecast.risk,
            "confidence": forecast.confidence,
        },
        "workload_class": workload_class,
        "rule_diagnosis": asdict(diagnosis),
        "rule_recommendation": asdict(recommendation.primary_action),
    }
    return (
        "Review this rack overheating case and any attached speech/image/text evidence. "
        "Return one JSON object with keys: evidence_summary, updated_diagnosis, "
        "confidence_delta, recommendation_override, report. The report must contain "
        "incident_summary, likely_cause, evidence_used, operator_next_step, and escalation_note.\n\n"
        f"Case context:\n{json.dumps(context, indent=2)}"
    )


def _local_evidence_update(text_note: str, has_image: bool, has_audio: bool) -> dict[str, Any]:
    note = (text_note or "").lower()
    evidence_bits = []
    if text_note:
        evidence_bits.append("operator text note")
    if has_image:
        evidence_bits.append("uploaded image")
    if has_audio:
        evidence_bits.append("uploaded audio")

    updated = ""
    if any(word in note for word in ("leak", "coolant", "drip", "puddle", "wet", "flow alarm")):
        updated = "liquid_cooling_degradation"
    elif any(word in note for word in ("fan", "rpm", "rattle", "stopped", "not spinning")):
        updated = "fan_failure"
    elif any(word in note for word in ("psu", "voltage", "breaker", "power shelf", "brownout")):
        updated = "power_supply_instability"
    elif any(word in note for word in ("network", "queue", "packet", "latency")):
        updated = "network_induced_queue_buildup"
    elif any(word in note for word in ("sensor", "dashboard", "stale", "n/a", "missing")):
        updated = "telemetry_fault_or_insufficient_evidence"

    return {
        "evidence_summary": (
            f"Local evidence parser saw {', '.join(evidence_bits) or 'no human evidence'}. "
            "Image/audio content requires Crusoe Nemotron for semantic analysis."
        ),
        "updated_diagnosis": updated,
        "confidence_delta": 0.38 if updated else 0.0,
        "recommendation_override": "",
        "report": {},
    }


def _apply_evidence_update(
    diagnosis: Diagnosis,
    recommendation: Recommendation,
    evidence_update: dict[str, Any],
) -> tuple[Diagnosis, Recommendation]:
    updated_cause = evidence_update.get("updated_diagnosis") or ""
    confidence_delta = _safe_float(evidence_update.get("confidence_delta"), 0.0) or 0.0
    evidence_summary = str(evidence_update.get("evidence_summary") or "").strip()

    if updated_cause:
        new_confidence = _clip(diagnosis.confidence + confidence_delta, 0.0, 0.96)
        if diagnosis.likely_cause == "telemetry_fault_or_insufficient_evidence":
            new_confidence = max(new_confidence, 0.72)
        diagnosis = Diagnosis(
            likely_cause=updated_cause,
            confidence=new_confidence,
            reasons=diagnosis.reasons + [f"Human evidence update: {evidence_summary}"],
            request_human_evidence=False,
            evidence_summary=evidence_summary,
        )
    elif evidence_summary:
        diagnosis.evidence_summary = evidence_summary

    override = str(evidence_update.get("recommendation_override") or "").strip()
    if override:
        override_candidate = ActionCandidate(
            "human_evidence_override",
            override,
            recommendation.primary_action.score + 5.0,
            "Updated by multimodal evidence.",
            "The human-evidence analysis changed the recommended action.",
            "medium",
        )
        recommendation = Recommendation(override_candidate, [override_candidate] + recommendation.candidates,
                                        recommendation.secondary_actions)
    return diagnosis, recommendation


def build_engineering_report(
    telemetry: RackTelemetry,
    forecast: ForecastSummary,
    diagnosis: Diagnosis,
    recommendation: Recommendation,
    evidence_update: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    model_report = (evidence_update or {}).get("report") if evidence_update else None
    if isinstance(model_report, dict) and model_report:
        return model_report

    crossing = "not expected inside horizon"
    if forecast.time_to_threshold_s is not None:
        crossing = f"{forecast.time_to_threshold_s / 60.0:.1f} min"

    subject = f"{telemetry.rack_id}/{telemetry.gpu_id}" if telemetry.gpu_id else telemetry.rack_id
    return {
        "incident_summary": (
            f"{subject} is {forecast.risk}: current GPU temp "
            f"{forecast.current_temp_c:.1f} C, forecast peak {forecast.peak_temp_c:.1f} C, "
            f"time to threshold {crossing}."
        ),
        "likely_cause": diagnosis.likely_cause,
        "evidence_used": diagnosis.reasons,
        "operator_next_step": recommendation.primary_action.label,
        "escalation_note": "Attach this report to the shift log if hardware or facility inspection is needed.",
    }


def evaluate_rack(
    telemetry: RackTelemetry | dict[str, Any],
    forecast: ForecastSummary,
    text_note: str = "",
    image_data_url: str = "",
    audio_data_url: str = "",
    use_crusoe: bool = True,
) -> dict[str, Any]:
    telemetry_obj = telemetry if isinstance(telemetry, RackTelemetry) else RackTelemetry.from_dict(telemetry)
    workload_class = classify_workload(telemetry_obj)
    diagnosis = diagnose(telemetry_obj, forecast)
    recommendation = recommend_actions(telemetry_obj, forecast, diagnosis, workload_class)

    evidence_update: dict[str, Any] = {}
    crusoe_status = {"model": CRUSOE_MODEL, "used": False, "error": ""}
    has_human_evidence = bool(text_note or image_data_url or audio_data_url)
    should_review_evidence = has_human_evidence and (
        diagnosis.request_human_evidence or diagnosis.confidence < 0.70 or bool(image_data_url or audio_data_url)
    )

    if should_review_evidence:
        prompt = _build_evidence_prompt(telemetry_obj, forecast, diagnosis, recommendation, workload_class)
        client = CrusoeEvidenceClient()
        if use_crusoe and client.available:
            try:
                evidence_update = client.analyze(prompt, text_note, image_data_url, audio_data_url)
                crusoe_status["used"] = True
            except Exception as exc:  # Keep the demo usable if the API is unavailable.
                crusoe_status["error"] = str(exc)
                evidence_update = _local_evidence_update(text_note, bool(image_data_url), bool(audio_data_url))
        else:
            if use_crusoe and not client.available:
                crusoe_status["error"] = "CRUSOE_API_KEY is not set; used local fallback."
            evidence_update = _local_evidence_update(text_note, bool(image_data_url), bool(audio_data_url))

        diagnosis, recommendation = _apply_evidence_update(diagnosis, recommendation, evidence_update)
        recommendation = recommend_actions(telemetry_obj, forecast, diagnosis, workload_class)

    report = build_engineering_report(telemetry_obj, forecast, diagnosis, recommendation, evidence_update)
    mode = "human_evidence" if diagnosis.request_human_evidence or should_review_evidence else "predictive_diagnosis"

    return {
        "case_id": str(uuid.uuid4()),
        "mode": mode,
        "telemetry": asdict(telemetry_obj),
        "forecast": asdict(forecast),
        "workload_class": workload_class,
        "diagnosis": asdict(diagnosis),
        "recommendation": {
            "primary_action": asdict(recommendation.primary_action),
            "candidates": [asdict(candidate) for candidate in recommendation.candidates],
            "secondary_actions": recommendation.secondary_actions,
        },
        "report": report,
        "crusoe": crusoe_status,
    }


def simulate_mitigation(result: dict[str, Any], action_id: Optional[str] = None) -> dict[str, Any]:
    forecast = result["forecast"]
    action = action_id or result["recommendation"]["primary_action"]["action_id"]
    reductions = {
        "migrate_inference_traffic": 10.0,
        "apply_power_frequency_cap": 7.0,
        "increase_cooling_request": 4.0,
        "move_batch_jobs": 3.5,
        "add_gpu_rack_capacity": 8.0,
        "escalate_technician": 0.5,
        "request_human_evidence": 0.0,
        "human_evidence_override": 6.0,
    }
    reduction = reductions.get(action, 2.0)
    current = float(forecast["current_temp_c"])
    threshold = float(forecast["threshold_c"])
    peak_before = float(forecast["peak_temp_c"])
    conv_before = float(forecast["convergence_temp_c"])
    peak_after = max(current, peak_before - reduction)
    conv_after = max(current, conv_before - reduction * 0.8)

    if peak_after < threshold:
        crossing_after = None
    else:
        before = forecast["time_to_threshold_s"] or forecast["horizon_s"]
        crossing_after = min(float(forecast["horizon_s"]), float(before) * 1.7)

    risk_after = classify_risk(current, peak_after, conv_after, crossing_after, threshold)
    return {
        "action_id": action,
        "before": {
            "risk": forecast["risk"],
            "peak_temp_c": round(peak_before, 2),
            "time_to_threshold_s": forecast["time_to_threshold_s"],
        },
        "after": {
            "risk": risk_after,
            "peak_temp_c": round(peak_after, 2),
            "convergence_temp_c": round(conv_after, 2),
            "time_to_threshold_s": crossing_after,
        },
    }


def apply_mitigation_effect(
    result: dict[str, Any],
    mitigation: dict[str, Any],
    plan: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a copy of a GPU result with its future forecast updated."""
    updated = deepcopy(result)
    forecast = dict(updated.get("forecast", {}))
    trajectory = list(forecast.get("trajectory") or [])
    before = mitigation.get("before", {})
    after = mitigation.get("after", {})
    reduction = max(
        0.0,
        float(before.get("peak_temp_c", forecast.get("peak_temp_c", 0.0)))
        - float(after.get("peak_temp_c", forecast.get("peak_temp_c", 0.0))),
    )
    traffic_percent = _safe_float((plan or {}).get("traffic_percent"), 0.0) or 0.0
    power_fraction = _clip(traffic_percent / 100.0 * 0.55, 0.08, 0.28) if traffic_percent else 0.18
    threshold = float(forecast.get("threshold_c", DEFAULT_THRESHOLD_C))
    current = float(forecast.get("current_temp_c", 0.0))

    new_points: list[dict[str, float]] = []
    crossing_after: Optional[float] = None
    for point in trajectory:
        t_s = float(point.get("t_s", 0.0))
        post = dict(point)
        if t_s > 0.0:
            post["gpu_temp_c"] = round(max(current, float(point.get("gpu_temp_c", current)) - reduction), 3)
            if "heatsink_temp_c" in post:
                post["heatsink_temp_c"] = round(
                    max(current - 1.5, float(point.get("heatsink_temp_c", current)) - reduction * 0.75),
                    3,
                )
            if "power_w" in post:
                post["power_w"] = round(max(0.0, float(point.get("power_w", 0.0)) * (1.0 - power_fraction)), 3)
        else:
            post["gpu_temp_c"] = round(current, 3)
        if crossing_after is None and float(post["gpu_temp_c"]) >= threshold:
            crossing_after = t_s
        new_points.append(post)

    if new_points:
        temps = [float(point["gpu_temp_c"]) for point in new_points]
        peak = max(temps)
        horizon = float(forecast.get("horizon_s", new_points[-1].get("t_s", 0.0)))
        tail = temps[-min(len(temps), 24):]
        convergence = sum(tail) / len(tail)
    else:
        peak = float(after.get("peak_temp_c", forecast.get("peak_temp_c", current)))
        convergence = float(after.get("convergence_temp_c", forecast.get("convergence_temp_c", peak)))
        horizon = float(forecast.get("horizon_s", 0.0))
        crossing_after = after.get("time_to_threshold_s")

    risk_after = classify_risk(current, peak, convergence, crossing_after, threshold)
    forecast.update({
        "peak_temp_c": round(peak, 3),
        "convergence_temp_c": round(convergence, 3),
        "time_to_threshold_s": crossing_after,
        "risk": risk_after,
        "confidence": round(max(0.50, float(forecast.get("confidence", 0.78)) - 0.03), 3),
        "trajectory": new_points,
        "horizon_s": horizon,
    })
    updated["forecast"] = forecast

    telemetry_data = dict(updated.get("telemetry", {}))
    if traffic_percent:
        old_load = _safe_float(telemetry_data.get("assigned_traffic_pct"), 0.0) or 0.0
        new_load = max(0.0, old_load - traffic_percent)
        old_queue = float(telemetry_data.get("inference_queue_len") or 0.0)
        load_ratio = new_load / max(old_load, 1.0)
        telemetry_data["assigned_traffic_pct"] = round(new_load, 1)
        telemetry_data["alternative_capacity_pct"] = round(max(0.0, GPU_SAFE_TRAFFIC_PCT - new_load), 1)
        telemetry_data["inference_queue_len"] = int(round(old_queue * load_ratio))
        telemetry_data["request_rate_rps"] = round((_safe_float(telemetry_data.get("request_rate_rps"), 0.0) or 0.0) * load_ratio, 2)
        telemetry_data["gpu_util_pct"] = int(round(_clip((_safe_float(telemetry_data.get("gpu_util_pct"), 0.0) or 0.0) - traffic_percent * 0.75, 0.0, 99.0)))
        telemetry_data["sm_util_pct"] = int(round(_clip((_safe_float(telemetry_data.get("sm_util_pct"), 0.0) or 0.0) - traffic_percent * 0.82, 0.0, 99.0)))
        telemetry_data["gpu_power_w"] = round(max(55.0, (_safe_float(telemetry_data.get("gpu_power_w"), 55.0) or 55.0) * (1.0 - power_fraction)), 2)
        telemetry_data["active_batches"] = max(1, int(round(float(telemetry_data.get("active_batches") or 1) * load_ratio)))
    updated["telemetry"] = telemetry_data

    telemetry = RackTelemetry.from_dict(telemetry_data)
    forecast_obj = ForecastSummary(**forecast)
    diagnosis = diagnose(telemetry, forecast_obj)
    workload_class = classify_workload(telemetry)
    recommendation = recommend_actions(telemetry, forecast_obj, diagnosis, workload_class)
    updated["workload_class"] = workload_class
    updated["diagnosis"] = asdict(diagnosis)
    updated["recommendation"] = {
        "primary_action": asdict(recommendation.primary_action),
        "candidates": [asdict(candidate) for candidate in recommendation.candidates],
        "secondary_actions": recommendation.secondary_actions,
    }
    updated["report"] = build_engineering_report(telemetry, forecast_obj, diagnosis, recommendation)
    updated["observed_table"] = telemetry_table(telemetry_data)
    updated["prediction_table"] = prediction_table(forecast_obj)
    updated["prediction_source"] = "post_migration_simulation"
    updated["mitigation_applied"] = {
        "action_id": mitigation.get("action_id"),
        "traffic_percent": traffic_percent,
        "targets": (plan or {}).get("targets", []),
        "before": before,
        "after": {
            "risk": risk_after,
            "peak_temp_c": round(peak, 2),
            "convergence_temp_c": round(convergence, 2),
            "time_to_threshold_s": crossing_after,
        },
    }
    return updated


def apply_target_migration_load(target_result: dict[str, Any], traffic_share_pct: float) -> dict[str, Any]:
    updated = deepcopy(target_result)
    telemetry_data = dict(updated.get("telemetry", {}))
    old_load = _safe_float(telemetry_data.get("assigned_traffic_pct"), 0.0) or 0.0
    new_load = min(GPU_MAX_TRAFFIC_PCT, old_load + traffic_share_pct)
    telemetry_data["assigned_traffic_pct"] = round(new_load, 1)
    telemetry_data["alternative_capacity_pct"] = round(max(0.0, GPU_SAFE_TRAFFIC_PCT - new_load), 1)
    telemetry_data["inference_queue_len"] = int(round(float(telemetry_data.get("inference_queue_len") or 0) + traffic_share_pct * 0.35))
    telemetry_data["request_rate_rps"] = round((_safe_float(telemetry_data.get("request_rate_rps"), 0.0) or 0.0) + traffic_share_pct * 0.18, 2)
    telemetry_data["gpu_util_pct"] = int(round(_clip((_safe_float(telemetry_data.get("gpu_util_pct"), 0.0) or 0.0) + traffic_share_pct * 0.72, 0.0, 99.0)))
    telemetry_data["sm_util_pct"] = int(round(_clip((_safe_float(telemetry_data.get("sm_util_pct"), 0.0) or 0.0) + traffic_share_pct * 0.78, 0.0, 99.0)))
    telemetry_data["gpu_power_w"] = round((_safe_float(telemetry_data.get("gpu_power_w"), 70.0) or 70.0) + traffic_share_pct * 2.4, 2)
    telemetry_data["active_batches"] = max(1, int(round(float(telemetry_data.get("active_batches") or 1) + traffic_share_pct / 10.0)))
    updated["telemetry"] = telemetry_data

    forecast = dict(updated.get("forecast", {}))
    trajectory = list(forecast.get("trajectory") or [])
    current = float(forecast.get("current_temp_c", telemetry_data.get("gpu_temp_c", simulator.T_AMB)))
    threshold = float(forecast.get("threshold_c", DEFAULT_THRESHOLD_C))
    uplift = traffic_share_pct * 0.14
    crossing = None
    new_points: list[dict[str, float]] = []
    for point in trajectory:
        t_s = float(point.get("t_s", 0.0))
        progress = 1.0 - math.exp(-t_s / 180.0)
        post = dict(point)
        if t_s > 0.0:
            post["gpu_temp_c"] = round(float(point.get("gpu_temp_c", current)) + uplift * progress, 3)
            if "heatsink_temp_c" in post:
                post["heatsink_temp_c"] = round(float(point.get("heatsink_temp_c", current - 3.0)) + uplift * 0.65 * progress, 3)
            if "power_w" in post:
                post["power_w"] = round(float(point.get("power_w", telemetry_data["gpu_power_w"])) + traffic_share_pct * 2.4, 3)
        if crossing is None and float(post["gpu_temp_c"]) >= threshold:
            crossing = t_s
        new_points.append(post)

    if new_points:
        temps = [float(point["gpu_temp_c"]) for point in new_points]
        peak = max(temps)
        tail = temps[-min(len(temps), 12):]
        convergence = sum(tail) / len(tail)
    else:
        peak = float(forecast.get("peak_temp_c", current))
        convergence = float(forecast.get("convergence_temp_c", peak))
    risk = classify_risk(current, peak, convergence, crossing, threshold)
    forecast.update({
        "peak_temp_c": round(peak, 3),
        "convergence_temp_c": round(convergence, 3),
        "time_to_threshold_s": crossing,
        "risk": risk,
        "trajectory": new_points,
    })
    updated["forecast"] = forecast

    telemetry = RackTelemetry.from_dict(telemetry_data)
    forecast_obj = ForecastSummary(**forecast)
    diagnosis = diagnose(telemetry, forecast_obj)
    workload_class = classify_workload(telemetry)
    recommendation = recommend_actions(telemetry, forecast_obj, diagnosis, workload_class)
    updated["workload_class"] = workload_class
    updated["diagnosis"] = asdict(diagnosis)
    updated["recommendation"] = {
        "primary_action": asdict(recommendation.primary_action),
        "candidates": [asdict(candidate) for candidate in recommendation.candidates],
        "secondary_actions": recommendation.secondary_actions,
    }
    updated["report"] = build_engineering_report(telemetry, forecast_obj, diagnosis, recommendation)
    updated["observed_table"] = telemetry_table(telemetry_data)
    updated["prediction_table"] = prediction_table(forecast_obj)
    updated["prediction_source"] = "post_migration_target_load"
    updated["received_migration"] = {"traffic_share_pct": round(traffic_share_pct, 1)}
    return updated


def _gpu_identity(result: dict[str, Any]) -> tuple[str, str]:
    telemetry = result.get("telemetry", {}) if isinstance(result, dict) else {}
    return str(telemetry.get("rack_id", "")), str(telemetry.get("gpu_id", ""))


def _iter_fleet_gpus(racks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for rack in racks or []:
        if not isinstance(rack, dict):
            continue
        for gpu in rack.get("gpus", []) or []:
            if isinstance(gpu, dict) and isinstance(gpu.get("telemetry"), dict):
                gpus.append(gpu)
    return gpus


def _find_gpu_result(racks: list[dict[str, Any]], rack_id: Any, gpu_id: Any) -> Optional[dict[str, Any]]:
    for gpu in _iter_fleet_gpus(racks):
        telemetry = gpu.get("telemetry", {})
        if telemetry.get("rack_id") == rack_id and telemetry.get("gpu_id") == gpu_id:
            return gpu
    return None


def propose_migration_plan(
    source_result: dict[str, Any],
    racks: list[dict[str, Any]],
    min_target_capacity_pct: float = 18.0,
    max_targets: int = 2,
) -> dict[str, Any]:
    """Find safe target GPUs for an operator-approved traffic migration.

    This stays deterministic on purpose: the LLM may explain the plan, but the
    control decision is bounded by telemetry, capacity, and explicit approval.
    """
    if not isinstance(source_result, dict) or not source_result.get("telemetry"):
        raise ValueError("source GPU result is required")

    source_t = source_result.get("telemetry", {})
    source_f = source_result.get("forecast", {})
    source_key = _gpu_identity(source_result)
    threshold = _safe_float(source_f.get("threshold_c"), DEFAULT_THRESHOLD_C) or DEFAULT_THRESHOLD_C
    source_load = _safe_float(source_t.get("assigned_traffic_pct"), 0.0) or 0.0
    source_queue = _safe_float(source_t.get("inference_queue_len"), 0.0) or 0.0
    source_excess = max(0.0, source_load - GPU_SAFE_TRAFFIC_PCT)

    candidates: list[dict[str, Any]] = []
    for gpu in _iter_fleet_gpus(racks):
        if _gpu_identity(gpu) == source_key:
            continue

        telemetry = gpu.get("telemetry", {})
        forecast = gpu.get("forecast", {})
        risk = str(forecast.get("risk", "")).upper()
        current_temp = _safe_float(telemetry.get("gpu_temp_c"), 999.0) or 999.0
        peak_temp = _safe_float(forecast.get("peak_temp_c"), 999.0) or 999.0
        capacity = _safe_float(telemetry.get("alternative_capacity_pct"), 0.0) or 0.0
        assigned_load = _safe_float(telemetry.get("assigned_traffic_pct"), 0.0) or 0.0
        latency = _safe_float(telemetry.get("network_latency_ms"), 999.0) or 999.0

        if risk != "SAFE":
            continue
        if capacity < min_target_capacity_pct:
            continue
        if current_temp >= threshold - 12.0 or peak_temp >= threshold - 4.0:
            continue

        different_rack_bonus = 6.0 if telemetry.get("rack_id") != source_t.get("rack_id") else 0.0
        score = capacity * 2.0 - current_temp * 0.7 - latency * 0.08 + different_rack_bonus
        candidates.append({
            "rack_id": telemetry.get("rack_id"),
            "gpu_id": telemetry.get("gpu_id"),
            "risk": risk,
            "current_temp_c": round(current_temp, 2),
            "peak_temp_c": round(peak_temp, 2),
            "assigned_traffic_pct": round(assigned_load, 1),
            "available_capacity_pct": round(capacity, 1),
            "network_latency_ms": round(latency, 1),
            "candidate_score": round(score, 2),
        })

    candidates.sort(key=lambda item: item["candidate_score"], reverse=True)
    targets = candidates[:max(1, min(max_targets, len(candidates)))]

    mitigation = simulate_mitigation(source_result, "migrate_inference_traffic")
    if not targets:
        return {
            "plan_id": str(uuid.uuid4()),
            "available": False,
            "requires_approval": True,
            "status": "no_safe_target",
            "action_id": "migrate_inference_traffic",
            "source": {
                "rack_id": source_t.get("rack_id"),
                "gpu_id": source_t.get("gpu_id"),
                "risk": source_f.get("risk"),
                "current_temp_c": source_t.get("gpu_temp_c"),
                "peak_temp_c": source_f.get("peak_temp_c"),
                "assigned_traffic_pct": source_load,
                "queue_len": source_queue,
            },
            "reason": (
                f"No SAFE target GPU has at least {min_target_capacity_pct:.0f}% available capacity "
                "while staying comfortably below the thermal threshold. Add GPU rack capacity or reduce incoming demand."
            ),
            "scale_recommendation": _scale_recommendation(source_result, racks),
            "expected": mitigation,
        }

    total_capacity = sum(float(target["available_capacity_pct"]) for target in targets)
    queue_pressure = min(14.0, source_queue / 18.0)
    desired_relief = max(12.0, source_excess + queue_pressure)
    traffic_percent = int(round(_clip(min(desired_relief, total_capacity * 0.75), 15.0, 40.0)))
    remaining = traffic_percent
    for idx, target in enumerate(targets):
        if idx == len(targets) - 1:
            target["traffic_share_pct"] = remaining
        else:
            share = int(round(traffic_percent * float(target["available_capacity_pct"]) / total_capacity))
            share = max(5, min(share, remaining - 5 * (len(targets) - idx - 1)))
            target["traffic_share_pct"] = share
            remaining -= share

    return {
        "plan_id": str(uuid.uuid4()),
        "available": True,
        "requires_approval": True,
        "status": "awaiting_engineer_approval",
        "action_id": "migrate_inference_traffic",
        "source": {
            "rack_id": source_t.get("rack_id"),
            "gpu_id": source_t.get("gpu_id"),
            "risk": source_f.get("risk"),
            "current_temp_c": source_t.get("gpu_temp_c"),
            "peak_temp_c": source_f.get("peak_temp_c"),
            "assigned_traffic_pct": source_load,
            "queue_len": source_queue,
        },
        "targets": targets,
        "traffic_percent": traffic_percent,
        "basis": {
            "source_assigned_traffic_pct": round(source_load, 1),
            "source_safe_load_pct": GPU_SAFE_TRAFFIC_PCT,
            "source_excess_pct": round(source_excess, 1),
            "source_queue_len": round(source_queue, 1),
            "target_safe_headroom_pct": round(total_capacity, 1),
            "formula": "min(source excess + queue pressure, 75% of target safe headroom), clipped to 15-40%",
        },
        "expected": mitigation,
        "guardrails": [
            "Requires explicit engineer approval before applying.",
            "Drain traffic gradually and keep live latency within the configured tolerance.",
            "Do not use targets that are already HIGH or CRITICAL risk.",
            "Rollback if target temperature or inference latency rises unexpectedly.",
            "Keep any facility inspection ticket open when the likely cause is hardware or cooling related.",
        ],
    }


def _scale_recommendation(source_result: dict[str, Any], racks: list[dict[str, Any]]) -> dict[str, Any]:
    fleet = _iter_fleet_gpus(racks)
    total_safe_headroom = sum(
        _safe_float(gpu.get("telemetry", {}).get("alternative_capacity_pct"), 0.0) or 0.0
        for gpu in fleet
    )
    source_queue = _safe_float(source_result.get("telemetry", {}).get("inference_queue_len"), 0.0) or 0.0
    source_load = _safe_float(source_result.get("telemetry", {}).get("assigned_traffic_pct"), 0.0) or 0.0
    estimated_needed_pct = max(0.0, source_load - GPU_SAFE_TRAFFIC_PCT) + min(30.0, source_queue / 8.0)
    missing_headroom = max(0.0, estimated_needed_pct - total_safe_headroom)
    additional_gpus = int(math.ceil(missing_headroom / GPU_SAFE_TRAFFIC_PCT)) if missing_headroom else 0
    return {
        "label": "Add GPU rack capacity",
        "total_safe_headroom_pct": round(total_safe_headroom, 1),
        "estimated_missing_headroom_pct": round(missing_headroom, 1),
        "additional_gpus_needed": additional_gpus,
        "additional_racks_needed": int(math.ceil(additional_gpus / GPU_RACK_SIZE)) if additional_gpus else 0,
        "reason": "Existing safe headroom is not enough to drain the queued high-compute inference load.",
    }


def execute_migration_plan(
    source_result: dict[str, Any],
    plan: dict[str, Any],
    racks: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Approve and simulate a migration plan.

    The demo never talks to real schedulers or networking controllers. A
    production integration would replace this function with an audited adapter.
    """
    if not isinstance(plan, dict) or not plan.get("available"):
        return {
            "executed": False,
            "mode": "simulation_only",
            "status": "not_executed",
            "reason": plan.get("reason", "Migration plan is not available.") if isinstance(plan, dict) else "Invalid plan.",
        }

    mitigation = simulate_mitigation(source_result, "migrate_inference_traffic")
    updated_result = apply_mitigation_effect(source_result, mitigation, plan)
    actual_after = updated_result.get("mitigation_applied", {}).get("after", mitigation["after"])
    target_updates = []
    if racks:
        for target in plan.get("targets", []) or []:
            target_result = _find_gpu_result(racks, target.get("rack_id"), target.get("gpu_id"))
            if target_result:
                target_updates.append(apply_target_migration_load(target_result, float(target.get("traffic_share_pct") or 0.0)))

    return {
        "executed": True,
        "mode": "simulation_only",
        "status": "approved_and_simulated",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan.get("plan_id"),
        "action_id": "migrate_inference_traffic",
        "source": plan.get("source", {}),
        "targets": plan.get("targets", []),
        "traffic_percent": plan.get("traffic_percent"),
        "before": mitigation["before"],
        "after": actual_after,
        "updated_result": updated_result,
        "target_updates": target_updates,
        "notes": [
            "This demo updates the selected GPU's post-migration forecast; it does not operate real infrastructure.",
            "In production, the same approval would call an audited scheduler or traffic router adapter.",
        ],
    }


def build_demo_case(name: str = "cooling_degradation") -> tuple[RackTelemetry, ForecastSummary]:
    if name == "normal":
        t, traj = simulator.simulate(profile="burst", duration=600.0, dt=0.5, incident_time=None)
        obs_time = 180.0
        horizon_s = 420.0
    else:
        t, traj = simulator.simulate(
            profile="ramp",
            duration=1500.0,
            dt=0.5,
            incident_time=120.0,
            incident_duration=None,
        )
        obs_time = 180.0
        horizon_s = 600.0

    dt = float(t[1] - t[0])
    obs_idx = int(round(obs_time / dt))
    horizon_steps = int(round(horizon_s / dt))
    future = traj[obs_idx:obs_idx + horizon_steps + 1]
    forecast = forecast_from_trajectory(future, dt=dt)

    prev_idx = max(0, obs_idx - int(round(60.0 / dt)))
    trend_c_per_min = float(traj[obs_idx, 1] - traj[prev_idx, 1])
    base = RackTelemetry(
        rack_id="rack-4",
        gpu_temp_c=round(float(traj[obs_idx, 1]), 2),
        heatsink_temp_c=round(float(traj[obs_idx, 2]), 2),
        ambient_temp_c=float(simulator.T_AMB),
        coolant_temp_c=24.2,
        fan_speed_rpm=14200,
        fan_command_pct=76,
        cooling_flow_lpm=0.42,
        cooling_command_pct=96,
        gpu_power_w=round(float(traj[obs_idx, 0]), 2),
        rack_power_kw=42.0,
        psu_voltage_v=12.05,
        psu_voltage_std_v=0.04,
        power_spike_ratio=1.34,
        workload_type="high-compute inference",
        gpu_util_pct=94,
        sm_util_pct=91,
        memory_util_pct=46,
        inference_queue_len=34,
        network_latency_ms=18,
        packet_loss_pct=0.1,
        frequency_mhz=1410,
        sensor_age_s=8,
        alternative_capacity_pct=28,
        latency_tolerance_ms=35,
        batch_jobs_present=False,
        missing_fields=[],
    )
    # Store trend as an extra dynamic attribute for frontends that want it.
    base_dict = asdict(base)
    base_dict["trend_c_per_min"] = round(trend_c_per_min, 2)
    base = RackTelemetry.from_dict(base_dict)

    if name == "telemetry_conflict":
        conflict = asdict(base)
        conflict.update({
            "gpu_temp_c": 83.0,
            "heatsink_temp_c": 25.0,
            "cooling_flow_lpm": None,
            "fan_speed_rpm": None,
            "psu_voltage_v": None,
            "sensor_age_s": 185,
            "missing_fields": ["cooling_flow_lpm", "fan_speed_rpm", "psu_voltage_v"],
        })
        return RackTelemetry.from_dict(conflict), forecast

    if name == "normal":
        normal = asdict(base)
        normal.update({
            "rack_id": "rack-2",
            "cooling_flow_lpm": 1.28,
            "cooling_command_pct": 42,
            "fan_command_pct": 45,
            "gpu_util_pct": 31,
            "sm_util_pct": 28,
            "memory_util_pct": 34,
            "power_spike_ratio": 1.05,
            "alternative_capacity_pct": 8,
        })
        return RackTelemetry.from_dict(normal), forecast

    return base, forecast


def build_simulated_fleet(count: int = 12, seed: Optional[int] = None) -> list[dict[str, Any]]:
    """Create a flat GPU snapshot for compatibility with the first demo UI."""
    racks = build_gpu_fleet(rack_count=max(1, math.ceil(count / 8)), gpus_per_rack=8, seed=seed)
    return [gpu for rack in racks for gpu in rack["gpus"]][:count]


def build_inference_fleet(
    rack_count: int = 3,
    gpus_per_rack: int = 8,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Create a workload-first fleet snapshot for the demo UI."""
    rng = random.Random(seed)
    workload = _simulate_inference_workload(rack_count, gpus_per_rack, rng)
    schedule = _schedule_inference_workload(workload, rack_count, gpus_per_rack, rng)
    cases: list[dict[str, Any]] = []
    racks = []

    for rack_idx in range(1, rack_count + 1):
        for gpu_idx in range(1, gpus_per_rack + 1):
            state = schedule[(rack_idx, gpu_idx)]
            telemetry, forecast, case_type = _build_workload_driven_case(
                rack_id=f"rack-{rack_idx}",
                gpu_id=f"gpu-{gpu_idx}",
                workload=workload,
                state=state,
                rng=rng,
            )
            cases.append({
                "rack_idx": rack_idx,
                "gpu_idx": gpu_idx,
                "state": state,
                "telemetry": telemetry,
                "fallback_forecast": forecast,
                "case_type": case_type,
                "context": {
                    "workload": workload,
                    "scheduler_state": state,
                    "extra_racks_needed": workload.get("extra_racks_needed", 0),
                },
            })

    bloc_a_forecasts = forecasts_many_with_bloc_a([
        {
            "telemetry": case["telemetry"],
            "context": case["context"],
        }
        for case in cases
    ])

    cases_by_rack: dict[int, list[dict[str, Any]]] = {}
    for case, bloc_a_forecast in zip(cases, bloc_a_forecasts):
        telemetry = case["telemetry"]
        forecast = bloc_a_forecast or case["fallback_forecast"]
        result = evaluate_rack(telemetry, forecast, use_crusoe=False)
        result["simulated_case"] = case["case_type"]
        result["prediction_source"] = getattr(forecast, "prediction_source", "workload_driven_thermal_simulation")
        if hasattr(forecast, "bloc_a"):
            result["bloc_a"] = getattr(forecast, "bloc_a")
        result["workload_request"] = workload
        result["scheduler_state"] = case["state"]
        result["observed_table"] = telemetry_table(asdict(telemetry))
        result["prediction_table"] = prediction_table(forecast)
        cases_by_rack.setdefault(case["rack_idx"], []).append(result)

    for rack_idx in range(1, rack_count + 1):
        gpus = cases_by_rack.get(rack_idx, [])
        if not gpus:
            continue
        rack = _aggregate_gpu_rack(f"rack-{rack_idx}", gpus)
        rack["scheduler_policy"] = workload["scheduler_policy"]
        racks.append(rack)

    return {"racks": racks, "workload": workload}


def build_gpu_fleet(rack_count: int = 3, gpus_per_rack: int = 8, seed: Optional[int] = None) -> list[dict[str, Any]]:
    """Create a 3-rack x 8-GPU snapshot for compatibility callers."""
    return build_inference_fleet(rack_count=rack_count, gpus_per_rack=gpus_per_rack, seed=seed)["racks"]


def _simulate_inference_workload(rack_count: int, gpus_per_rack: int, rng: random.Random) -> dict[str, Any]:
    safe_capacity = rack_count * gpus_per_rack * GPU_SAFE_TRAFFIC_PCT
    request_type = rng.choice([
        "prompt_burst",
        "queued_high_compute_inference",
        "batch_inference_spike",
        "mixed_prompt_and_batch",
    ])
    demand_factor = {
        "prompt_burst": rng.uniform(0.78, 1.08),
        "queued_high_compute_inference": rng.uniform(0.96, 1.34),
        "batch_inference_spike": rng.uniform(0.72, 1.18),
        "mixed_prompt_and_batch": rng.uniform(0.84, 1.26),
    }[request_type]
    demand_units = round(safe_capacity * demand_factor, 1)
    prompt_jobs = rng.randint(180, 900)
    batch_jobs = rng.randint(8, 70) if "batch" in request_type or request_type == "mixed_prompt_and_batch" else rng.randint(0, 18)
    queued_jobs = int(max(20, demand_units - safe_capacity) * rng.uniform(0.18, 0.42) + rng.randint(25, 180))
    avg_prompt_tokens = rng.choice([512, 768, 1024, 1536, 2048, 3072])
    avg_output_tokens = rng.choice([128, 256, 384, 512, 768])
    request_rate_rps = round(demand_units / rng.uniform(32.0, 58.0), 1)
    required_safe_racks = int(math.ceil(demand_units / max(1.0, gpus_per_rack * GPU_SAFE_TRAFFIC_PCT)))

    return {
        "workload_id": f"req-{uuid.uuid4().hex[:8]}",
        "request_type": request_type,
        "scheduler_policy": "rack-first packing with spillover",
        "demand_units": demand_units,
        "safe_capacity_units": round(safe_capacity, 1),
        "required_safe_racks": required_safe_racks,
        "extra_racks_needed": max(0, required_safe_racks - rack_count),
        "prompt_jobs": prompt_jobs,
        "batch_jobs": batch_jobs,
        "queued_jobs": queued_jobs,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_output_tokens": avg_output_tokens,
        "request_rate_rps": request_rate_rps,
        "packing_limit_pct": round(rng.uniform(86.0, 94.0), 1),
        "safe_gpu_load_pct": GPU_SAFE_TRAFFIC_PCT,
    }


def _schedule_inference_workload(
    workload: dict[str, Any],
    rack_count: int,
    gpus_per_rack: int,
    rng: random.Random,
) -> dict[tuple[int, int], dict[str, Any]]:
    remaining = float(workload["demand_units"])
    packing_limit = float(workload["packing_limit_pct"])
    states: dict[tuple[int, int], dict[str, Any]] = {}

    for rack_idx in range(1, rack_count + 1):
        for gpu_idx in range(1, gpus_per_rack + 1):
            if remaining > 0:
                fill_limit = _clip(packing_limit + rng.uniform(-5.5, 4.5), 68.0, GPU_MAX_TRAFFIC_PCT)
                assigned = min(fill_limit, remaining)
                remaining -= assigned
            else:
                assigned = rng.uniform(4.0, 18.0)

            states[(rack_idx, gpu_idx)] = {
                "rack_id": f"rack-{rack_idx}",
                "gpu_id": f"gpu-{gpu_idx}",
                "assigned_traffic_pct": round(assigned, 1),
                "safe_headroom_pct": round(max(0.0, GPU_SAFE_TRAFFIC_PCT - assigned), 1),
                "scheduler_order": (rack_idx - 1) * gpus_per_rack + gpu_idx,
                "spillover": rack_idx > 1,
            }

    pressure_sum = sum(
        max(0.0, state["assigned_traffic_pct"] - GPU_SAFE_TRAFFIC_PCT)
        for state in states.values()
    )
    total_queue = int(workload["queued_jobs"])
    for state in states.values():
        overload = max(0.0, state["assigned_traffic_pct"] - GPU_SAFE_TRAFFIC_PCT)
        if pressure_sum > 0:
            queue = int(round(total_queue * overload / pressure_sum)) + rng.randint(0, 8)
        else:
            queue = rng.randint(2, 22)
        state["queue_len"] = max(0, queue)
        state["load_state"] = (
            "overloaded" if state["assigned_traffic_pct"] >= 86.0
            else "busy" if state["assigned_traffic_pct"] >= GPU_SAFE_TRAFFIC_PCT
            else "available"
        )
    return states


def _build_workload_driven_case(
    rack_id: str,
    gpu_id: str,
    workload: dict[str, Any],
    state: dict[str, Any],
    rng: random.Random,
) -> tuple[RackTelemetry, ForecastSummary, str]:
    load = float(state["assigned_traffic_pct"])
    queue = int(state["queue_len"])
    request_rate = float(workload["request_rate_rps"]) * max(0.03, load / max(1.0, float(workload["demand_units"]))) * GPU_RACK_SIZE
    ambient = float(simulator.T_AMB) + rng.uniform(-0.8, 1.8)
    coolant = 23.0 + rng.uniform(-1.0, 2.4)
    current_temp = ambient + 3.2 + load * 0.34 + min(8.0, queue * 0.025) + rng.uniform(-1.4, 1.6)
    current_temp = round(_clip(current_temp, ambient + 2.0, DEFAULT_THRESHOLD_C - 1.0), 2)
    power = round(_clip(55.0 + load * 3.35 + queue * 0.18 + rng.uniform(-9.0, 12.0), 55.0, 420.0), 2)
    fan_command = int(_clip(35.0 + load * 0.58 + queue * 0.05, 35.0, 100.0))
    cooling_command = int(_clip(38.0 + load * 0.55 + queue * 0.04, 35.0, 100.0))
    sm_util = int(_clip(load + rng.uniform(2.0, 12.0), 10.0, 99.0))
    memory_util = int(_clip(32.0 + load * rng.uniform(0.28, 0.48), 18.0, 96.0))
    gpu_util = int(_clip(load + rng.uniform(0.0, 8.0), 8.0, 99.0))
    network_latency = round(12.0 + queue * 0.42 + max(0.0, load - GPU_SAFE_TRAFFIC_PCT) * 0.65 + rng.uniform(-3.0, 8.0), 1)

    telemetry = RackTelemetry(
        rack_id=rack_id,
        gpu_id=gpu_id,
        gpu_temp_c=current_temp,
        heatsink_temp_c=round(current_temp - rng.uniform(2.0, 4.8), 2),
        ambient_temp_c=round(ambient, 2),
        coolant_temp_c=round(coolant, 2),
        fan_speed_rpm=int(_clip(fan_command * 165 + rng.randint(-650, 850), 3200, 16500)),
        fan_command_pct=fan_command,
        cooling_flow_lpm=round(_clip(0.72 + cooling_command / 100.0 * 0.72 + rng.uniform(-0.08, 0.08), 0.52, 1.55), 2),
        cooling_command_pct=cooling_command,
        gpu_power_w=power,
        rack_power_kw=round(5.2 + power / 1000.0 * GPU_RACK_SIZE, 2),
        psu_voltage_v=round(12.05 + rng.uniform(-0.12, 0.10), 2),
        psu_voltage_std_v=round(rng.uniform(0.03, 0.10), 2),
        power_spike_ratio=round(1.06 + max(0.0, load - 50.0) / 125.0 + queue / 900.0, 2),
        workload_type=workload["request_type"].replace("_", " "),
        gpu_util_pct=gpu_util,
        sm_util_pct=sm_util,
        memory_util_pct=memory_util,
        inference_queue_len=queue,
        request_rate_rps=round(request_rate, 2),
        active_batches=max(1, int(round(load / 9.0 + queue / 45.0))),
        assigned_traffic_pct=round(load, 1),
        workload_demand_units=float(workload["demand_units"]),
        prompt_tokens=int(workload["avg_prompt_tokens"]),
        output_tokens=int(workload["avg_output_tokens"]),
        network_latency_ms=network_latency,
        packet_loss_pct=round(_clip(queue / 280.0 + rng.uniform(0.0, 0.18), 0.0, 3.0), 2),
        frequency_mhz=int(_clip(1530 - max(0.0, current_temp - 58.0) * 10.0 + rng.randint(-35, 45), 1040, 1585)),
        sensor_age_s=rng.randint(3, 22),
        alternative_capacity_pct=round(max(0.0, GPU_SAFE_TRAFFIC_PCT - load), 1),
        latency_tolerance_ms=rng.choice([20, 35, 50]),
        batch_jobs_present=workload["batch_jobs"] > 0,
        missing_fields=[],
    )
    forecast = _forecast_from_workload(telemetry, workload, state, rng)
    case_type = "workload_queue_pressure" if queue >= 60 else "workload_heat" if load >= GPU_SAFE_TRAFFIC_PCT else "available_capacity"
    return telemetry, forecast, case_type


def _forecast_from_workload(
    telemetry: RackTelemetry,
    workload: dict[str, Any],
    state: dict[str, Any],
    rng: random.Random,
) -> ForecastSummary:
    load = _safe_float(telemetry.assigned_traffic_pct, 0.0) or 0.0
    queue = float(telemetry.inference_queue_len or 0)
    current = float(telemetry.gpu_temp_c or simulator.T_AMB)
    threshold = DEFAULT_THRESHOLD_C
    horizon = rng.choice([420.0, 480.0, 540.0, 600.0])
    overload = max(0.0, load - GPU_SAFE_TRAFFIC_PCT)
    peak_delta = 2.0 + overload * 0.58 + min(13.0, queue * 0.055) + max(0.0, load - 55.0) * 0.08
    if workload["extra_racks_needed"] > 0:
        peak_delta += min(7.0, workload["extra_racks_needed"] * 3.0)
    peak = max(current + 1.0, current + peak_delta + rng.uniform(-1.0, 1.4))
    if load < 45.0 and queue < 15:
        peak = min(peak, threshold - rng.uniform(10.0, 18.0))

    points = []
    crossing = None
    base_power = float(telemetry.gpu_power_w or 80.0)
    for idx in range(0, int(horizon) + 1, 10):
        progress = 1.0 - math.exp(-idx / max(90.0, horizon * 0.34))
        temp = current + (peak - current) * progress
        temp += math.sin(idx / 55.0) * 0.35
        temp = round(max(current, temp), 3)
        power_wave = 1.0 + math.sin(idx / 47.0) * 0.035
        point = {
            "t_s": float(idx),
            "power_w": round(base_power * power_wave, 3),
            "gpu_temp_c": temp,
            "heatsink_temp_c": round(max(current - 3.0, temp - 3.2), 3),
        }
        if crossing is None and temp >= threshold:
            crossing = float(idx)
        points.append(point)

    temps = [point["gpu_temp_c"] for point in points]
    tail = temps[-min(len(temps), 12):]
    convergence = sum(tail) / len(tail)
    risk = classify_risk(current, max(temps), convergence, crossing, threshold)
    confidence = 0.78 + min(0.14, abs(load - GPU_SAFE_TRAFFIC_PCT) / 120.0) + rng.random() * 0.04
    return ForecastSummary(
        threshold_c=threshold,
        horizon_s=horizon,
        current_temp_c=current,
        peak_temp_c=max(temps),
        convergence_temp_c=convergence,
        time_to_threshold_s=crossing,
        risk=risk,
        confidence=round(_clip(confidence, 0.70, 0.94), 3),
        trajectory=points,
    )


def _legacy_random_gpu_fleet(rack_count: int, gpus_per_rack: int, seed: Optional[int] = None) -> list[dict[str, Any]]:
    """Older incident-mix simulator kept as a local fallback/reference."""
    rng = random.Random(seed)
    incident_mix = [
        "normal",
        "normal",
        "workload_heat",
        "cooling_degradation",
        "fan_failure",
        "power_supply",
        "network_queue",
        "telemetry_conflict",
    ]
    racks = []
    for rack_idx in range(1, rack_count + 1):
        gpus = []
        for gpu_idx in range(1, gpus_per_rack + 1):
            if rack_idx == 1 and gpu_idx == 1:
                case_type = "cooling_degradation"
            else:
                case_type = rng.choice(incident_mix)
            telemetry, forecast = _build_randomized_case(
                rack_id=f"rack-{rack_idx}",
                gpu_id=f"gpu-{gpu_idx}",
                case_type=case_type,
                rng=rng,
            )
            result = evaluate_rack(telemetry, forecast, use_crusoe=False)
            result["simulated_case"] = case_type
            result["prediction_source"] = "simulator_forecast_placeholder"
            result["observed_table"] = telemetry_table(asdict(telemetry))
            result["prediction_table"] = prediction_table(forecast)
            gpus.append(result)
        racks.append(_aggregate_gpu_rack(f"rack-{rack_idx}", gpus))

    return racks


def telemetry_table(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("GPU", telemetry.get("gpu_id"), ""),
        ("Workload type", telemetry.get("workload_type"), ""),
        ("Assigned inference traffic", telemetry.get("assigned_traffic_pct"), "%"),
        ("Safe traffic headroom", telemetry.get("alternative_capacity_pct"), "%"),
        ("Request rate", telemetry.get("request_rate_rps"), "req/s"),
        ("Active batches", telemetry.get("active_batches"), "batches"),
        ("GPU temp", telemetry.get("gpu_temp_c"), "C"),
        ("Ambient temp", telemetry.get("ambient_temp_c"), "C"),
        ("Coolant temp", telemetry.get("coolant_temp_c"), "C"),
        ("GPU power", telemetry.get("gpu_power_w"), "W"),
        ("Fan command", telemetry.get("fan_command_pct"), "%"),
        ("Fan speed", telemetry.get("fan_speed_rpm"), "RPM"),
        ("Cooling command", telemetry.get("cooling_command_pct"), "%"),
        ("Cooling flow", telemetry.get("cooling_flow_lpm"), "L/min"),
        ("PSU voltage", telemetry.get("psu_voltage_v"), "V"),
        ("Queue length", telemetry.get("inference_queue_len"), "jobs"),
        ("Network latency", telemetry.get("network_latency_ms"), "ms"),
        ("SM utilization", telemetry.get("sm_util_pct"), "%"),
        ("Memory utilization", telemetry.get("memory_util_pct"), "%"),
        ("Prompt tokens", telemetry.get("prompt_tokens"), "avg"),
        ("Output tokens", telemetry.get("output_tokens"), "avg"),
    ]
    return [
        {"parameter": name, "value": _display_value(value), "unit": unit}
        for name, value, unit in rows
    ]


def prediction_table(forecast: ForecastSummary) -> list[dict[str, Any]]:
    points = forecast.trajectory or []
    if not points:
        return []
    targets = [0, 60, 120, 180, 300, int(forecast.horizon_s)]
    table = []
    for target in targets:
        nearest = min(points, key=lambda point: abs(point["t_s"] - target))
        if table and table[-1]["t_s"] == nearest["t_s"]:
            continue
        table.append({
            "t_s": nearest["t_s"],
            "gpu_temp_c": nearest["gpu_temp_c"],
            "power_w": nearest["power_w"],
        })
    return table


def _build_randomized_case(
    rack_id: str,
    case_type: str,
    rng: random.Random,
    gpu_id: str = "gpu-1",
) -> tuple[RackTelemetry, ForecastSummary]:
    telemetry, forecast = _simulate_varied_case(rack_id, gpu_id, case_type, rng)
    data = asdict(telemetry)
    data["rack_id"] = rack_id
    data["gpu_id"] = gpu_id

    # Small per-rack variation without changing the diagnosis class.
    for key, spread in {
        "gpu_temp_c": 1.8,
        "heatsink_temp_c": 1.4,
        "gpu_power_w": 5.0,
        "rack_power_kw": 2.5,
        "coolant_temp_c": 1.0,
    }.items():
        if _is_number(data.get(key)):
            data[key] = round(float(data[key]) + rng.uniform(-spread, spread), 2)

    data["alternative_capacity_pct"] = rng.randint(8, 35)
    data["latency_tolerance_ms"] = rng.choice([10, 20, 35, 50])
    data["batch_jobs_present"] = rng.random() < 0.35

    if case_type == "normal":
        data.update({
            "cooling_flow_lpm": round(rng.uniform(1.05, 1.55), 2),
            "cooling_command_pct": rng.randint(35, 58),
            "fan_speed_rpm": rng.randint(8200, 11800),
            "fan_command_pct": rng.randint(35, 58),
            "psu_voltage_v": round(rng.uniform(11.95, 12.15), 2),
            "psu_voltage_std_v": round(rng.uniform(0.02, 0.07), 2),
            "power_spike_ratio": round(rng.uniform(1.0, 1.12), 2),
            "workload_type": rng.choice(["light inference", "batch idle", "steady inference"]),
            "gpu_util_pct": rng.randint(18, 45),
            "sm_util_pct": rng.randint(18, 48),
            "memory_util_pct": rng.randint(20, 52),
            "inference_queue_len": rng.randint(4, 22),
            "network_latency_ms": rng.randint(8, 28),
            "packet_loss_pct": round(rng.uniform(0.0, 0.2), 2),
        })
    elif case_type == "workload_heat":
        data.update({
            "cooling_flow_lpm": round(rng.uniform(1.05, 1.4), 2),
            "cooling_command_pct": rng.randint(55, 76),
            "fan_speed_rpm": rng.randint(11200, 15600),
            "fan_command_pct": rng.randint(62, 82),
            "psu_voltage_v": round(rng.uniform(11.95, 12.18), 2),
            "psu_voltage_std_v": round(rng.uniform(0.03, 0.08), 2),
            "power_spike_ratio": round(rng.uniform(1.28, 1.48), 2),
            "workload_type": "high-compute inference",
            "gpu_util_pct": rng.randint(86, 98),
            "sm_util_pct": rng.randint(82, 97),
            "memory_util_pct": rng.randint(38, 62),
            "inference_queue_len": rng.randint(18, 48),
            "network_latency_ms": rng.randint(12, 34),
        })
    elif case_type == "fan_failure":
        data.update({
            "cooling_flow_lpm": round(rng.uniform(1.05, 1.35), 2),
            "cooling_command_pct": rng.randint(70, 96),
            "fan_speed_rpm": rng.randint(900, 2900),
            "fan_command_pct": rng.randint(82, 100),
            "workload_type": "inference",
            "gpu_util_pct": rng.randint(72, 92),
            "sm_util_pct": rng.randint(70, 92),
            "memory_util_pct": rng.randint(38, 76),
        })
    elif case_type == "power_supply":
        data.update({
            "cooling_flow_lpm": round(rng.uniform(1.05, 1.35), 2),
            "cooling_command_pct": rng.randint(56, 82),
            "fan_speed_rpm": rng.randint(10000, 14800),
            "fan_command_pct": rng.randint(58, 84),
            "psu_voltage_v": round(rng.choice([11.25, 11.38, 12.72, 12.85]) + rng.uniform(-0.04, 0.04), 2),
            "psu_voltage_std_v": round(rng.uniform(0.28, 0.48), 2),
            "workload_type": "inference",
            "gpu_util_pct": rng.randint(66, 88),
            "sm_util_pct": rng.randint(62, 85),
        })
    elif case_type == "network_queue":
        data.update({
            "cooling_flow_lpm": round(rng.uniform(1.05, 1.35), 2),
            "cooling_command_pct": rng.randint(54, 78),
            "fan_speed_rpm": rng.randint(9800, 14600),
            "fan_command_pct": rng.randint(55, 82),
            "network_latency_ms": rng.randint(95, 180),
            "packet_loss_pct": round(rng.uniform(0.6, 2.8), 2),
            "inference_queue_len": rng.randint(65, 180),
            "workload_type": "queued inference",
            "gpu_util_pct": rng.randint(70, 94),
            "sm_util_pct": rng.randint(66, 90),
        })
    elif case_type == "telemetry_conflict":
        data["missing_fields"] = ["cooling_flow_lpm", "fan_speed_rpm", "psu_voltage_v"]
    else:
        data.update({
            "cooling_flow_lpm": round(rng.uniform(0.28, 0.62), 2),
            "cooling_command_pct": rng.randint(86, 100),
            "fan_speed_rpm": rng.randint(11800, 16200),
            "fan_command_pct": rng.randint(66, 92),
        })

    return RackTelemetry.from_dict(data), forecast


def _simulate_varied_case(
    rack_id: str,
    gpu_id: str,
    case_type: str,
    rng: random.Random,
) -> tuple[RackTelemetry, ForecastSummary]:
    dt = 0.5
    duration = rng.choice([900.0, 1050.0, 1200.0, 1350.0, 1500.0])
    horizon_s = rng.choice([420.0, 480.0, 540.0, 600.0, 660.0])
    obs_time = rng.uniform(150.0, min(330.0, duration - horizon_s - 20.0))

    if case_type == "normal":
        profile = rng.choice(["idle", "burst"])
        incident_time = None
    elif case_type == "workload_heat":
        profile = rng.choice(["ramp", "sustained"])
        incident_time = None
    else:
        profile = rng.choice(["ramp", "sustained"])
        incident_time = rng.uniform(70.0, max(72.0, obs_time - 20.0))

    state0 = simulator.np.array([
        simulator.P_IDLE + rng.uniform(-3.0, 4.0),
        simulator.T_AMB + rng.uniform(1.5, 6.0),
        simulator.T_AMB + rng.uniform(0.5, 3.0),
    ])
    t, traj = simulator.simulate(
        profile=profile,
        duration=duration,
        dt=dt,
        incident_time=incident_time,
        incident_duration=None,
        state0=state0,
    )

    obs_idx = int(round(obs_time / dt))
    horizon_steps = int(round(min(horizon_s, duration - obs_time) / dt))
    future = traj[obs_idx:obs_idx + horizon_steps + 1]
    forecast = forecast_from_trajectory(future, dt=dt, confidence=0.78 + rng.random() * 0.14)

    telemetry = RackTelemetry(
        rack_id=rack_id,
        gpu_id=gpu_id,
        gpu_temp_c=round(float(traj[obs_idx, 1]), 2),
        heatsink_temp_c=round(float(traj[obs_idx, 2]), 2),
        ambient_temp_c=float(simulator.T_AMB),
        coolant_temp_c=round(23.0 + rng.uniform(-1.0, 2.5), 2),
        fan_speed_rpm=rng.randint(9500, 15000),
        fan_command_pct=rng.randint(52, 85),
        cooling_flow_lpm=round(rng.uniform(0.9, 1.45), 2),
        cooling_command_pct=rng.randint(50, 82),
        gpu_power_w=round(float(traj[obs_idx, 0]), 2),
        rack_power_kw=round(rng.uniform(35.0, 50.0), 2),
        psu_voltage_v=round(rng.uniform(11.9, 12.2), 2),
        psu_voltage_std_v=round(rng.uniform(0.02, 0.08), 2),
        power_spike_ratio=round(rng.uniform(1.08, 1.32), 2),
        workload_type="inference",
        gpu_util_pct=rng.randint(55, 92),
        sm_util_pct=rng.randint(52, 91),
        memory_util_pct=rng.randint(35, 78),
        inference_queue_len=rng.randint(10, 55),
        network_latency_ms=rng.randint(12, 45),
        packet_loss_pct=round(rng.uniform(0.0, 0.4), 2),
        frequency_mhz=rng.randint(1180, 1530),
        sensor_age_s=rng.randint(4, 35),
        alternative_capacity_pct=rng.randint(8, 35),
        latency_tolerance_ms=rng.choice([10, 20, 35, 50]),
        batch_jobs_present=rng.random() < 0.35,
        missing_fields=[],
    )
    return telemetry, forecast


def _aggregate_gpu_rack(rack_id: str, gpus: list[dict[str, Any]]) -> dict[str, Any]:
    risks = [gpu["forecast"]["risk"] for gpu in gpus]
    temps = [float(gpu["telemetry"]["gpu_temp_c"]) for gpu in gpus if _is_number(gpu["telemetry"].get("gpu_temp_c"))]
    loads = [float(gpu["telemetry"].get("assigned_traffic_pct") or 0.0) for gpu in gpus]
    headrooms = [float(gpu["telemetry"].get("alternative_capacity_pct") or 0.0) for gpu in gpus]
    queues = [int(gpu["telemetry"].get("inference_queue_len") or 0) for gpu in gpus]
    rack_temp_c = round(sum(temps) / len(temps), 2) if temps else None
    if rack_temp_c is not None and rack_temp_c > 80.0:
        rack_temp_state = "CRITICAL"
    elif rack_temp_c is not None and rack_temp_c >= 50.0:
        rack_temp_state = "MEDIUM"
    else:
        rack_temp_state = "SAFE"
    critical_count = sum(1 for risk in risks if risk == "CRITICAL")
    high_count = sum(1 for risk in risks if risk == "HIGH")
    watch_count = sum(1 for risk in risks if risk == "WATCH")
    half_gpu_count = len(gpus) / 2
    if critical_count > half_gpu_count:
        rack_state = "CRITICAL"
    elif critical_count == half_gpu_count:
        rack_state = "MEDIUM"
    else:
        rack_state = "SAFE"

    top_gpu = max(gpus, key=lambda gpu: (
        {"CRITICAL": 4, "HIGH": 3, "WATCH": 2, "SAFE": 1}.get(gpu["forecast"]["risk"], 0),
        gpu["forecast"]["peak_temp_c"],
    ))
    return {
        "rack_id": rack_id,
        "rack_state": rack_state,
        "gpu_count": len(gpus),
        "safe_count": sum(1 for risk in risks if risk == "SAFE"),
        "watch_count": watch_count,
        "high_count": high_count,
        "critical_count": critical_count,
        "rack_temp_c": rack_temp_c,
        "rack_temp_state": rack_temp_state,
        "max_gpu_temp_c": round(max(temps), 2) if temps else None,
        "avg_gpu_temp_c": rack_temp_c,
        "avg_assigned_traffic_pct": round(sum(loads) / len(loads), 1) if loads else 0.0,
        "safe_headroom_pct": round(sum(headrooms), 1),
        "queued_jobs": sum(queues),
        "top_gpu_id": top_gpu["telemetry"].get("gpu_id"),
        "top_risk": top_gpu["forecast"]["risk"],
        "dominant_cause": top_gpu["diagnosis"]["likely_cause"],
        "gpus": gpus,
    }


def _display_value(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


if __name__ == "__main__":
    telemetry, forecast = build_demo_case("cooling_degradation")
    print(json.dumps(evaluate_rack(telemetry, forecast, use_crusoe=False), indent=2))

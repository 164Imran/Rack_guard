"""Adapter for Imran's Bloc A temperature forecaster.

The current Neural ODE checkpoint predicts the thermal state
``[P, T_die, T_hs]`` from a fixed 31-point observed window. Rack Guardian's
Bloc B uses richer operational telemetry, so this module bridges the two:

* use a provided observed window when the caller has one;
* otherwise synthesize the 31-point window from current telemetry;
* keep fan/cooling/queue/workload signals as context adjustments, because the
  NODE itself does not consume those channels yet.
"""

from __future__ import annotations

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import simulator
    from .model_node import ThermalNODE
except ImportError:  # pragma: no cover - useful when run from rack_guardians/
    import simulator
    from model_node import ThermalNODE


HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
DATA_DIR = HERE / "data"

DEFAULT_OBS_DURATION_S = 150.0
DEFAULT_PRED_DURATION_S = 150.0
DEFAULT_SAMPLE_DT_S = 5.0
DEFAULT_DENSE_DT_S = 0.5
DEFAULT_L_OBS = 31
DEFAULT_PRED_LEN = 300
DEFAULT_N_SUBSTEPS = 4

MODEL_POWER_MIN_W = float(simulator.P_IDLE)
MODEL_POWER_MAX_W = float(simulator.P_BURST)
DEMO_POWER_MIN_W = 55.0
DEMO_POWER_MAX_W = 420.0


def model_available() -> bool:
    return (CKPT_DIR / "node.pt").exists()


def input_schema() -> dict[str, Any]:
    meta = _load_meta()
    return {
        "model": "ThermalNODE",
        "required_channels": ["power_w", "gpu_temp_c", "heatsink_temp_c"],
        "observed_points": meta["l_obs"],
        "observed_duration_s": meta["obs_duration"],
        "observed_sample_dt_s": meta["sample_dt"],
        "prediction_steps": meta["pred_len"],
        "prediction_dt_s": meta["dt"],
        "prediction_horizon_s": meta["pred_duration"],
        "extra_telemetry_used_as_context": [
            "assigned_traffic_pct",
            "inference_queue_len",
            "cooling_flow_lpm",
            "network_latency_ms",
            "gpu_power_w",
        ],
    }


def predict_gpu_temperature(
    observed_window: Any = None,
    telemetry: Any = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a GPU temperature trajectory with the Bloc A Neural ODE.

    ``observed_window`` may be a list of dicts with ``power_w``,
    ``gpu_temp_c`` and ``heatsink_temp_c`` keys, or a list/array shaped
    ``(n, 3)`` containing ``[P, T_die, T_hs]``. When omitted, the adapter
    synthesizes a plausible fixed-length observation window from the current
    telemetry snapshot.
    """

    predictions = predict_gpu_temperature_batch([{
        "observed_window": observed_window,
        "telemetry": telemetry,
        "context": context or {},
    }])
    return predictions[0] if predictions else {"ok": False, "error": "Bloc A NODE prediction failed"}


def predict_gpu_temperature_batch(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch version of ``predict_gpu_temperature`` for fleet simulation."""

    if not items:
        return []
    if not model_available():
        return [{"ok": False, "error": "Bloc A NODE checkpoint not found"} for _ in items]

    meta = _load_meta()
    prepared: list[dict[str, Any]] = []
    for item in items:
        telemetry = item.get("telemetry")
        observed, input_mode, scaled_power = _prepare_observed_window(
            observed_window=item.get("observed_window"),
            telemetry=telemetry,
            meta=meta,
        )
        if observed is None:
            prepared.append({"ok": False, "error": "Could not build NODE observed window"})
        else:
            prepared.append({
                "ok": True,
                "observed": observed,
                "input_mode": input_mode,
                "scaled_power": scaled_power,
                "telemetry": telemetry,
                "context": item.get("context") or {},
            })

    valid_indices = [idx for idx, item in enumerate(prepared) if item.get("ok")]
    if not valid_indices:
        return prepared

    try:
        model = _load_node_model(meta["l_obs"], meta["pred_duration"])
        obs_batch = np.stack([prepared[idx]["observed"] for idx in valid_indices], axis=0)
        t_obs = torch.linspace(0.0, meta["obs_duration"], meta["l_obs"], dtype=torch.float32).unsqueeze(0)
        t_obs = t_obs.repeat(len(valid_indices), 1)
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
        with torch.inference_mode():
            pred = model(
                t_obs,
                obs_tensor,
                pred_len=meta["pred_len"],
                dt=meta["dt"],
                obs_duration=meta["obs_duration"],
                n_substeps=meta["n_substeps"],
            ).cpu().numpy()
    except Exception as exc:
        return [{"ok": False, "error": f"Bloc A NODE inference failed: {exc}"} for _ in items]

    if not np.isfinite(pred).all():
        return [{"ok": False, "error": "Bloc A NODE produced non-finite values"} for _ in items]

    results = list(prepared)
    for pred_index, item_index in enumerate(valid_indices):
        item = prepared[item_index]
        results[item_index] = _prediction_payload(
            pred[pred_index],
            observed=item["observed"],
            telemetry=item["telemetry"],
            context=item["context"],
            input_mode=item["input_mode"],
            scaled_power=bool(item["scaled_power"]),
            meta=meta,
        )
    return results


def _prediction_payload(
    pred: np.ndarray,
    observed: np.ndarray,
    telemetry: Any,
    context: dict[str, Any],
    input_mode: str,
    scaled_power: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    current_display = _current_display_state(observed, scaled_power, telemetry)
    context_adjustment = _context_adjustment_c(telemetry, context)
    trajectory = [_trajectory_point(0.0, current_display[0], current_display[1], current_display[2])]

    for index, row in enumerate(pred):
        t_s = (index + 1) * meta["dt"]
        progress = 1.0 - math.exp(-t_s / 95.0)
        power = _from_model_power(float(row[0]), scaled_power)
        die = float(row[1]) + context_adjustment * progress
        hs = float(row[2]) + context_adjustment * progress * 0.72
        trajectory.append(_trajectory_point(t_s, power, die, hs))

    temps = [point["gpu_temp_c"] for point in trajectory]
    if min(temps) < -20.0 or max(temps) > 140.0:
        return {"ok": False, "error": "Bloc A NODE trajectory outside plausible temperature range"}

    confidence = 0.84 if input_mode == "provided_window" else 0.76
    if context_adjustment:
        confidence -= min(0.08, abs(context_adjustment) / 100.0)

    return {
        "ok": True,
        "source": "bloc_a_neural_ode",
        "input_mode": input_mode,
        "model_meta": {
            "l_obs": meta["l_obs"],
            "pred_len": meta["pred_len"],
            "dt": meta["dt"],
            "obs_duration": meta["obs_duration"],
            "n_substeps": meta["n_substeps"],
        },
        "input_schema": input_schema(),
        "power_scaled_for_node": scaled_power,
        "context_adjustment_c": round(context_adjustment, 3),
        "confidence": round(max(0.62, min(0.91, confidence)), 3),
        "horizon_s": meta["pred_duration"],
        "dt_s": meta["dt"],
        "observed_window": [
            _trajectory_point(
                idx * meta["sample_dt"],
                _from_model_power(float(row[0]), scaled_power),
                float(row[1]),
                float(row[2]),
            )
            for idx, row in enumerate(observed)
        ],
        "trajectory": trajectory,
    }


@lru_cache(maxsize=1)
def _load_meta() -> dict[str, Any]:
    meta = {
        "l_obs": DEFAULT_L_OBS,
        "pred_len": DEFAULT_PRED_LEN,
        "n_substeps": DEFAULT_N_SUBSTEPS,
        "obs_duration": DEFAULT_OBS_DURATION_S,
        "pred_duration": DEFAULT_PRED_DURATION_S,
        "sample_dt": DEFAULT_SAMPLE_DT_S,
        "dt": DEFAULT_DENSE_DT_S,
    }
    meta_path = CKPT_DIR / "meta.pkl"
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            saved = pickle.load(handle)
        meta.update({k: saved[k] for k in ("l_obs", "pred_len", "n_substeps") if k in saved})

    scenario_path = DATA_DIR / "scenarios.pkl"
    if scenario_path.exists():
        with scenario_path.open("rb") as handle:
            config = pickle.load(handle).get("config", {})
        meta["obs_duration"] = float(config.get("obs_duration", meta["obs_duration"]))
        meta["pred_duration"] = float(config.get("pred_duration", meta["pred_duration"]))
        meta["sample_dt"] = float(config.get("sample_dt", meta["sample_dt"]))
        meta["dt"] = float(config.get("dt", meta["dt"]))
    return meta


@lru_cache(maxsize=1)
def _load_node_model(l_obs: int, horizon_s: float) -> ThermalNODE:
    model = ThermalNODE(l_obs=l_obs, horizon_s=horizon_s)
    state_path = CKPT_DIR / "node.pt"
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _prepare_observed_window(
    observed_window: Any,
    telemetry: Any,
    meta: dict[str, Any],
) -> tuple[np.ndarray | None, str, bool]:
    if observed_window is not None:
        parsed = _parse_observed_window(observed_window)
        if parsed is not None:
            scaled = any(row[0] > 150.0 for row in parsed)
            parsed[:, 0] = np.array([_to_model_power(power)[0] for power in parsed[:, 0]], dtype=float)
            return _resample_window(parsed, meta["l_obs"]), "provided_window", scaled
    synthetic, scaled = _synthetic_window_from_telemetry(telemetry, meta)
    return synthetic, "synthetic_window", scaled


def _parse_observed_window(observed_window: Any) -> np.ndarray | None:
    rows: list[list[float]] = []
    for item in list(observed_window):
        if isinstance(item, dict):
            power = _first_number(item, ("power_w", "gpu_power_w", "P", "p"))
            die = _first_number(item, ("gpu_temp_c", "T_die", "t_die", "temperature"))
            hs = _first_number(item, ("heatsink_temp_c", "T_hs", "t_hs", "heatsink"))
        else:
            values = list(item)
            if len(values) >= 4:
                values = values[-3:]
            if len(values) < 3:
                continue
            power, die, hs = values[0], values[1], values[2]
        if power is None or die is None or hs is None:
            continue
        rows.append([float(power), float(die), float(hs)])
    if not rows:
        return None
    return np.asarray(rows, dtype=float)


def _resample_window(rows: np.ndarray, l_obs: int) -> np.ndarray:
    if rows.shape[0] == l_obs:
        return rows
    if rows.shape[0] == 1:
        return np.repeat(rows, l_obs, axis=0)
    x_old = np.linspace(0.0, 1.0, rows.shape[0])
    x_new = np.linspace(0.0, 1.0, l_obs)
    return np.stack([np.interp(x_new, x_old, rows[:, channel]) for channel in range(3)], axis=1)


def _synthetic_window_from_telemetry(telemetry: Any, meta: dict[str, Any]) -> tuple[np.ndarray, bool]:
    l_obs = int(meta["l_obs"])
    current_temp = _get_number(telemetry, "gpu_temp_c", float(simulator.T_AMB) + 12.0)
    current_hs = _get_number(telemetry, "heatsink_temp_c", current_temp - 3.2)
    ambient = _get_number(telemetry, "ambient_temp_c", float(simulator.T_AMB))
    load = _get_number(telemetry, "assigned_traffic_pct", _get_number(telemetry, "gpu_util_pct", 45.0))
    queue = _get_number(telemetry, "inference_queue_len", 0.0)
    cooling = _get_number(telemetry, "cooling_flow_lpm", 1.15)
    raw_power = _get_number(telemetry, "gpu_power_w", 55.0 + load * 3.2)
    current_power, scaled_power = _to_model_power(raw_power)

    thermal_rise = (
        max(1.2, (current_temp - ambient) * 0.22)
        + max(0.0, load - 55.0) * 0.055
        + min(4.5, queue * 0.018)
        + max(0.0, 1.0 - cooling) * 3.5
    )
    thermal_rise = max(1.0, min(15.0, thermal_rise))
    power_rise = max(3.0, min(26.0, max(0.0, load - 35.0) * 0.22 + queue * 0.035))
    start_temp = max(ambient + 2.0, current_temp - thermal_rise)
    start_hs = max(ambient + 1.0, current_hs - thermal_rise * 0.65)
    start_power = max(MODEL_POWER_MIN_W, current_power - power_rise)

    rows = []
    for index in range(l_obs):
        progress = index / max(1, l_obs - 1)
        ease = progress ** 1.35
        ripple = math.sin(progress * math.pi * 3.0) * 0.18
        rows.append([
            start_power + (current_power - start_power) * min(1.0, progress ** 0.82),
            start_temp + (current_temp - start_temp) * ease + ripple,
            start_hs + (current_hs - start_hs) * (progress ** 1.18) + ripple * 0.6,
        ])
    rows[-1] = [current_power, current_temp, current_hs]
    return np.asarray(rows, dtype=float), scaled_power


def _context_adjustment_c(telemetry: Any, context: dict[str, Any] | None) -> float:
    context = context or {}
    load = _get_number(telemetry, "assigned_traffic_pct", _get_number(telemetry, "gpu_util_pct", 45.0))
    queue = _get_number(telemetry, "inference_queue_len", 0.0)
    cooling = _get_number(telemetry, "cooling_flow_lpm", 1.15)
    latency = _get_number(telemetry, "network_latency_ms", 5.0)
    raw_power = _get_number(telemetry, "gpu_power_w", 120.0)
    extra_racks = float(context.get("extra_racks_needed") or 0.0)

    adjustment = 0.0
    adjustment += max(0.0, load - 72.0) * 0.085
    adjustment += min(3.2, queue * 0.025)
    adjustment += max(0.0, 0.95 - cooling) * 4.0
    adjustment += max(0.0, latency - 12.0) * 0.07
    adjustment += max(0.0, raw_power - 260.0) * 0.008
    adjustment += min(2.5, extra_racks * 1.2)
    return max(0.0, min(8.0, adjustment))


def _current_display_state(observed: np.ndarray, scaled_power: bool, telemetry: Any) -> tuple[float, float, float]:
    last = observed[-1]
    power = _get_number(telemetry, "gpu_power_w", _from_model_power(float(last[0]), scaled_power))
    die = _get_number(telemetry, "gpu_temp_c", float(last[1]))
    hs = _get_number(telemetry, "heatsink_temp_c", float(last[2]))
    return power, die, hs


def _to_model_power(power_w: float) -> tuple[float, bool]:
    if power_w <= 150.0:
        return max(MODEL_POWER_MIN_W, min(MODEL_POWER_MAX_W, float(power_w))), False
    ratio = (float(power_w) - DEMO_POWER_MIN_W) / (DEMO_POWER_MAX_W - DEMO_POWER_MIN_W)
    scaled = MODEL_POWER_MIN_W + max(0.0, min(1.0, ratio)) * (MODEL_POWER_MAX_W - MODEL_POWER_MIN_W)
    return scaled, True


def _from_model_power(model_power_w: float, scaled: bool) -> float:
    value = max(MODEL_POWER_MIN_W, min(MODEL_POWER_MAX_W, float(model_power_w)))
    if not scaled:
        return value
    ratio = (value - MODEL_POWER_MIN_W) / (MODEL_POWER_MAX_W - MODEL_POWER_MIN_W)
    return DEMO_POWER_MIN_W + ratio * (DEMO_POWER_MAX_W - DEMO_POWER_MIN_W)


def _trajectory_point(t_s: float, power_w: float, gpu_temp_c: float, heatsink_temp_c: float) -> dict[str, float]:
    return {
        "t_s": round(float(t_s), 3),
        "power_w": round(float(power_w), 3),
        "gpu_temp_c": round(float(gpu_temp_c), 3),
        "heatsink_temp_c": round(float(heatsink_temp_c), 3),
    }


def _get_number(source: Any, key: str, default: float) -> float:
    if source is None:
        return float(default)
    if isinstance(source, dict):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _first_number(source: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in source and source[key] is not None:
            try:
                number = float(source[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None

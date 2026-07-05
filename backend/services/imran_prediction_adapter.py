"""Optional adapter for Imran's physics-informed Neural ODE."""

from __future__ import annotations

import math
import os
from importlib.util import find_spec
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

OBS_DURATION_S = 150.0
PREDICTION_DURATION_S = 150.0
HISTORY_STEP_S = 5.0
FUTURE_STEP_S = 0.5
PREDICTION_LENGTH = 300
N_SUBSTEPS = 4

MODEL_ROOT = Path(__file__).resolve().parents[1] / "imran"
CHECKPOINT_PATH = MODEL_ROOT / "checkpoints" / "node.pt"


def imran_enabled() -> bool:
    return os.getenv("USE_IMRAN_PREDICTION", "").lower() in {"1", "true", "yes", "on"}


def imran_runtime_status() -> str:
    if not imran_enabled():
        return "disabled"
    if not CHECKPOINT_PATH.is_file() or find_spec("torch") is None:
        return "fallback"
    return "configured"


def predict_temperature_with_imran(
    input_data: Mapping[str, Any],
    fallback_prediction: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict a GPU trajectory, returning the supplied Minh result on any failure."""

    fallback = dict(fallback_prediction or {})
    fallback["source"] = _fallback_source(fallback)
    if not imran_enabled():
        return fallback

    try:
        torch, model, l_obs = _load_model()
        history, history_source = _build_history(input_data, l_obs)
        if history is None:
            return fallback

        obs = torch.tensor([history], dtype=torch.float32)
        t_obs = torch.linspace(0.0, OBS_DURATION_S, l_obs).unsqueeze(0)
        with torch.inference_mode():
            output = model(
                t_obs,
                obs,
                pred_len=PREDICTION_LENGTH,
                dt=FUTURE_STEP_S,
                obs_duration=OBS_DURATION_S,
                n_substeps=N_SUBSTEPS,
            )[0]

        values = output.detach().cpu().tolist()
        if not _valid_trajectory(values):
            return fallback
        return _normalize_prediction(input_data, fallback, values, history_source)
    except Exception:
        return fallback


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, int]:
    import torch

    from backend.imran.model_node import ThermalNODE

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)
    state = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    first_layer = state["encoder.net.0.weight"]
    l_obs = int(first_layer.shape[1] // 4)
    norm_scale = tuple(float(value) for value in state["norm_scale"].tolist())
    model = ThermalNODE(
        l_obs=l_obs,
        norm_scale=norm_scale,
        horizon_s=PREDICTION_DURATION_S,
    )
    model.load_state_dict(state)
    model.eval()
    return torch, model, l_obs


def _build_history(
    input_data: Mapping[str, Any],
    l_obs: int,
) -> tuple[list[list[float]] | None, str]:
    supplied = input_data.get("history")
    if isinstance(supplied, list) and len(supplied) >= l_obs:
        parsed = [_history_point(point) for point in supplied[-l_obs:]]
        if all(point is not None for point in parsed):
            return [point for point in parsed if point is not None], "observed"

    power = _number(input_data.get("power_draw_w"))
    current = _number(input_data.get("current_temp_c"))
    predicted = _number(input_data.get("predicted_temp_c"))
    utilization = _number(input_data.get("utilization_percent"))
    if power is None or current is None:
        return None, "missing"

    start_temp = current - max(1.0, min(8.0, (predicted or current) - current))
    start_power = power * max(0.55, min(0.95, (utilization or 70.0) / 100.0))
    history = []
    for index in range(l_obs):
        ratio = index / max(1, l_obs - 1)
        point_power = start_power + (power - start_power) * ratio
        point_temp = start_temp + (current - start_temp) * ratio
        history.append([point_power, point_temp, max(15.0, point_temp - 8.0)])
    return history, "synthetic"


def _history_point(point: object) -> list[float] | None:
    if not isinstance(point, Mapping):
        return None
    values = [
        _number(point.get("power_w")),
        _number(point.get("gpu_temp_c")),
        _number(point.get("heatsink_temp_c")),
    ]
    return None if any(value is None for value in values) else [float(value) for value in values]


def _normalize_prediction(
    input_data: Mapping[str, Any],
    fallback: Mapping[str, Any],
    values: list[list[float]],
    history_source: str,
) -> dict[str, Any]:
    temperatures = [point[1] for point in values]
    safe_limit = float(fallback.get("safe_limit_c") or 85.0)
    peak = max(temperatures)
    equilibrium = sum(temperatures[-20:]) / min(20, len(temperatures))
    crossing = next(
        (round((index + 1) * FUTURE_STEP_S, 2) for index, value in enumerate(temperatures) if value >= safe_limit),
        None,
    )
    risk = "critical" if peak >= safe_limit + 3 else "warning" if peak >= safe_limit - 5 else "safe"
    sample_step = max(1, len(values) // 20)
    trajectory = [
        {
            "t_s": round((index + 1) * FUTURE_STEP_S, 2),
            "power_w": round(values[index][0], 2),
            "gpu_temp_c": round(values[index][1], 2),
            "heatsink_temp_c": round(values[index][2], 2),
        }
        for index in range(0, len(values), sample_step)
    ]
    return {
        "gpu_id": input_data.get("gpu_id") or fallback.get("gpu_id"),
        "rack_id": input_data.get("rack_id") or fallback.get("rack_id"),
        "source": "imran_node",
        "source_history": history_source,
        "current_temp_c": input_data.get("current_temp_c") or fallback.get("current_temp_c"),
        "predicted_peak_temp_c": round(peak, 2),
        "predicted_equilibrium_temp_c": round(equilibrium, 2),
        "safe_limit_c": safe_limit,
        "time_to_threshold_s": crossing,
        "risk": risk,
        "confidence": 0.78 if history_source == "synthetic" else 0.86,
        "trajectory": trajectory[:22],
    }


def _valid_trajectory(values: object) -> bool:
    if not isinstance(values, list) or len(values) != PREDICTION_LENGTH:
        return False
    return all(
        isinstance(point, list)
        and len(point) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
        and 0.0 <= point[1] <= 200.0
        for point in values
    )


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _fallback_source(fallback: Mapping[str, Any]) -> str:
    source = str(fallback.get("source") or "")
    return "minh" if "minh" in source or "simulator" in source else "fallback"

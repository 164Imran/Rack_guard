"""Thermal mitigation simulation endpoint."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.thermal_adapter import simulate_action

ActionType = Literal[
    "increase_ventilation",
    "reduce_power_cap",
    "migrate_job",
    "no_action",
]

router = APIRouter(prefix="/api/simulations", tags=["simulations"])


class ActionSimulationRequest(BaseModel):
    gpu_id: str = Field(min_length=1, max_length=80)
    action_type: ActionType
    seed: int = 42


@router.post("/actions")
async def action_simulation(
    payload: ActionSimulationRequest,
) -> dict[str, object]:
    result = simulate_action(payload.gpu_id, payload.action_type, payload.seed)
    if result is None:
        raise HTTPException(status_code=404, detail="GPU not found")
    return result

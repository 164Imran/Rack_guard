"""Thin HTTP controller around the deterministic ROI engine."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.roi_engine import calculate_roi

router = APIRouter(prefix="/api", tags=["roi"])


class ROIRequest(BaseModel):
    action_type: Literal[
        "increase_ventilation",
        "reduce_power_cap",
        "migrate_job",
        "no_action",
    ]
    risk_no_action: float = Field(ge=0, le=1)
    risk_after_action: float = Field(ge=0, le=1)
    gpu_hour_value_eur: float = Field(ge=0)
    job_remaining_hours: float = Field(ge=0)
    expected_throttle_loss_percent: float = Field(ge=0, le=1)
    electricity_price_eur_kwh: float = Field(default=0, ge=0)
    extra_power_kw: float = Field(default=0, ge=0)
    performance_loss_percent: float = Field(default=0, ge=0, le=1)
    migration_time_hours: float = Field(default=0, ge=0)
    transfer_cost_eur: float = Field(default=0, ge=0)
    interruption_penalty_eur: float = Field(default=0, ge=0)


@router.post("/roi")
async def roi(payload: ROIRequest) -> dict[str, object]:
    try:
        return calculate_roi(payload.model_dump())
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

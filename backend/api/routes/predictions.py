"""Temperature prediction endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.thermal_adapter import get_temperature_prediction

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class TemperaturePredictionRequest(BaseModel):
    gpu_id: str = Field(min_length=1, max_length=80)
    seed: int = 42


@router.post("/temperature")
async def temperature_prediction(
    payload: TemperaturePredictionRequest,
) -> dict[str, object]:
    prediction = get_temperature_prediction(payload.gpu_id, payload.seed)
    if prediction is None:
        raise HTTPException(status_code=404, detail="GPU not found")
    return prediction

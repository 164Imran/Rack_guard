"""Final recommendation endpoint composing all available modules."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.debug_overrides import apply_extreme_debug
from ...services.decision_pipeline import build_final_recommendation

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


class FinalRecommendationRequest(BaseModel):
    gpu_id: str = Field(min_length=1, max_length=80)
    seed: int = 42
    financial_inputs: dict[str, dict[str, float]] | None = None
    risk_scores: dict[str, float] | None = None
    debug_extreme: bool = False


@router.post("/final")
async def final_recommendation(
    payload: FinalRecommendationRequest,
) -> dict[str, Any]:
    result = build_final_recommendation(
        gpu_id=payload.gpu_id,
        seed=payload.seed,
        financial_inputs=payload.financial_inputs,
        risk_scores=payload.risk_scores,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="GPU not found")
    if payload.debug_extreme:
        return apply_extreme_debug(result)
    return result

"""Current GPU state endpoint."""

from fastapi import APIRouter, Query

from ...services.thermal_adapter import get_current_gpu_state

router = APIRouter(prefix="/api/gpus", tags=["gpu-state"])


@router.get("/current")
async def current_gpu_state(
    rack_count: int = Query(default=1, ge=1, le=8),
    gpus_per_rack: int = Query(default=8, ge=1, le=16),
    seed: int = Query(default=42),
) -> dict[str, object]:
    return get_current_gpu_state(rack_count, gpus_per_rack, seed)

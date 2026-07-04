"""Health endpoint for the modular Rack Guardian backend."""

from fastapi import APIRouter

from ...services.thermal_adapter import thermal_source_status

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, object]:
    """Report orchestrator readiness without loading heavy model code."""

    return {
        "status": "ok",
        "service": "rack-guardian-backend",
        "modules": {
            "minh_thermal": thermal_source_status(),
            "roi_and_audit": "ready",
            "imran": "pending",
        },
        "mock_fallback": True,
    }

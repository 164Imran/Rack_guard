"""FastAPI application factory for the shared Rack Guardian backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.gpus import router as gpus_router
from .routes.health import router as health_router
from .routes.predictions import router as predictions_router
from .routes.recommendations import router as recommendations_router
from .routes.roi import router as roi_router
from .routes.simulations import router as simulations_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Rack Guardian Backend",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health_router)
    app.include_router(gpus_router)
    app.include_router(predictions_router)
    app.include_router(simulations_router)
    app.include_router(roi_router)
    app.include_router(recommendations_router)
    return app

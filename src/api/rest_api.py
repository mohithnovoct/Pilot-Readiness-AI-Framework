"""
REST API Server
================
FastAPI application exposing the Pilot Readiness Framework via
HTTP endpoints for enterprise integration.

Endpoints
---------
- ``POST /predict``       — batch inference
- ``POST /calibrate``     — per-pilot calibration
- ``GET  /status``        — system status
- ``GET  /components``    — list registered components
- ``GET  /config``        — current configuration
- ``GET  /health``        — health check

Launch
------
    uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8000

Or from Python:
    from src.api.rest_api import create_app, run_server
    run_server(config_path="default_config.yaml")
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("pilot_readiness.api")

# Lazy imports so the module can be imported even if FastAPI isn't installed
_app = None


def create_app(config: Optional[str | dict] = None):
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel as PydanticModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the REST API. "
            "Install with: pip install pilot-readiness[api]"
        )

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, PROJECT_ROOT)

    from src.core.framework import PilotReadinessFramework

    # ── Pydantic Models ──────────────────────────────────────

    class PredictRequest(PydanticModel):
        features: List[List[float]]
        feature_names: Optional[List[str]] = None

    class PredictResponse(PydanticModel):
        predictions: List[Dict[str, Any]]
        elapsed_ms: float

    class CalibrateRequest(PydanticModel):
        pilot_id: str
        features: List[List[float]]
        feature_names: Optional[List[str]] = None
        alpha: float = 0.05

    class CalibrateResponse(PydanticModel):
        pilot_id: str
        threshold: float
        baseline_risk_mean: float
        n_windows: int

    class StatusResponse(PydanticModel):
        status: str
        is_fitted: bool
        dataset: str
        n_models: int
        components: Dict[str, List[str]]

    class HealthResponse(PydanticModel):
        status: str
        timestamp: float
        version: str

    # ── App Setup ────────────────────────────────────────────

    app = FastAPI(
        title="Pilot Readiness AI Framework",
        description="REST API for real-time pilot readiness monitoring",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Framework instance
    fw = PilotReadinessFramework(config=config)

    # Store on app state for access in endpoints
    app.state.framework = fw

    # ── Endpoints ────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok",
            timestamp=time.time(),
            version="1.0.0",
        )

    @app.get("/status", response_model=StatusResponse)
    async def status():
        return StatusResponse(
            status="ready" if fw.is_fitted else "not_fitted",
            is_fitted=fw.is_fitted,
            dataset=fw.config.dataset,
            n_models=len(fw._models),
            components=fw.list_components(),
        )

    @app.get("/config")
    async def get_config():
        return fw.config.to_dict()

    @app.get("/components")
    async def list_components():
        return fw.list_components()

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        if not fw.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Framework not fitted. Train models first or load cached models.",
            )

        start = time.time()
        X = np.array(request.features, dtype=float)
        results = fw.predict(X=X)

        predictions = [
            {
                "risk_score": r.risk_score,
                "stress_probability": r.stress_probability,
                "performance_score": r.performance_score,
                "alert_level": r.alert_level,
                "decision": r.decision,
                "confidence": r.confidence,
            }
            for r in results
        ]

        elapsed = (time.time() - start) * 1000

        return PredictResponse(
            predictions=predictions,
            elapsed_ms=round(elapsed, 3),
        )

    @app.post("/calibrate", response_model=CalibrateResponse)
    async def calibrate(request: CalibrateRequest):
        if not fw.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Framework not fitted. Train models first.",
            )

        X = np.array(request.features, dtype=float)
        profile = fw.calibrate(
            pilot_id=request.pilot_id,
            baseline_features=X,
        )

        return CalibrateResponse(
            pilot_id=request.pilot_id,
            threshold=profile["threshold"],
            baseline_risk_mean=profile["baseline_risk_mean"],
            n_windows=profile["n_baseline_windows"],
        )

    @app.get("/profiles")
    async def list_profiles():
        from src.core.calibration import CalibrationSession
        profiles = CalibrationSession.list_profiles(fw.config.calibration.profile_dir)
        return {"profiles": profiles}

    @app.get("/profiles/{pilot_id}")
    async def get_profile(pilot_id: str):
        from src.core.calibration import CalibrationSession
        try:
            profile = CalibrationSession.load_profile(
                pilot_id, fw.config.calibration.profile_dir
            )
            from dataclasses import asdict
            return asdict(profile)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Profile for {pilot_id} not found")

    return app


def run_server(
    config: Optional[str | dict] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Launch the REST API server."""
    import uvicorn

    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


# Module-level app for `uvicorn src.api.rest_api:app`
app = create_app()

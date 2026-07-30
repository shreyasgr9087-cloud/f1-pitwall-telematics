from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from app.recourse_engine import TelemetryRecourseEngine, UnreachableTargetError

router = APIRouter(prefix="/api/recourse", tags=["recourse"])

class CurrentSetup(BaseModel):
    camber_front: float = Field(..., ge=-4.5, le=-1.5)
    tire_pressure_psi: float = Field(..., ge=20.0, le=30.0)
    brake_bias: float = Field(..., ge=50.0, le=65.0)
    driving_style_aggression: float = Field(..., ge=0.8, le=1.2)
    TrackTemp: float = Field(..., ge=0.0, le=70.0)
    AirTemp: float = Field(..., ge=-10.0, le=50.0)

class RecourseRequest(BaseModel):
    current_setup: CurrentSetup
    target_laps: float = Field(..., ge=4, le=100, description="Target laps until critical wear")
    penalty_weight: float = Field(100.0, gt=0)
    n_restarts: int = Field(5, ge=1, le=20)

@router.post("/")
def get_recourse(payload: RecourseRequest, request: Request):
    model = request.app.state.model
    metadata = request.app.state.metadata

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    engine = TelemetryRecourseEngine(
        model=model,
        feature_names=metadata["feature_names"],
        residual_by_region=metadata.get("residual_by_region"),
    )

    try:
        result = engine.optimize_setup(
            current_setup=payload.current_setup.model_dump(),
            target_laps=payload.target_laps,
            penalty_weight=payload.penalty_weight,
            n_restarts=payload.n_restarts,
        )
    except UnreachableTargetError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unreachable_target",
                "message": str(e),
                "target_laps": e.target_laps,
                "best_achievable_laps": e.best_achievable_laps,
                "best_setup_within_bounds": e.best_setup,
            },
        )

    return {
        "original_prediction": float(result["original_prediction"]),
        "original_prediction_uncertainty": result["original_prediction_uncertainty"],
        "new_prediction": float(result["new_prediction"]),
        "new_prediction_uncertainty": result["new_prediction_uncertainty"],
        "setup_changes": {k: float(v) for k, v in result["setup_changes"].items()},
        "new_setup": {k: float(v) for k, v in result["new_setup"].items()},
        "optimization_diagnostics": result["optimization_diagnostics"],
    }
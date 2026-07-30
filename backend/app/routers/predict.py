from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

router = APIRouter(prefix="/api/predict", tags=["predict"])

class SetupInput(BaseModel):
    camber_front: float = Field(..., ge=-4.5, le=-1.5)
    tire_pressure_psi: float = Field(..., ge=20.0, le=30.0)
    brake_bias: float = Field(..., ge=50.0, le=65.0)
    driving_style_aggression: float = Field(..., ge=0.8, le=1.2)
    TrackTemp: float = Field(..., ge=0.0, le=70.0)
    AirTemp: float = Field(..., ge=-10.0, le=50.0)

@router.post("/")
def predict_stint(payload: SetupInput, request: Request):
    model = request.app.state.model
    metadata = request.app.state.metadata

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = metadata["feature_names"]
    input_data = pd.DataFrame([payload.model_dump()], columns=features)

    # Model predicts effective_wear_rate directly
    predicted_wear_rate = float(model.predict(input_data)[0])
    
    # Safeguard against division by zero
    predicted_wear_rate = max(0.001, predicted_wear_rate)

    # Transform wear rate back into deterministic laps
    predicted_laps = float(70.0 / predicted_wear_rate)

    return {
        "predicted_laps": round(predicted_laps, 2),
        "model_trained_at": metadata.get("trained_at"),
    }
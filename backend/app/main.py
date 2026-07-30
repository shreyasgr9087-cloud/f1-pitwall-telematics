from contextlib import asynccontextmanager
from pathlib import Path
import json
import pickle
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict, recourse

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = Path(__file__).parent / "models" / "surrogate_xgb.pkl"
METADATA_PATH = Path(__file__).parent / "models" / "surrogate_xgb_metadata.json"

REQUIRED_METADATA_KEYS = {"feature_names", "training_data_hash", "trained_at", "holdout_r2"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load model + metadata once, fail loudly if inconsistent ---
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run train_model.py first.")
    if not METADATA_PATH.exists():
        raise RuntimeError(
            f"Metadata sidecar not found at {METADATA_PATH}. "
            "A model without metadata cannot be safely served."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    missing_keys = REQUIRED_METADATA_KEYS - metadata.keys()
    if missing_keys:
        raise RuntimeError(f"Metadata sidecar missing required keys: {missing_keys}")

    expected_n_features = len(metadata["feature_names"])
    model_n_features = getattr(model, "n_features_in_", None)
    if model_n_features is not None and model_n_features != expected_n_features:
        raise RuntimeError(
            f"Model/metadata schema mismatch: model expects {model_n_features} features, "
            f"metadata lists {expected_n_features}."
        )

    logger.info(
        f"Loaded surrogate model trained_at={metadata['trained_at']} "
        f"holdout_r2={metadata['holdout_r2']} features={metadata['feature_names']}"
    )

    app.state.model = model
    app.state.metadata = metadata

    yield

    # --- Shutdown cleanup ---
    app.state.model = None
    app.state.metadata = None

app = FastAPI(title="Telemetry Recourse API", lifespan=lifespan)

# Pinned explicit origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://localhost:5174", 
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(recourse.router)

@app.get("/")
def read_root():
    return {"status": "Pit-wall backend online"}

@app.get("/health")
def health(request: Request):
    metadata = request.app.state.metadata
    return {
        "status": "ok" if request.app.state.model is not None else "model not loaded",
        "model_trained_at": metadata["trained_at"] if metadata else None,
        "model_holdout_r2": metadata["holdout_r2"] if metadata else None,
    }
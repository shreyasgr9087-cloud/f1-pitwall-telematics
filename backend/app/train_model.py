from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pickle

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "app" / "data" / "hybrid_stint_telemetry.csv"
MODEL_DIR = BASE_DIR / "app" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "surrogate_xgb.pkl"
METADATA_PATH = MODEL_DIR / "surrogate_xgb_metadata.json"

FEATURES = [
    "camber_front",
    "tire_pressure_psi",
    "brake_bias",
    "driving_style_aggression",
    "TrackTemp",
    "AirTemp",
]
# Train on wear rate directly to avoid reciprocal distortion
TARGET = "effective_wear_rate"

def hash_dataframe(df: pd.DataFrame) -> str:
    data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()

def residual_analysis(y_test_laps, lap_predictions, X_test):
    """Evaluates residuals in terms of LAPS (transformed back via 70 / wear_rate)"""
    residuals = y_test_laps - lap_predictions
    abs_residuals = np.abs(residuals)

    print("\n--- Residual Analysis (Evaluated in Laps) ---")
    print(f"Mean absolute error:   {abs_residuals.mean():.3f} laps")
    print(f"95th pct abs error:    {np.percentile(abs_residuals, 95):.3f} laps")
    print(f"Max abs error:         {abs_residuals.max():.3f} laps")

    test_df = X_test.copy()
    test_df["y_true_laps"] = y_test_laps
    test_df["y_pred_laps"] = lap_predictions
    test_df["abs_error"] = abs_residuals

    low_life = test_df[test_df["y_true_laps"] <= test_df["y_true_laps"].quantile(0.25)]
    high_life = test_df[test_df["y_true_laps"] >= test_df["y_true_laps"].quantile(0.75)]

    print(f"Mean abs error, bottom 25% (short stints):  {low_life['abs_error'].mean():.3f} laps")
    print(f"Mean abs error, top 25% (long stints):      {high_life['abs_error'].mean():.3f} laps")

    global_mae = abs_residuals.mean()
    if low_life["abs_error"].mean() > 1.5 * global_mae:
        print("WARNING: error concentrated in short-stint region.")
    if high_life["abs_error"].mean() > 1.5 * global_mae:
        print("WARNING: error concentrated in long-stint region.")

    worst_idx = test_df["abs_error"].idxmax()
    worst_row = test_df.loc[worst_idx]
    print("\n--- Worst single prediction (max abs error) ---")
    print(worst_row[FEATURES + ["y_true_laps", "y_pred_laps", "abs_error"]].to_string())
    print("------------------------------------------------\n")

    return {
        "mae": float(global_mae),
        "p95_abs_error": float(np.percentile(abs_residuals, 95)),
        "max_abs_error": float(abs_residuals.max()),
        "mae_short_stint_q25": float(low_life["abs_error"].mean()),
        "mae_long_stint_q75": float(high_life["abs_error"].mean()),
        "worst_case_row": worst_row[FEATURES + ["y_true_laps", "y_pred_laps", "abs_error"]].to_dict(),
    }

def compute_regional_residual_stats(y_test_laps, lap_predictions, n_bins=4):
    abs_err = np.abs(y_test_laps - lap_predictions)
    edges = np.quantile(y_test_laps, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf 

    regions = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_test_laps >= lo) & (y_test_laps < hi) if i < n_bins - 1 else (y_test_laps >= lo) & (y_test_laps <= hi)
        bucket_err = abs_err[mask]
        regions.append({
            "bin_index": i,
            "lower": float(lo),
            "upper": float(hi),
            "n": int(mask.sum()),
            "mae": float(bucket_err.mean()) if mask.sum() > 0 else None,
            "p95_abs_error": float(np.percentile(bucket_err, 95)) if mask.sum() > 0 else None,
        })
    return regions

def train_surrogate():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing expected columns: {missing}")

    data_hash = hash_dataframe(df)
    print(f"Training data SHA256: {data_hash}")

    X = df[FEATURES]
    y = df[TARGET] # Training on wear rate!

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training XGBoost Surrogate Model on Wear Rate...")
    model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict wear rate, then convert to laps via reciprocal transform
    pred_wear_rate = model.predict(X_test)
    predictions_laps = 70.0 / pred_wear_rate
    y_test_laps = 70.0 / y_test.values

    rmse = float(np.sqrt(mean_squared_error(y_test_laps, predictions_laps)))
    r2 = float(r2_score(y_test_laps, predictions_laps))

    print("\n--- Model Evaluation (holdout set evaluated in laps) ---")
    print(f"RMSE (Laps): {rmse:.2f}")
    print(f"R² Score:    {r2:.4f}")
    print("---------------------------------------")

    residual_stats = residual_analysis(y_test_laps, predictions_laps, X_test)
    
    regional_stats = compute_regional_residual_stats(y_test_laps, predictions_laps, n_bins=4)
    print("\n--- Regional Residual Stats (for recourse uncertainty) ---")
    for r in regional_stats:
        print(f"  [{r['lower']:.1f}, {r['upper']:.1f}) laps: n={r['n']}, MAE={r['mae']:.3f}, p95={r['p95_abs_error']:.3f}")
    print("------------------------------------------------------------\n")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}")

    metadata = {
        "feature_names": FEATURES,
        "target": TARGET,
        "training_data_hash": data_hash,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "holdout_rmse": rmse,
        "holdout_r2": r2,
        "n_train_samples": int(len(X_train)),
        "n_test_samples": int(len(X_test)),
        "residual_stats": residual_stats,
        "residual_by_region": regional_stats,
        "model_params": model.get_params(),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata sidecar saved to: {METADATA_PATH}")

if __name__ == "__main__":
    train_surrogate()
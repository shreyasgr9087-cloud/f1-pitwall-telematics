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

# The noise std used in hybrid_physics_generator.py (line 62).
# This is the irreducible noise floor; even a perfect model cannot eliminate it.
GENERATOR_NOISE_STD = 0.1


def compute_bayes_ceiling(y_test_wear_rate, n_mc=5000, seed=99):
    """
    Computes the theoretical best-possible R²/RMSE a perfect model could achieve,
    given the known Gaussian noise floor (σ=0.1) injected in wear-rate space.

    Method: For each holdout sample, the "true" noiseless wear rate is unknown,
    but the observed wear rate = noiseless + N(0, σ). A perfect model would predict
    the noiseless value exactly. The irreducible error in laps-space is therefore
    caused solely by the noise propagating through the 70/w reciprocal transform.

    We estimate this via Monte Carlo: for each observed w, draw MC samples of the
    noise ε ~ N(0, σ), compute laps_noisy = 70/(w) vs laps_clean = 70/(w - ε),
    and measure the residual. This gives the Bayes-optimal error floor.
    """
    rng = np.random.default_rng(seed)
    y_laps_observed = 70.0 / y_test_wear_rate

    # For each sample, the noiseless wear rate is w_obs - ε where ε ~ N(0, σ)
    # A perfect model predicts w_noiseless = w_obs - ε, so its lap prediction
    # would be 70/(w_obs - ε). The "true" laps we evaluate against is 70/w_obs.
    # The irreducible squared error per sample is E[(70/w_obs - 70/(w_obs - ε))²].
    sq_errors = []
    for w_obs in y_test_wear_rate:
        eps_samples = rng.normal(0, GENERATOR_NOISE_STD, n_mc)
        w_clean = np.maximum(0.001, w_obs - eps_samples)
        laps_clean = 70.0 / w_clean
        laps_observed = 70.0 / w_obs
        sq_errors.append(np.mean((laps_observed - laps_clean) ** 2))

    ceiling_mse = np.mean(sq_errors)
    ceiling_rmse = np.sqrt(ceiling_mse)

    # Bayes-optimal R² in laps space
    ss_tot = np.sum((y_laps_observed - y_laps_observed.mean()) ** 2)
    # The irreducible variance sets a floor on SS_res
    ss_res_floor = ceiling_mse * len(y_laps_observed)
    ceiling_r2 = 1.0 - (ss_res_floor / ss_tot)

    return float(ceiling_r2), float(ceiling_rmse)


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

    # Bayes ceiling: the best any model could achieve given the noise floor
    ceiling_r2, ceiling_rmse = compute_bayes_ceiling(y_test.values)
    pct_of_ceiling = (r2 / ceiling_r2) * 100.0 if ceiling_r2 > 0 else 0.0

    print("\n--- Model Evaluation (holdout set evaluated in laps) ---")
    print(f"RMSE (Laps):           {rmse:.2f}")
    print(f"R^2 Score:             {r2:.4f}")
    print(f"Bayes Ceiling R^2:     {ceiling_r2:.4f}  (irreducible noise floor sigma={GENERATOR_NOISE_STD})")
    print(f"Bayes Ceiling RMSE:    {ceiling_rmse:.4f} laps")
    print(f"Model % of ceiling:    {pct_of_ceiling:.1f}%")
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
        "bayes_ceiling_r2_laps": ceiling_r2,
        "bayes_ceiling_rmse_laps": ceiling_rmse,
        "model_pct_of_ceiling": round(pct_of_ceiling, 1),
        "generator_noise_std": GENERATOR_NOISE_STD,
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
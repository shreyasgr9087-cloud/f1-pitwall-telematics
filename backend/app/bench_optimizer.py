"""
Optimizer Comparison Benchmark
==============================
Compares SciPy's differential_evolution (gradient-free, used in production)
against multi-start SLSQP (gradient-based) on the same objective/bounds.

Reports wall-clock time and best objective value for each method.
Results inform the "Known Issues / Engineering Trade-offs" README section.

Run:  python backend/app/bench_optimizer.py
"""
import time
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "models" / "surrogate_xgb.pkl"
METADATA_PATH = BASE_DIR / "app" / "models" / "surrogate_xgb_metadata.json"

ADJUSTABLE_FEATURES = ["camber_front", "tire_pressure_psi", "brake_bias"]
CONTEXT_FEATURES = ["driving_style_aggression", "TrackTemp", "AirTemp"]
BOUNDS = [(-4.5, -1.5), (20.0, 30.0), (50.0, 65.0)]
NORM_DENOMS = [b[1] - b[0] for b in BOUNDS]


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata["feature_names"]


def predict_laps(model, features, adjustable_vals, fixed_ctx):
    row = {**dict(zip(ADJUSTABLE_FEATURES, adjustable_vals)), **fixed_ctx}
    df = pd.DataFrame([row], columns=features)
    wear_rate = float(model.predict(df)[0])
    wear_rate = max(0.001, wear_rate)
    return 70.0 / wear_rate


def make_objective(model, features, current_adjustable, fixed_ctx, target_laps, penalty_weight=100.0):
    def objective(x):
        pred = predict_laps(model, features, x, fixed_ctx)
        penalty = 0.0
        if pred < target_laps:
            penalty = penalty_weight * ((target_laps - pred) ** 2)
        l1_dist = sum(
            abs(x[i] - current_adjustable[i]) / NORM_DENOMS[i]
            for i in range(len(ADJUSTABLE_FEATURES))
        )
        return penalty + l1_dist
    return objective


def run_benchmark():
    model, feature_names = load_model()

    test_cases = [
        {
            "name": "Mid-range setup, moderate target",
            "setup": {"camber_front": -3.0, "tire_pressure_psi": 24.5, "brake_bias": 56.0,
                      "driving_style_aggression": 1.0, "TrackTemp": 35.0, "AirTemp": 22.0},
            "target": 20.0,
        },
        {
            "name": "Harsh setup, ambitious target",
            "setup": {"camber_front": -4.0, "tire_pressure_psi": 22.0, "brake_bias": 62.0,
                      "driving_style_aggression": 1.1, "TrackTemp": 45.0, "AirTemp": 30.0},
            "target": 18.0,
        },
        {
            "name": "Gentle setup, easy target",
            "setup": {"camber_front": -2.0, "tire_pressure_psi": 25.0, "brake_bias": 52.0,
                      "driving_style_aggression": 0.9, "TrackTemp": 28.0, "AirTemp": 18.0},
            "target": 25.0,
        },
    ]

    n_restarts = 5

    print("=" * 80)
    print("OPTIMIZER COMPARISON: differential_evolution vs multi-start SLSQP")
    print("=" * 80)

    for case in test_cases:
        setup = case["setup"]
        target = case["target"]
        fixed_ctx = {f: setup[f] for f in CONTEXT_FEATURES}
        current_adj = np.array([setup[f] for f in ADJUSTABLE_FEATURES])
        obj = make_objective(model, feature_names, current_adj, fixed_ctx, target)

        print(f"\n--- {case['name']} (target={target} laps) ---")

        # Differential Evolution (production method)
        t0 = time.perf_counter()
        de_results = []
        for seed in range(n_restarts):
            res = differential_evolution(obj, BOUNDS, seed=seed, popsize=15, maxiter=50, polish=True)
            de_results.append(res)
        de_time = time.perf_counter() - t0
        de_best = min(de_results, key=lambda r: r.fun)
        de_laps = predict_laps(model, feature_names, de_best.x, fixed_ctx)

        # Multi-start SLSQP (gradient-based alternative)
        t0 = time.perf_counter()
        slsqp_results = []
        rng = np.random.default_rng(42)
        for _ in range(n_restarts):
            x0 = np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS])
            res = minimize(obj, x0, method="SLSQP",
                           bounds=BOUNDS, options={"maxiter": 200, "ftol": 1e-9})
            slsqp_results.append(res)
        slsqp_time = time.perf_counter() - t0
        slsqp_best = min(slsqp_results, key=lambda r: r.fun)
        slsqp_laps = predict_laps(model, feature_names, slsqp_best.x, fixed_ctx)

        print(f"  {'Method':<30} {'Objective':>12} {'Laps':>10} {'Time (s)':>10}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*10}")
        print(f"  {'Differential Evolution':<30} {de_best.fun:>12.6f} {de_laps:>10.2f} {de_time:>10.3f}")
        print(f"  {'Multi-start SLSQP':<30} {slsqp_best.fun:>12.6f} {slsqp_laps:>10.2f} {slsqp_time:>10.3f}")

        if de_best.fun <= slsqp_best.fun:
            winner = "Differential Evolution"
        else:
            winner = "Multi-start SLSQP"
        speed = "DE" if de_time < slsqp_time else "SLSQP"
        print(f"  → Better objective: {winner}")
        print(f"  → Faster wall-clock: {speed}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("  Differential Evolution is used in production because XGBoost trees are")
    print("  non-differentiable step functions. SLSQP uses finite-difference gradient")
    print("  approximations which can miss step boundaries. DE is gradient-free and")
    print("  globally searching, making it more robust for this landscape -- at the")
    print("  cost of higher wall-clock time per restart.")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

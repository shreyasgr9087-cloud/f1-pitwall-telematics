from pathlib import Path
import pickle
import json

import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "models" / "surrogate_xgb.pkl"
METADATA_PATH = BASE_DIR / "app" / "models" / "surrogate_xgb_metadata.json"

ADJUSTABLE_FEATURES = ["camber_front", "tire_pressure_psi", "brake_bias"]
CONTEXT_FEATURES = ["driving_style_aggression", "TrackTemp", "AirTemp"]
BOUNDS = [(-4.5, -1.5), (20.0, 30.0), (50.0, 65.0)]

NORM_DENOMINATORS = {
    "camber_front": BOUNDS[0][1] - BOUNDS[0][0],
    "tire_pressure_psi": BOUNDS[1][1] - BOUNDS[1][0],
    "brake_bias": BOUNDS[2][1] - BOUNDS[2][0],
}

class UnreachableTargetError(Exception):
    def __init__(self, target_laps, best_achievable_laps, best_setup):
        self.target_laps = target_laps
        self.best_achievable_laps = best_achievable_laps
        self.best_setup = best_setup
        super().__init__(
            f"Target of {target_laps} laps is unreachable within setup bounds. "
            f"Best achievable within search space: {best_achievable_laps:.1f} laps."
        )

class TelemetryRecourseEngine:
    def __init__(self, model, feature_names, residual_by_region=None):
        self.model = model
        self.features = feature_names
        self.residual_by_region = residual_by_region or []

        missing = [f for f in ADJUSTABLE_FEATURES + CONTEXT_FEATURES if f not in feature_names]
        if missing:
            raise ValueError(f"Injected feature schema is missing expected columns: {missing}")

    @classmethod
    def from_disk(cls):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        return cls(
            model=model,
            feature_names=metadata["feature_names"],
            residual_by_region=metadata.get("residual_by_region"),
        )

    def _predict(self, adjustable_values, fixed_context):
        """Predicts wear rate directly, then transforms to laps via 70 / wear_rate"""
        row = {**dict(zip(ADJUSTABLE_FEATURES, adjustable_values)), **fixed_context}
        df = pd.DataFrame([row], columns=self.features)
        predicted_wear_rate = float(self.model.predict(df)[0])
        predicted_wear_rate = max(0.001, predicted_wear_rate)
        return 70.0 / predicted_wear_rate

    def _uncertainty_for(self, predicted_value):
        for region in self.residual_by_region:
            if region["lower"] <= predicted_value < region["upper"] or (
                region["upper"] == float("inf") and predicted_value >= region["lower"]
            ):
                if region["mae"] is not None:
                    return {
                        "mae_laps": region["mae"],
                        "p95_error_laps": region["p95_abs_error"],
                        "region_sample_size": region["n"],
                        "source": "regional",
                    }
        return {
            "mae_laps": None,
            "p95_error_laps": None,
            "region_sample_size": 0,
            "source": "unavailable",
        }

    def _validate_current_setup(self, current_setup):
        missing = [f for f in ADJUSTABLE_FEATURES + CONTEXT_FEATURES if f not in current_setup]
        if missing:
            raise ValueError(f"current_setup is missing required fields: {missing}")
        for name, (lo, hi) in zip(ADJUSTABLE_FEATURES, BOUNDS):
            value = current_setup[name]
            if not (lo <= value <= hi):
                raise ValueError(
                    f"current_setup['{name}']={value} is outside physical bounds [{lo}, {hi}]"
                )

    def optimize_setup(
        self,
        current_setup: dict,
        target_laps: float,
        penalty_weight: float = 100.0,
        n_restarts: int = 5,
        feasibility_tolerance: float = 0.5,
        maxiter: int = 50,
        popsize: int = 15,
    ):
        self._validate_current_setup(current_setup)

        fixed_context = {f: current_setup[f] for f in CONTEXT_FEATURES}
        current_adjustable = np.array([current_setup[f] for f in ADJUSTABLE_FEATURES])

        original_prediction = self._predict(current_adjustable, fixed_context)

        def objective_function(x):
            predicted_laps = self._predict(x, fixed_context)
            target_penalty = 0.0
            if predicted_laps < target_laps:
                target_penalty = penalty_weight * ((target_laps - predicted_laps) ** 2)
            norm_distance = sum(
                abs(x[i] - current_adjustable[i]) / NORM_DENOMINATORS[name]
                for i, name in enumerate(ADJUSTABLE_FEATURES)
            )
            return target_penalty + norm_distance

        restart_results = []
        for seed in range(n_restarts):
            result = differential_evolution(
                objective_function, BOUNDS, seed=seed, popsize=popsize, maxiter=maxiter, polish=True,
            )
            restart_results.append(result)

        best_result = min(restart_results, key=lambda r: r.fun)
        best_setup = best_result.x
        best_predicted_laps = self._predict(best_setup, fixed_context)

        other_setups = np.array([r.x for r in restart_results if not np.array_equal(r.x, best_setup)])
        if len(other_setups) > 0:
            spread = np.mean([
                sum(abs(s[i] - best_setup[i]) / NORM_DENOMINATORS[name] for i, name in enumerate(ADJUSTABLE_FEATURES))
                for s in other_setups
            ])
        else:
            spread = 0.0

        # Stability thresholds: these are empirically calibrated placeholders based on
        # observed restart spreads during development. They are NOT derived from a
        # rigorous statistical analysis of the optimization landscape. The intuition:
        #   < 0.1 normalized L1 distance → restarts converge tightly → "stable"
        #   0.1–0.3 → moderate variation between restarts → "moderate"
        #   > 0.3 → restarts find substantially different optima → "unstable"
        # If these do not match your problem's observed spread distribution, tune them
        # by running bench_optimizer.py across a sample of setups/targets.
        stability = "stable" if spread < 0.1 else ("moderate" if spread < 0.3 else "unstable")

        if best_predicted_laps < target_laps - feasibility_tolerance:
            raise UnreachableTargetError(
                target_laps=target_laps,
                best_achievable_laps=best_predicted_laps,
                best_setup=dict(zip(ADJUSTABLE_FEATURES, best_setup)),
            )

        new_uncertainty = self._uncertainty_for(best_predicted_laps)
        original_uncertainty = self._uncertainty_for(original_prediction)

        return {
            "original_prediction": original_prediction,
            "original_prediction_uncertainty": original_uncertainty,
            "new_prediction": best_predicted_laps,
            "new_prediction_uncertainty": new_uncertainty,
            "setup_changes": {
                f"{name}_change": float(best_setup[i] - current_adjustable[i])
                for i, name in enumerate(ADJUSTABLE_FEATURES)
            },
            "new_setup": {name: float(best_setup[i]) for i, name in enumerate(ADJUSTABLE_FEATURES)},
            "optimization_diagnostics": {
                "n_restarts": n_restarts,
                "stability": stability,
                "restart_spread": float(spread),
                "penalty_weight_used": penalty_weight,
                "best_objective_value": float(best_result.fun),
            },
        }

if __name__ == "__main__":
    engine = TelemetryRecourseEngine.from_disk()
    test_setup = {
        "camber_front": -3.8, "tire_pressure_psi": 22.0, "brake_bias": 60.0,
        "driving_style_aggression": 1.1, "TrackTemp": 42.0, "AirTemp": 28.0,
    }
    try:
        result = engine.optimize_setup(test_setup, target_laps=18)
        print(json.dumps(result, indent=2))
    except UnreachableTargetError as e:
        print(f"Infeasible: {e}")
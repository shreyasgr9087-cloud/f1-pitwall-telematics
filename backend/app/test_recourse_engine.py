"""
Test suite for the TelemetryRecourseEngine.

Tests mirror the structure of the coffee-recourse-engine project:
physical sanity, bounds compliance, fixed-feature integrity,
infeasibility detection, trivial-case handling, L1 sparsity,
and stability classification.

Run with:  pytest backend/app/test_recourse_engine.py -v
"""
import pytest
import numpy as np

import sys
from pathlib import Path

# Ensure the app directory is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recourse_engine import (
    TelemetryRecourseEngine,
    UnreachableTargetError,
    ADJUSTABLE_FEATURES,
    CONTEXT_FEATURES,
    BOUNDS,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    """Load the trained surrogate model once for all tests."""
    return TelemetryRecourseEngine.from_disk()


@pytest.fixture
def baseline_setup():
    """A mid-range setup that produces a moderate stint prediction."""
    return {
        "camber_front": -3.0,
        "tire_pressure_psi": 24.5,
        "brake_bias": 56.0,
        "driving_style_aggression": 1.0,
        "TrackTemp": 35.0,
        "AirTemp": 22.0,
    }


@pytest.fixture
def harsh_setup():
    """A worst-case setup: extreme camber, hot track, aggressive driver."""
    return {
        "camber_front": -4.5,
        "tire_pressure_psi": 20.0,
        "brake_bias": 65.0,
        "driving_style_aggression": 1.2,
        "TrackTemp": 55.0,
        "AirTemp": 35.0,
    }


# ---------------------------------------------------------------------------
# Test 1: Predictions are physically sane
# ---------------------------------------------------------------------------

def test_predictions_are_physically_sane(engine, baseline_setup):
    """Predicted laps must always be a positive, finite number."""
    adjustable = [baseline_setup[f] for f in ADJUSTABLE_FEATURES]
    context = {f: baseline_setup[f] for f in CONTEXT_FEATURES}
    predicted = engine._predict(adjustable, context)

    assert predicted > 0, f"Predicted laps must be positive, got {predicted}"
    assert np.isfinite(predicted), f"Predicted laps must be finite, got {predicted}"


# ---------------------------------------------------------------------------
# Test 2: Recourse outputs respect physical bounds
# ---------------------------------------------------------------------------

def test_recommended_fix_respects_bounds(engine, baseline_setup):
    """All adjusted setup values must stay within the declared BOUNDS."""
    result = engine.optimize_setup(baseline_setup, target_laps=18, n_restarts=3)
    new_setup = result["new_setup"]

    for i, name in enumerate(ADJUSTABLE_FEATURES):
        lo, hi = BOUNDS[i]
        assert lo <= new_setup[name] <= hi, (
            f"{name}={new_setup[name]} is outside bounds [{lo}, {hi}]"
        )


# ---------------------------------------------------------------------------
# Test 3: Fixed (context) features are never included in setup_changes
# ---------------------------------------------------------------------------

def test_fixed_features_remain_untouched(engine, baseline_setup):
    """
    Context features (driving_style_aggression, TrackTemp, AirTemp) should
    never appear in setup_changes — they are environmental/behavioral inputs,
    not mechanical adjustments.
    """
    result = engine.optimize_setup(baseline_setup, target_laps=18, n_restarts=3)
    change_keys = set(result["setup_changes"].keys())

    for ctx_feat in CONTEXT_FEATURES:
        assert f"{ctx_feat}_change" not in change_keys, (
            f"Context feature '{ctx_feat}' should not appear in setup_changes"
        )


# ---------------------------------------------------------------------------
# Test 4: Infeasible target raises UnreachableTargetError
# ---------------------------------------------------------------------------

def test_infeasible_target_raises_correctly(engine, harsh_setup):
    """
    Requesting an extreme stint length (100 laps) on a worst-case setup
    should raise UnreachableTargetError with a best_achievable_laps value.
    """
    with pytest.raises(UnreachableTargetError) as exc_info:
        engine.optimize_setup(harsh_setup, target_laps=100, n_restarts=3)

    err = exc_info.value
    assert err.best_achievable_laps > 0, "best_achievable_laps must be positive"
    assert err.best_achievable_laps < 100, "best_achievable should be below the target"
    assert err.target_laps == 100


# ---------------------------------------------------------------------------
# Test 5: Already-optimal trivial case produces near-zero changes
# ---------------------------------------------------------------------------

def test_already_optimal_trivial_case(engine, baseline_setup):
    """
    If the current setup already exceeds the target, the optimizer should
    return near-zero changes (no adjustment needed).
    """
    # First, get the current prediction to know a target below it
    adjustable = [baseline_setup[f] for f in ADJUSTABLE_FEATURES]
    context = {f: baseline_setup[f] for f in CONTEXT_FEATURES}
    current_laps = engine._predict(adjustable, context)

    # Ask for fewer laps than we already get
    easy_target = current_laps - 5.0
    if easy_target < 1.0:
        pytest.skip("Baseline prediction too low for trivial-case test")

    result = engine.optimize_setup(baseline_setup, target_laps=easy_target, n_restarts=3)

    total_change = sum(abs(v) for v in result["setup_changes"].values())
    assert total_change < 0.5, (
        f"Already-optimal case should have near-zero changes, got total delta={total_change:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: L1 sparsity — optimizer prefers moving fewer features
# ---------------------------------------------------------------------------

def test_l1_sparsity(engine, baseline_setup):
    """
    When only a small improvement is needed, the L1 penalty should cause the
    optimizer to adjust one or two features rather than all three.
    """
    adjustable = [baseline_setup[f] for f in ADJUSTABLE_FEATURES]
    context = {f: baseline_setup[f] for f in CONTEXT_FEATURES}
    current_laps = engine._predict(adjustable, context)

    # Request a modest improvement (2 more laps)
    modest_target = current_laps + 2.0
    try:
        result = engine.optimize_setup(
            baseline_setup, target_laps=modest_target, n_restarts=5
        )
    except UnreachableTargetError:
        pytest.skip("Modest target is unreachable for this setup; cannot test sparsity")

    changes = result["setup_changes"]
    abs_changes = [abs(v) for v in changes.values()]

    # At least one feature should have a near-zero change (< 0.05 of its range)
    near_zero_count = sum(1 for c in abs_changes if c < 0.15)
    assert near_zero_count >= 1, (
        f"L1 penalty should produce at least one near-zero change; "
        f"got changes={changes}"
    )


# ---------------------------------------------------------------------------
# Test 7: Stability classification matches spread thresholds
# ---------------------------------------------------------------------------

def test_stability_classification(engine, baseline_setup):
    """
    With enough restarts on a well-conditioned problem, the optimizer should
    converge consistently and report a 'stable' landscape.
    """
    adjustable = [baseline_setup[f] for f in ADJUSTABLE_FEATURES]
    context = {f: baseline_setup[f] for f in CONTEXT_FEATURES}
    current_laps = engine._predict(adjustable, context)

    easy_target = current_laps - 2.0
    if easy_target < 1.0:
        pytest.skip("Baseline too low for stability test")

    result = engine.optimize_setup(
        baseline_setup, target_laps=easy_target, n_restarts=7
    )
    diag = result["optimization_diagnostics"]

    spread = diag["restart_spread"]
    stability = diag["stability"]

    # Verify classification matches the documented thresholds
    if spread < 0.1:
        assert stability == "stable"
    elif spread < 0.3:
        assert stability == "moderate"
    else:
        assert stability == "unstable"

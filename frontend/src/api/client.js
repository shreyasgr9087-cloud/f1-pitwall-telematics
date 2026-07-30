const BASE_URL = "http://127.0.0.1:8001";

/**
 * Thin fetch wrapper with structured error handling.
 * Parses JSON responses and surfaces API-level errors
 * (including 422 UnreachableTargetError) with their full detail payload.
 */
async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const config = {
    headers: { "Content-Type": "application/json" },
    ...options,
  };

  const response = await fetch(url, config);

  if (!response.ok) {
    const errorBody = await response.json().catch(() => null);

    const error = new Error(
      errorBody?.detail?.message || errorBody?.detail || `API error ${response.status}`
    );
    error.status = response.status;
    error.detail = errorBody?.detail || null;
    throw error;
  }

  return response.json();
}

/**
 * GET /health — Backend readiness & model metadata.
 */
export async function fetchHealth() {
  return request("/health");
}

/**
 * POST /api/predict — Surrogate model prediction.
 * @param {Object} setup - 6 feature values matching SetupInput schema
 * @returns {{ predicted_laps: number, model_trained_at: string }}
 */
export async function fetchPrediction(setup) {
  return request("/api/predict/", {
    method: "POST",
    body: JSON.stringify(setup),
  });
}

/**
 * POST /api/recourse — Differential evolution recourse optimization.
 * @param {Object} currentSetup - Current 6-feature setup dict
 * @param {number} targetLaps - Desired stint length
 * @param {number} [penaltyWeight=100] - Objective penalty weight
 * @param {number} [nRestarts=5] - DE multi-seed restarts
 * @returns {Object} Recourse result with uncertainty bands and diagnostics
 * @throws {Error} With error.detail containing unreachable_target info on 422
 */
export async function fetchRecourse(currentSetup, targetLaps, penaltyWeight = 100, nRestarts = 5) {
  return request("/api/recourse/", {
    method: "POST",
    body: JSON.stringify({
      current_setup: currentSetup,
      target_laps: targetLaps,
      penalty_weight: penaltyWeight,
      n_restarts: nRestarts,
    }),
  });
}

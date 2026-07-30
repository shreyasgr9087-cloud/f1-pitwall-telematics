import React, { useState } from "react";
import {
  Target,
  Loader2,
  ArrowRightLeft,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Wrench,
  ShieldAlert,
  Zap,
} from "lucide-react";
import { fetchRecourse } from "../api/client";

/**
 * Formats a numeric delta with sign and fixed precision.
 */
function formatDelta(value, decimals = 2) {
  const sign = value > 0.005 ? "+" : value < -0.005 ? "" : "±";
  if (Math.abs(value) < 0.005) return "±0.00";
  return `${sign}${value.toFixed(decimals)}`;
}

/**
 * Returns a CSS class based on delta direction.
 */
function deltaClass(value) {
  if (value > 0.005) return "delta-positive";
  if (value < -0.005) return "delta-negative";
  return "delta-neutral";
}

/**
 * Maps stability string to visual badge config.
 */
function stabilityBadge(stability) {
  switch (stability) {
    case "stable":
      return {
        icon: CheckCircle2,
        color: "text-neon-emerald bg-neon-emerald/10 border-neon-emerald/20",
        label: "STABLE",
      };
    case "moderate":
      return {
        icon: AlertTriangle,
        color: "text-neon-amber bg-neon-amber/10 border-neon-amber/20",
        label: "MODERATE",
      };
    case "unstable":
      return {
        icon: ShieldAlert,
        color: "text-neon-crimson bg-neon-crimson/10 border-neon-crimson/20",
        label: "UNSTABLE",
      };
    default:
      return {
        icon: AlertTriangle,
        color: "text-slate-400 bg-slate-400/10 border-slate-400/20",
        label: stability?.toUpperCase() || "UNKNOWN",
      };
  }
}

const SETUP_LABELS = {
  camber_front_change: { label: "Camber", unit: "°", icon: "↻" },
  tire_pressure_psi_change: { label: "Pressure", unit: "PSI", icon: "◎" },
  brake_bias_change: { label: "Bias", unit: "%", icon: "⬡" },
};

export default function RecourseCard({ currentSetup, onRecourseResult }) {
  const [targetLaps, setTargetLaps] = useState(22);
  const [penaltyWeight, setPenaltyWeight] = useState(100);
  const [nRestarts, setNRestarts] = useState(5);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await fetchRecourse(currentSetup, targetLaps, penaltyWeight, nRestarts);
      setResult(data);
      onRecourseResult(data);
    } catch (err) {
      if (err.status === 422 && err.detail?.error === "unreachable_target") {
        setError({
          type: "unreachable",
          message: err.detail.message,
          targetLaps: err.detail.target_laps,
          bestAchievable: err.detail.best_achievable_laps,
          bestSetup: err.detail.best_setup_within_bounds,
        });
      } else {
        setError({
          type: "generic",
          message: err.message || "Optimization failed",
        });
      }
      onRecourseResult(null);
    } finally {
      setLoading(false);
    }
  };

  const badge = result?.optimization_diagnostics
    ? stabilityBadge(result.optimization_diagnostics.stability)
    : null;

  return (
    <div className="pitwall-card animate-fade-in">
      <div className="pitwall-card-header flex items-center gap-2">
        <Target className="w-3.5 h-3.5" />
        Recourse Engine
      </div>

      <div className="p-5 space-y-5">
        {/* Target Input */}
        <div>
          <label className="text-xs text-slate-400 font-medium block mb-2">
            Target Stint Length
          </label>
          <div className="flex items-center gap-3">
            <input
              id="target-laps-input"
              type="number"
              min={4}
              max={100}
              step={0.5}
              value={targetLaps}
              onChange={(e) => setTargetLaps(parseFloat(e.target.value) || 4)}
              className="flex-1 bg-pitwall-bg border border-pitwall-border rounded-lg px-3 py-2
                         text-sm font-mono text-slate-100 focus:outline-none focus:border-neon-cyan/50
                         focus:ring-1 focus:ring-neon-cyan/20 transition-colors"
            />
            <span className="text-xs font-mono text-slate-500">laps</span>
          </div>
        </div>

        {/* Advanced Params (collapsed) */}
        <details className="group">
          <summary className="text-[10px] text-slate-600 font-mono uppercase tracking-wider cursor-pointer hover:text-slate-400 transition-colors list-none flex items-center gap-1">
            <ChevronDown className="w-3 h-3 group-open:rotate-180 transition-transform" />
            Advanced Parameters
          </summary>
          <div className="mt-3 space-y-3 animate-slide-up">
            <div className="flex items-center gap-3">
              <label className="text-[10px] text-slate-500 font-mono w-28">Penalty Weight</label>
              <input
                type="number"
                min={1}
                max={1000}
                value={penaltyWeight}
                onChange={(e) => setPenaltyWeight(parseFloat(e.target.value) || 100)}
                className="flex-1 bg-pitwall-bg border border-pitwall-border rounded px-2 py-1
                           text-xs font-mono text-slate-300 focus:outline-none focus:border-neon-cyan/30"
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="text-[10px] text-slate-500 font-mono w-28">DE Restarts</label>
              <input
                type="number"
                min={1}
                max={20}
                value={nRestarts}
                onChange={(e) => setNRestarts(parseInt(e.target.value) || 5)}
                className="flex-1 bg-pitwall-bg border border-pitwall-border rounded px-2 py-1
                           text-xs font-mono text-slate-300 focus:outline-none focus:border-neon-cyan/30"
              />
            </div>
          </div>
        </details>

        {/* Optimize Button */}
        <button
          id="optimize-button"
          onClick={handleOptimize}
          disabled={loading}
          className="btn-primary w-full"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running Differential Evolution…
            </>
          ) : (
            <>
              <Zap className="w-4 h-4" />
              Compute Optimal Setup
            </>
          )}
        </button>

        {/* ====== RESULT DISPLAY ====== */}
        {result && (
          <div className="space-y-4 animate-slide-up">
            {/* Prediction Comparison */}
            <div className="bg-pitwall-bg rounded-lg p-4 border border-pitwall-border">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div>
                  <p className="text-[10px] text-slate-600 font-mono uppercase mb-1">Current</p>
                  <p className="text-lg font-mono font-bold text-slate-400 tabular-nums">
                    {result.original_prediction.toFixed(1)}
                  </p>
                  {result.original_prediction_uncertainty?.mae_laps != null && (
                    <p className="text-[10px] font-mono text-slate-600 mt-0.5">
                      ±{result.original_prediction_uncertainty.mae_laps.toFixed(2)}
                    </p>
                  )}
                </div>
                <div className="flex items-center justify-center">
                  <ArrowRightLeft className="w-4 h-4 text-slate-600" />
                </div>
                <div>
                  <p className="text-[10px] text-neon-emerald/60 font-mono uppercase mb-1">Optimized</p>
                  <p className="text-lg font-mono font-bold text-neon-emerald tabular-nums">
                    {result.new_prediction.toFixed(1)}
                  </p>
                  {result.new_prediction_uncertainty?.mae_laps != null && (
                    <p className="text-[10px] font-mono text-slate-600 mt-0.5">
                      ±{result.new_prediction_uncertainty.mae_laps.toFixed(2)}
                    </p>
                  )}
                </div>
              </div>
              <div className="mt-3 text-center">
                <span className="text-xs font-mono text-slate-500">laps gained: </span>
                <span className="text-sm font-mono font-bold text-neon-emerald tabular-nums">
                  +{(result.new_prediction - result.original_prediction).toFixed(1)}
                </span>
              </div>
            </div>

            {/* Setup Changes (Mechanical Clicks) */}
            <div>
              <p className="text-[10px] text-slate-500 font-mono uppercase tracking-wider mb-2 flex items-center gap-1.5">
                <Wrench className="w-3 h-3" />
                Setup Clicks Required
              </p>
              <div className="space-y-2">
                {Object.entries(result.setup_changes).map(([key, value]) => {
                  const meta = SETUP_LABELS[key];
                  if (!meta) return null;
                  return (
                    <div
                      key={key}
                      className="flex items-center justify-between bg-pitwall-bg rounded-lg px-3 py-2 border border-pitwall-border"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-sm">{meta.icon}</span>
                        <span className="text-xs text-slate-400">{meta.label}</span>
                      </div>
                      <span
                        className={`font-mono text-sm font-bold tabular-nums ${deltaClass(value)}`}
                      >
                        {formatDelta(value)} {meta.unit}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Stability Badge + Diagnostics Toggle */}
            {badge && (
              <div>
                <button
                  onClick={() => setShowDiagnostics(!showDiagnostics)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-lg bg-pitwall-bg border border-pitwall-border hover:border-pitwall-border-bright transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <div
                      className={`neon-badge border ${badge.color}`}
                    >
                      <badge.icon className="w-3 h-3" />
                      {badge.label}
                    </div>
                    <span className="text-[10px] text-slate-600 font-mono">
                      {result.optimization_diagnostics.n_restarts} restarts
                    </span>
                  </div>
                  {showDiagnostics ? (
                    <ChevronUp className="w-3.5 h-3.5 text-slate-600" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5 text-slate-600" />
                  )}
                </button>

                {showDiagnostics && (
                  <div className="mt-2 bg-pitwall-bg rounded-lg px-3 py-2 border border-pitwall-border space-y-1 animate-slide-up">
                    {[
                      ["Restart Spread", result.optimization_diagnostics.restart_spread.toFixed(4)],
                      ["Penalty Weight", result.optimization_diagnostics.penalty_weight_used],
                      ["Best Objective", result.optimization_diagnostics.best_objective_value.toFixed(4)],
                      ["P95 Error", result.new_prediction_uncertainty?.p95_error_laps?.toFixed(2) ?? "—"],
                      ["Region Samples", result.new_prediction_uncertainty?.region_sample_size ?? "—"],
                    ].map(([label, value]) => (
                      <div key={label} className="flex justify-between text-[11px] font-mono">
                        <span className="text-slate-600">{label}</span>
                        <span className="text-slate-400">{value}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ====== ERROR DISPLAY ====== */}
        {error && (
          <div className="animate-slide-up">
            {error.type === "unreachable" ? (
              <div className="bg-neon-crimson/5 border border-neon-crimson/20 rounded-lg p-4 space-y-3">
                <div className="flex items-start gap-2.5">
                  <XCircle className="w-4 h-4 text-neon-crimson mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-semibold text-neon-crimson">
                      Unreachable Target
                    </p>
                    <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                      Target of{" "}
                      <span className="font-mono text-slate-300">
                        {error.targetLaps}
                      </span>{" "}
                      laps exceeds the physical capability of available setup bounds.
                    </p>
                  </div>
                </div>

                <div className="bg-pitwall-bg rounded-lg p-3 border border-pitwall-border">
                  <p className="text-[10px] text-slate-600 font-mono uppercase mb-2">
                    Best Achievable Ceiling
                  </p>
                  <p className="text-xl font-mono font-bold text-neon-amber tabular-nums">
                    {error.bestAchievable.toFixed(1)}
                    <span className="text-sm text-slate-500 ml-1">laps</span>
                  </p>
                  {error.bestSetup && (
                    <div className="mt-2 pt-2 border-t border-pitwall-border space-y-1">
                      {Object.entries(error.bestSetup).map(([key, val]) => (
                        <div key={key} className="flex justify-between text-[11px] font-mono">
                          <span className="text-slate-600">{key.replace(/_/g, " ")}</span>
                          <span className="text-slate-400">{val.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => {
                    setTargetLaps(Math.floor(error.bestAchievable));
                    setError(null);
                  }}
                  className="btn-danger w-full text-xs"
                >
                  <Target className="w-3.5 h-3.5" />
                  Set target to {Math.floor(error.bestAchievable)} laps
                </button>
              </div>
            ) : (
              <div className="bg-neon-crimson/5 border border-neon-crimson/20 rounded-lg p-4 flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-neon-amber mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-neon-amber">Optimization Error</p>
                  <p className="text-xs text-slate-400 mt-1">{error.message}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

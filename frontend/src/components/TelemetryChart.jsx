import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingDown, AlertTriangle, Info } from "lucide-react";

/**
 * Generates a synthetic tire degradation curve.
 * The backend predicts a single "laps until critical wear" value.
 * We model the wear trajectory as an exponential-ish ramp from 0% to 100%,
 * where 70% is the critical threshold and the predicted laps is where we hit it.
 *
 * Uncertainty bands (MAE/P95) are projected symmetrically around the critical lap mark,
 * then propagated backward through the wear curve as widening confidence corridors.
 */
function generateDegradationCurve(predictedLaps, uncertainty) {
  if (!predictedLaps || predictedLaps <= 0) return [];

  const maxLaps = Math.ceil(predictedLaps * 1.4);
  const dataPoints = [];

  const mae = uncertainty?.mae_laps || 0;
  const p95 = uncertainty?.p95_error_laps || 0;

  for (let lap = 0; lap <= maxLaps; lap++) {
    // Wear fraction: exponential-ish curve hitting 70% at predictedLaps
    const t = lap / predictedLaps;
    const wearPct = Math.min(100, 70 * Math.pow(t, 1.15));

    // Project uncertainty through the wear curve at this lap
    const tMae = lap / Math.max(1, predictedLaps + mae);
    const tMaeNeg = lap / Math.max(1, predictedLaps - mae);
    const tP95 = lap / Math.max(1, predictedLaps + p95);
    const tP95Neg = lap / Math.max(1, predictedLaps - p95);

    const wearMaeHigh = Math.min(100, 70 * Math.pow(tMaeNeg, 1.15));
    const wearMaeLow = Math.min(100, 70 * Math.pow(tMae, 1.15));
    const wearP95High = Math.min(100, 70 * Math.pow(tP95Neg, 1.15));
    const wearP95Low = Math.min(100, 70 * Math.pow(tP95, 1.15));

    dataPoints.push({
      lap,
      wear: parseFloat(wearPct.toFixed(1)),
      wearMaeRange: [
        parseFloat(wearMaeLow.toFixed(1)),
        parseFloat(wearMaeHigh.toFixed(1)),
      ],
      wearP95Range: [
        parseFloat(wearP95Low.toFixed(1)),
        parseFloat(wearP95High.toFixed(1)),
      ],
    });
  }

  return dataPoints;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const wear = payload.find((p) => p.dataKey === "wear");
  return (
    <div className="bg-pitwall-card border border-pitwall-border-bright rounded-lg px-3 py-2 shadow-xl">
      <p className="text-[10px] text-slate-500 font-mono uppercase mb-1">
        Lap {label}
      </p>
      {wear && (
        <p className="text-sm font-mono font-bold text-neon-cyan">
          {wear.value}% <span className="text-slate-500 text-xs">wear</span>
        </p>
      )}
    </div>
  );
}

export default function TelemetryChart({ prediction, uncertainty, recoursePrediction }) {
  const data = useMemo(
    () => generateDegradationCurve(prediction, uncertainty),
    [prediction, uncertainty]
  );

  const hasData = data.length > 0;
  const hasBands = uncertainty?.mae_laps != null;

  return (
    <div className="pitwall-card animate-fade-in flex flex-col overflow-hidden">
      <div className="pitwall-card-header flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingDown className="w-3.5 h-3.5" />
          Tire Degradation Model
        </div>
        {prediction != null && (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-0.5 bg-neon-cyan rounded" />
              <span className="text-[10px] text-slate-500">Wear Curve</span>
            </div>
            {hasBands && (
              <>
                <div className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2 bg-neon-cyan/20 rounded-sm" />
                  <span className="text-[10px] text-slate-500">MAE Band</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="w-2.5 h-2 bg-neon-cyan/8 rounded-sm" />
                  <span className="text-[10px] text-slate-500">P95 Band</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <div className="p-5">
        {!hasData ? (
          <div className="h-full min-h-[320px] flex flex-col items-center justify-center text-slate-600 gap-3">
            <TrendingDown className="w-10 h-10 opacity-30" />
            <p className="text-sm font-mono">Adjust setup sliders to generate predictions</p>
          </div>
        ) : (
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
                <defs>
                  <linearGradient id="wearGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#22D3EE" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#22D3EE" stopOpacity={0.02} />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1E293B"
                  vertical={false}
                />
                <XAxis
                  dataKey="lap"
                  stroke="#475569"
                  fontSize={10}
                  fontFamily="JetBrains Mono"
                  tickLine={false}
                  axisLine={{ stroke: "#1E293B" }}
                  label={{
                    value: "LAP",
                    position: "insideBottomRight",
                    offset: -5,
                    fontSize: 9,
                    fill: "#64748B",
                    fontFamily: "JetBrains Mono",
                  }}
                />
                <YAxis
                  stroke="#475569"
                  fontSize={10}
                  fontFamily="JetBrains Mono"
                  tickLine={false}
                  axisLine={{ stroke: "#1E293B" }}
                  domain={[0, 100]}
                  tickFormatter={(v) => `${v}%`}
                  label={{
                    value: "WEAR",
                    angle: -90,
                    position: "insideLeft",
                    offset: 20,
                    fontSize: 9,
                    fill: "#64748B",
                    fontFamily: "JetBrains Mono",
                  }}
                />

                <Tooltip content={<CustomTooltip />} />

                {/* P95 confidence band (widest) */}
                {hasBands && (
                  <Area
                    dataKey="wearP95Range"
                    stroke="none"
                    fill="#22D3EE"
                    fillOpacity={0.05}
                    isAnimationActive={false}
                  />
                )}

                {/* MAE confidence band */}
                {hasBands && (
                  <Area
                    dataKey="wearMaeRange"
                    stroke="none"
                    fill="#22D3EE"
                    fillOpacity={0.12}
                    isAnimationActive={false}
                  />
                )}

                {/* Main wear curve */}
                <Area
                  type="monotone"
                  dataKey="wear"
                  stroke="#22D3EE"
                  strokeWidth={2}
                  fill="url(#wearGradient)"
                  dot={false}
                  activeDot={{
                    r: 4,
                    stroke: "#22D3EE",
                    strokeWidth: 2,
                    fill: "#0B0F17",
                  }}
                />

                {/* 70% critical threshold */}
                <ReferenceLine
                  y={70}
                  stroke="#F87171"
                  strokeDasharray="6 4"
                  strokeWidth={1.5}
                  label={{
                    value: "CRITICAL 70%",
                    position: "right",
                    fontSize: 9,
                    fill: "#F87171",
                    fontFamily: "JetBrains Mono",
                  }}
                />

                {/* Predicted critical lap marker */}
                {prediction && (
                  <ReferenceLine
                    x={Math.round(prediction)}
                    stroke="#FBBF24"
                    strokeDasharray="4 3"
                    strokeWidth={1}
                    label={{
                      value: `L${Math.round(prediction)}`,
                      position: "top",
                      fontSize: 10,
                      fill: "#FBBF24",
                      fontFamily: "JetBrains Mono",
                      fontWeight: 700,
                    }}
                  />
                )}

                {/* Recourse target lap marker */}
                {recoursePrediction && (
                  <ReferenceLine
                    x={Math.round(recoursePrediction)}
                    stroke="#34D399"
                    strokeDasharray="4 3"
                    strokeWidth={1}
                    label={{
                      value: `L${Math.round(recoursePrediction)} (recourse)`,
                      position: "top",
                      fontSize: 10,
                      fill: "#34D399",
                      fontFamily: "JetBrains Mono",
                      fontWeight: 600,
                    }}
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Prediction readout strip */}
        {prediction != null && (
          <div className="mt-4 pt-4 border-t border-pitwall-border flex items-center justify-between animate-slide-up">
            <div className="flex items-center gap-4">
              <div>
                <p className="text-[10px] text-slate-500 font-mono uppercase">Predicted Stint</p>
                <p className="text-2xl font-mono font-bold text-neon-cyan tabular-nums">
                  {prediction.toFixed(1)}
                  <span className="text-sm text-slate-500 ml-1">laps</span>
                </p>
              </div>
              {hasBands && (
                <div className="pl-4 border-l border-pitwall-border">
                  <p className="text-[10px] text-slate-500 font-mono uppercase">Uncertainty</p>
                  <p className="text-sm font-mono text-slate-300">
                    ±{uncertainty.mae_laps.toFixed(2)}
                    <span className="text-slate-500 text-xs ml-1">MAE</span>
                    <span className="text-slate-600 mx-2">|</span>
                    ±{uncertainty.p95_error_laps.toFixed(2)}
                    <span className="text-slate-500 text-xs ml-1">P95</span>
                  </p>
                </div>
              )}
            </div>
            {recoursePrediction != null && (
              <div className="text-right">
                <p className="text-[10px] text-slate-500 font-mono uppercase">After Recourse</p>
                <p className="text-xl font-mono font-bold text-neon-emerald tabular-nums">
                  {recoursePrediction.toFixed(1)}
                  <span className="text-sm text-slate-500 ml-1">laps</span>
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

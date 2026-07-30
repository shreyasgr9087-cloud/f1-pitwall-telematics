import React, { useCallback, useEffect, useRef } from "react";
import { SlidersHorizontal } from "lucide-react";

/**
 * Slider definitions matching the backend's Pydantic SetupInput field validators.
 * Each entry maps directly to a Field(..., ge=, le=) constraint in predict.py.
 */
const SLIDER_CONFIG = [
  {
    key: "camber_front",
    label: "Front Camber",
    unit: "°",
    min: -4.5,
    max: -1.5,
    step: 0.1,
    color: "text-neon-cyan",
    description: "Negative camber angle",
  },
  {
    key: "tire_pressure_psi",
    label: "Tire Pressure",
    unit: "PSI",
    min: 20,
    max: 30,
    step: 0.1,
    color: "text-neon-emerald",
    description: "Cold inflation pressure",
  },
  {
    key: "brake_bias",
    label: "Brake Bias",
    unit: "%",
    min: 50,
    max: 65,
    step: 0.5,
    color: "text-neon-amber",
    description: "Front brake distribution",
  },
  {
    key: "driving_style_aggression",
    label: "Aggression",
    unit: "×",
    min: 0.8,
    max: 1.2,
    step: 0.01,
    color: "text-neon-crimson",
    description: "Driving style multiplier",
  },
  {
    key: "TrackTemp",
    label: "Track Temp",
    unit: "°C",
    min: 0,
    max: 70,
    step: 0.5,
    color: "text-neon-violet",
    description: "Circuit surface temperature",
  },
  {
    key: "AirTemp",
    label: "Air Temp",
    unit: "°C",
    min: -10,
    max: 50,
    step: 0.5,
    color: "text-slate-300",
    description: "Ambient air temperature",
  },
];

/**
 * Returns the percentage position of a value within [min, max].
 */
function pct(value, min, max) {
  return ((value - min) / (max - min)) * 100;
}

export default function SetupSliders({ setup, onChange, onDebouncedChange }) {
  const debounceRef = useRef(null);

  const handleSliderChange = useCallback(
    (key, rawValue) => {
      const value = parseFloat(rawValue);
      const updated = { ...setup, [key]: value };
      onChange(updated);

      // Debounced prediction trigger — 300ms after last interaction
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        onDebouncedChange(updated);
      }, 300);
    },
    [setup, onChange, onDebouncedChange]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <div className="pitwall-card animate-fade-in">
      <div className="pitwall-card-header flex items-center gap-2">
        <SlidersHorizontal className="w-3.5 h-3.5" />
        Setup Parameters
      </div>

      <div className="p-5 space-y-5">
        {SLIDER_CONFIG.map((slider) => {
          const value = setup[slider.key];
          const fillPct = pct(value, slider.min, slider.max);

          return (
            <div key={slider.key} className="group">
              {/* Label row */}
              <div className="flex items-baseline justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400 font-medium">
                    {slider.label}
                  </span>
                  <span className="text-[10px] text-slate-600 font-mono hidden group-hover:inline transition-opacity">
                    [{slider.min}, {slider.max}]
                  </span>
                </div>
                <span className={`font-mono text-sm font-bold tabular-nums ${slider.color}`}>
                  {value.toFixed(slider.step < 0.1 ? 2 : 1)}
                  <span className="text-[10px] text-slate-500 ml-0.5">
                    {slider.unit}
                  </span>
                </span>
              </div>

              {/* Slider with filled track illusion */}
              <div className="relative">
                <div className="absolute top-1/2 left-0 h-1.5 rounded-full bg-neon-cyan/20 -translate-y-1/2 pointer-events-none transition-all duration-150"
                  style={{ width: `${fillPct}%` }}
                />
                <input
                  id={`slider-${slider.key}`}
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={value}
                  onChange={(e) => handleSliderChange(slider.key, e.target.value)}
                  className="slider-track relative z-10"
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

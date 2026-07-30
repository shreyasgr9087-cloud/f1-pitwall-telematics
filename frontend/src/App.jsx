import React, { useState, useCallback } from "react";
import Header from "./components/Header";
import SetupSliders from "./components/SetupSliders";
import TelemetryChart from "./components/TelemetryChart";
import RecourseCard from "./components/RecourseCard";
import { fetchPrediction } from "./api/client";

/**
 * Default setup matching reasonable mid-range values
 * within the backend's Pydantic-validated bounds.
 */
const DEFAULT_SETUP = {
  camber_front: -3.0,
  tire_pressure_psi: 24.5,
  brake_bias: 56.0,
  driving_style_aggression: 1.0,
  TrackTemp: 35.0,
  AirTemp: 22.0,
};

export default function App() {
  const [setup, setSetup] = useState(DEFAULT_SETUP);
  const [prediction, setPrediction] = useState(null);
  const [uncertainty, setUncertainty] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [recoursePrediction, setRecoursePrediction] = useState(null);
  const [recourseUncertainty, setRecourseUncertainty] = useState(null);

  /**
   * Called by SetupSliders after debounce.
   * Fires a live prediction and surfaces regional uncertainty if available.
   */
  const handleDebouncedPrediction = useCallback(async (updatedSetup) => {
    setPredictionLoading(true);
    try {
      const data = await fetchPrediction(updatedSetup);
      setPrediction(data.predicted_laps);
      // Note: /predict does not return uncertainty — only /recourse does.
      // We keep the last recourse uncertainty if it's contextually relevant,
      // but clear the recourse result since the setup has changed.
      setRecoursePrediction(null);
      setRecourseUncertainty(null);
      setUncertainty(null);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setPredictionLoading(false);
    }
  }, []);

  /**
   * Called by RecourseCard when optimization succeeds.
   * Surfaces the new prediction alongside its empirical uncertainty.
   */
  const handleRecourseResult = useCallback((result) => {
    if (result) {
      setRecoursePrediction(result.new_prediction);
      setRecourseUncertainty(result.new_prediction_uncertainty);
      // Also update the original prediction's uncertainty now that we have it
      setUncertainty(result.original_prediction_uncertainty);
    } else {
      setRecoursePrediction(null);
      setRecourseUncertainty(null);
    }
  }, []);

  return (
    <div className="min-h-screen bg-pitwall-bg flex flex-col">
      <Header />

      <main className="flex-1 p-4 lg:p-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 lg:gap-6 max-w-[1600px] mx-auto">
          {/* Left Column: Controls */}
          <div className="lg:col-span-4 xl:col-span-3 space-y-4 lg:space-y-6">
            <SetupSliders
              setup={setup}
              onChange={setSetup}
              onDebouncedChange={handleDebouncedPrediction}
            />
            <RecourseCard
              currentSetup={setup}
              onRecourseResult={handleRecourseResult}
            />
          </div>

          {/* Right Column: Visualization */}
          <div className="lg:col-span-8 xl:col-span-9">
            <TelemetryChart
              prediction={prediction}
              uncertainty={uncertainty}
              recoursePrediction={recoursePrediction}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-pitwall-border px-6 py-3 flex items-center justify-between">
        <p className="text-[10px] font-mono text-slate-700">
          Surrogate Model: XGBoost on effective_wear_rate · Recourse: SciPy Differential Evolution
        </p>
        <p className="text-[10px] font-mono text-slate-700">
          Data anchored to 2023 FastF1 ambient conditions
        </p>
      </footer>
    </div>
  );
}

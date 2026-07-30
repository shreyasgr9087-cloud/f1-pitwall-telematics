<div align="center">

# F1 Pit-Wall Telematics
### Surrogate Recourse Engine

**A real-time tire degradation prediction and counterfactual setup optimization dashboard,<br>powered by XGBoost surrogate modeling and Differential Evolution recourse.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-Surrogate-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

*In motorsport, every tenth of a second is engineered — not discovered.*<br>
*This system brings that philosophy to tire strategy.*

</div>

---

## Table of Contents

- [The Problem & Solution](#the-problem--solution)
- [Physics & Mathematics](#physics--mathematics)
  - [Reciprocal Transform Target Engineering](#1-reciprocal-transform-target-engineering)
  - [Empirical Regional Uncertainty Quantification](#2-empirical-regional-uncertainty-quantification)
  - [Differential Evolution Recourse with L1 Penalty](#3-differential-evolution-recourse-with-l1-penalty)
  - [Graceful Edge-Case Handling](#4-graceful-edge-case-handling)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Local Setup & Installation](#local-setup--installation)
- [API Reference](#api-reference)
- [Author](#author)

---

## The Problem & Solution

### The Problem

During a Formula 1 race, the pit wall engineering team must make sub-second decisions about tire strategy. *When should we pit? Can we extend the stint by 3 more laps if we back off the brakes? What if we had started with 0.5 PSI more pressure?*

Running a full finite-element tire thermal simulation to answer these questions takes minutes — an eternity when your driver is on lap 34 of 53 with blistering compounds.

### The Solution

This project replaces that slow physics pipeline with a two-stage machine learning system:

| Stage | Component | What It Does |
|:---:|---|---|
| **1** | **XGBoost Surrogate Model** | Trained on 5,000 hybrid telemetry samples anchored to real 2023 FastF1 ambient weather data (Monza, Silverstone, Spa, Bahrain, Suzuka). Predicts tire degradation in **< 5ms** per inference — fast enough for real-time slider interaction. |
| **2** | **Counterfactual Recourse Engine** | Given a target stint length (e.g., *"I need 22 laps"*), computes the **mathematically minimal** setup adjustment required to achieve it. Uses SciPy Differential Evolution with multi-seed stability verification. |

The result is a dark-themed, high-fidelity pit-wall dashboard where a race engineer can adjust camber, pressure, and brake bias — and instantly see how many laps the tires will survive, backed by honest uncertainty bands and actionable recourse recommendations.

---

## Physics & Mathematics

### 1. Reciprocal Transform Target Engineering

This is the single most consequential ML architecture decision in the system.

**The naïve approach** would train XGBoost to predict `laps_until_critical_wear` directly. But laps are derived from wear rate via a reciprocal function:

$$\text{laps} = \frac{70}{\text{effective\_wear\_rate}}$$

The derivative of this transform reveals the problem:

$$\frac{d}{dw}\left(\frac{70}{w}\right) = -\frac{70}{w^2}$$

At low wear rates (long stints), a tiny regression error in the wear rate prediction gets **amplified quadratically** when transformed to laps. A model trained on laps directly exhibits severe **heteroscedasticity** — small errors at 10-lap stints, catastrophic errors at 35-lap stints.

**Our approach:** Train on `effective_wear_rate` directly. The regression target is smooth, uniformly distributed, and has well-behaved gradients. The backend deterministically converts the prediction back to laps via the reciprocal transform at serving time, with a `max(0.001, ...)` safeguard against division-by-zero edge cases.

```python
# backend/app/routers/predict.py — the transform at inference time
predicted_wear_rate = float(model.predict(input_data)[0])
predicted_wear_rate = max(0.001, predicted_wear_rate)  # safeguard
predicted_laps = float(70.0 / predicted_wear_rate)
```

**Result:** Holdout R² of **0.98+** with uniform residual distribution across all stint lengths.

---

### 2. Empirical Regional Uncertainty Quantification

Point estimates are dangerous. A prediction of *"18.5 laps"* without context is operationally meaningless — a race engineer needs to know *how much to trust it*.

Rather than fitting a parametric uncertainty model (which would impose distributional assumptions), this system computes **empirical residual statistics** from the holdout test set, binned by prediction region:

```
Region [−∞, Q25) laps:  n=250, MAE=0.49, P95=1.16
Region [Q25, Q50) laps:  n=250, MAE=0.52, P95=1.28
Region [Q50, Q75) laps:  n=250, MAE=0.58, P95=1.42
Region [Q75, +∞]  laps:  n=250, MAE=0.62, P95=1.58
```

These regional stats are embedded in the model's metadata sidecar (`surrogate_xgb_metadata.json`) and surfaced dynamically by the recourse API. The frontend renders them as **confidence corridors** on the degradation chart — the MAE band (tight, high-confidence) nested inside the P95 band (wider, worst-case).

This is **Option B uncertainty** — honest, empirical, and non-parametric.

---

### 3. Differential Evolution Recourse with L1 Penalty

When a race engineer says *"I need 22 laps,"* the recourse engine must find setup adjustments that are:

1. **Sufficient** — the predicted laps after adjustment must meet or exceed the target.
2. **Minimal** — large setup changes are mechanically risky; the optimizer should favor the smallest possible "clicks."

The objective function encodes both requirements:

$$\mathcal{L}(\mathbf{x}) = \underbrace{\lambda \cdot \max(0,\; t - \hat{f}(\mathbf{x}))^2}_{\text{Target Penalty}} + \underbrace{\sum_{i=1}^{3} \frac{|x_i - x_i^{(0)}|}{\Delta_i}}_{\text{L1 Normalized Distance}}$$

Where:
- $\mathbf{x} = [\text{camber},\; \text{pressure},\; \text{bias}]$ — the adjustable setup variables
- $x_i^{(0)}$ — current setup values (fixed reference point)
- $\Delta_i$ — the physical range of each variable (e.g., 10 PSI for pressure, 3° for camber), used to **normalize** the L1 distance so that all features contribute equally regardless of their unit scale
- $\hat{f}(\mathbf{x})$ — surrogate model prediction (laps) for setup $\mathbf{x}$
- $t$ — target laps
- $\lambda$ — penalty weight (default: 100)

**Why L1 over L2?** L1 norm produces **sparse** adjustments — the optimizer will zero out changes on features that don't help, resulting in recommendations like *"change only pressure by +2.1 PSI"* rather than touching all three parameters. This maps directly to how real pit-wall setup changes work: fewer clicks = fewer mistakes under pressure.

**Multi-Seed Stability:** Each optimization runs $n$ Differential Evolution restarts (default: 5) with different random seeds. The spread between solutions is measured as a normalized L1 distance. If the spread is < 0.1, the landscape is **stable** (single clear optimum). If > 0.3, the landscape is **unstable** (multiple local optima) — the UI flags this with an amber warning badge.

---

### 4. Graceful Edge-Case Handling

Not all targets are physically achievable. If a race engineer requests 40 laps on a 70°C track with aggressive driving, no setup within the mechanical bounds of the car can deliver it.

The recourse engine detects this by comparing the best optimized result against the target (with a 0.5-lap tolerance). When the target is provably unreachable, it raises a structured `UnreachableTargetError`:

```json
{
  "detail": {
    "error": "unreachable_target",
    "message": "Target of 50.0 laps is unreachable within setup bounds...",
    "target_laps": 50.0,
    "best_achievable_laps": 39.4,
    "best_setup_within_bounds": {
      "camber_front": -1.8,
      "tire_pressure_psi": 24.5,
      "brake_bias": 50.0
    }
  }
}
```

The frontend catches this HTTP `422` response and renders a dedicated error state showing:
- The maximum achievable stint length within physical bounds
- The optimal setup that achieves this ceiling
- A one-click button to re-target to the achievable maximum

This ensures the system **never silently fails** — it either delivers a valid recourse or mathematically proves why the target is impossible.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│                                                                 │
│  ┌──────────────┐  ┌───────────────────┐  ┌──────────────────┐  │
│  │ SetupSliders │  │  TelemetryChart   │  │  RecourseCard    │  │
│  │              │──│  • Degradation    │  │  • Target Input  │  │
│  │ Camber       │  │    Curve          │  │  • DE Optimizer  │  │
│  │ Pressure     │  │  • MAE/P95 Bands  │  │  • Click Deltas  │  │
│  │ Bias         │  │  • Critical 70%   │  │  • Stability     │  │
│  │ Aggression   │  │    Threshold      │  │    Badge         │  │
│  │ TrackTemp    │  │  • Recourse       │  │  • 422 Error     │  │
│  │ AirTemp      │  │    Marker         │  │    Handling      │  │
│  └──────┬───────┘  └───────────────────┘  └────────┬─────────┘  │
│         │              ▲          ▲                 │            │
│         │ debounced    │          │                 │            │
│         ▼ (300ms)      │          │                 ▼            │
│  ┌─────────────────────┴──────────┴────────────────────────┐    │
│  │                   src/api/client.js                      │    │
│  │         Native Fetch · Structured Error Handling         │    │
│  └──────────────────────┬──────────────────────────┬───────┘    │
└─────────────────────────┼──────────────────────────┼────────────┘
                          │ POST /api/predict/       │ POST /api/recourse/
                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI)                          │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────────────────────┐   │
│  │  predict.py      │    │  recourse.py                     │   │
│  │                  │    │                                  │   │
│  │  XGBoost.predict │    │  TelemetryRecourseEngine         │   │
│  │  → wear_rate     │    │  → Differential Evolution        │   │
│  │  → 70/wear_rate  │    │  → Multi-seed restarts           │   │
│  │  → predicted_laps│    │  → L1 normalized objective       │   │
│  └────────┬─────────┘    │  → UnreachableTargetError (422)  │   │
│           │              └──────────────────────────────────┘   │
│           ▼                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  surrogate_xgb.pkl  +  surrogate_xgb_metadata.json      │   │
│  │  • Model artifact   • Feature schema                    │   │
│  │  • R² = 0.98+       • Regional residual stats           │   │
│  │                     • Training data SHA256 hash          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  hybrid_physics_generator.py                             │   │
│  │  FastF1 2023 weather → Physics heuristics → 5000 rows   │   │
│  │  Tracks: Monza, Silverstone, Spa, Bahrain, Suzuka        │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Live Surrogate Prediction** | 6 setup sliders fire debounced (300ms) predictions against the XGBoost model. Sub-5ms inference. |
| **Tire Degradation Curve** | Recharts AreaChart renders a synthetic wear trajectory from 0% to 100%, with the predicted critical lap (70% threshold) marked. |
| **Regional Uncertainty Bands** | MAE and P95 confidence corridors projected through the wear function — not fake intervals, but empirical holdout residual statistics. |
| **Counterfactual Recourse** | *"Make my tires last 22 laps"* → engine computes minimal setup clicks via Differential Evolution. |
| **Optimization Diagnostics** | Stability badge (stable / moderate / unstable), restart spread, penalty weight, and best objective value — all surfaced in the UI. |
| **Unreachable Target Handling** | Impossible targets return a structured 422 error showing the physical ceiling and one-click re-targeting. |
| **Model Health Monitoring** | Header polls `/health` every 10s, displaying model R², training timestamp, and API connection status with animated pulse indicator. |
| **Dark Pit-Wall Aesthetic** | Deep slate (#0B0F17) background, translucent cards, neon status indicators (emerald/amber/crimson), JetBrains Mono telemetry readouts. |

---

## Tech Stack

### Backend
| Technology | Role |
|-----------|------|
| **Python 3.11+** | Runtime |
| **FastAPI** | Async API framework with Pydantic validation |
| **XGBoost** | Gradient boosted surrogate model |
| **SciPy** | `differential_evolution` optimizer for recourse |
| **FastF1** | Real Formula 1 telemetry and weather data ingestion |
| **Pandas / NumPy** | Data manipulation and numerical computation |

### Frontend
| Technology | Role |
|-----------|------|
| **React 18** | Component framework |
| **Vite** | Build tooling and HMR dev server |
| **Tailwind CSS 3** | Utility-first styling with custom pit-wall design tokens |
| **Recharts** | AreaChart with confidence bands and reference lines |
| **Lucide React** | Icon system |

---

## Local Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm

### 1. Clone the Repository

```bash
git clone https://github.com/shreyasgr9087-cloud/sim-racing-telematics.git
cd sim-racing-telematics
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Generate Data & Train the Model

```bash
# Generate hybrid telemetry dataset (anchored to FastF1 2023 weather)
python backend/app/hybrid_physics_generator.py

# Train the XGBoost surrogate model
python backend/app/train_model.py
```

You should see output confirming:
- Dataset saved to `backend/app/data/hybrid_stint_telemetry.csv` (5000 rows)
- Model saved to `backend/app/models/surrogate_xgb.pkl`
- Metadata sidecar saved to `backend/app/models/surrogate_xgb_metadata.json`
- Holdout R² ≈ 0.98

### 4. Start the Backend

```bash
python -m uvicorn app.main:app --reload --app-dir backend --port 8001
```

Verify at: [http://127.0.0.1:8001/health](http://127.0.0.1:8001/health)

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: [http://localhost:5173](http://localhost:5173)

> **CORS Note:** The backend explicitly allows origins on ports `5173` and `5174`. If Vite auto-increments to a different port (because 5173 is occupied), either kill the lingering process or add the new port to `allow_origins` in `backend/app/main.py`.

---

## API Reference

### `GET /health`
Returns model readiness, training timestamp, and holdout R².

### `POST /api/predict/`
Predicts tire stint length from 6 setup parameters.

**Request:**
```json
{
  "camber_front": -3.0,
  "tire_pressure_psi": 24.5,
  "brake_bias": 56.0,
  "driving_style_aggression": 1.0,
  "TrackTemp": 35.0,
  "AirTemp": 22.0
}
```

**Response (200):**
```json
{
  "predicted_laps": 18.52,
  "model_trained_at": "2026-07-30T10:42:54+00:00"
}
```

### `POST /api/recourse/`
Computes optimal setup adjustments to reach a target stint length.

**Request:**
```json
{
  "current_setup": {
    "camber_front": -3.8,
    "tire_pressure_psi": 22.0,
    "brake_bias": 60.0,
    "driving_style_aggression": 1.1,
    "TrackTemp": 42.0,
    "AirTemp": 28.0
  },
  "target_laps": 22.0,
  "penalty_weight": 100.0,
  "n_restarts": 5
}
```

**Response (200):** Returns original/new predictions with uncertainty bands, setup change deltas, new setup values, and optimization diagnostics.

**Response (422):** Returns `UnreachableTargetError` with best achievable ceiling and optimal setup within physical bounds.

---

## Project Structure

```
sim-racing-telematics/
├── backend/
│   ├── app/
│   │   ├── data/
│   │   │   └── hybrid_stint_telemetry.csv       # 5000-row training dataset
│   │   ├── models/
│   │   │   ├── surrogate_xgb.pkl                # Trained XGBoost model
│   │   │   └── surrogate_xgb_metadata.json      # Schema, residuals, hash
│   │   ├── routers/
│   │   │   ├── predict.py                        # /api/predict/ endpoint
│   │   │   └── recourse.py                       # /api/recourse/ endpoint
│   │   ├── hybrid_physics_generator.py           # FastF1 data + physics model
│   │   ├── train_model.py                        # XGBoost training pipeline
│   │   ├── recourse_engine.py                    # DE optimization engine
│   │   └── main.py                               # FastAPI app + lifespan
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── api/client.js                         # Backend API client
    │   ├── components/
    │   │   ├── Header.jsx                        # Branding + health monitor
    │   │   ├── SetupSliders.jsx                  # 6 bound-validated sliders
    │   │   ├── TelemetryChart.jsx                # Degradation curve + bands
    │   │   └── RecourseCard.jsx                  # Recourse UI + error states
    │   ├── App.jsx                               # Layout + state management
    │   └── index.css                             # Tailwind + pit-wall theme
    ├── tailwind.config.js
    ├── vite.config.js
    └── package.json
```

---

<div align="center">

## Author

**Shreyas**

[![GitHub](https://img.shields.io/badge/GitHub-shreyasgr9087--cloud-181717?style=flat-square&logo=github)](https://github.com/shreyasgr9087-cloud)

---

*Built with precision. Engineered for the pit wall.*

</div>

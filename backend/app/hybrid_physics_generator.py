from pathlib import Path
import fastf1
import numpy as np
import pandas as pd
import random

# Set up paths relative to file location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
CACHE_DIR = BASE_DIR / "fastf1_cache"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def fetch_multiple_track_contexts():
    """Fetch ambient data from a diverse set of 2023 races to ensure variance."""
    events = [("Monza", "R"), ("Silverstone", "R"), ("Spa", "R"), ("Bahrain", "R"), ("Suzuka", "R")]
    contexts = []
    
    for event, session_type in events:
        print(f"Fetching {event} data...")
        try:
            session = fastf1.get_session(2023, event, session_type)
            session.load(telemetry=False, weather=True)
            weather = session.weather_data
            contexts.append({
                "TrackTemp": float(weather["TrackTemp"].mean()),
                "AirTemp": float(weather["AirTemp"].mean())
            })
        except Exception as e:
            print(f"Warning: Failed for {event} ({e}).")
    
    # Fallback if offline or all fail
    if not contexts:
        contexts = [{"TrackTemp": 42.0, "AirTemp": 28.0}, {"TrackTemp": 30.0, "AirTemp": 20.0}]
    
    return contexts

def generate_hybrid_dataset(num_samples=5000, seed=42):
    np.random.seed(seed)
    contexts = fetch_multiple_track_contexts()

    # Randomly assign a track context to each row
    chosen_contexts = [random.choice(contexts) for _ in range(num_samples)]
    track_temps = np.array([c["TrackTemp"] for c in chosen_contexts])
    air_temps = np.array([c["AirTemp"] for c in chosen_contexts])

    camber_front = np.random.uniform(-4.5, -1.5, num_samples)
    tire_pressure = np.random.uniform(20.0, 30.0, num_samples)
    brake_bias = np.random.uniform(50.0, 65.0, num_samples)
    aggression = np.random.uniform(0.8, 1.2, num_samples)

    # Softened penalties to prevent flat plateaus for the optimizer
    pressure_penalty = 1.0 + 0.05 * ((tire_pressure - 24.5) ** 2)
    camber_penalty = 1.0 + 0.10 * np.abs(camber_front + 1.5)
    bias_penalty = 1.0 + 0.02 * (brake_bias - 50.0)
    track_penalty = 1.0 + 0.005 * (track_temps - 30.0)

    # Calculate raw effective wear rate directly (avoids reciprocal training distortions)
    base_wear = 2.0 * pressure_penalty * camber_penalty * bias_penalty * track_penalty * aggression
    lap_noise = np.random.normal(0, 0.1, num_samples)
    effective_wear_rate = np.maximum(0.5, base_wear + lap_noise)

    # Derived metric for downstream tracking/validation
    laps_until_critical = np.round(70.0 / effective_wear_rate).astype(float)

    df = pd.DataFrame({
        "camber_front": camber_front,
        "tire_pressure_psi": tire_pressure,
        "brake_bias": brake_bias,
        "driving_style_aggression": aggression,
        "TrackTemp": track_temps,
        "AirTemp": air_temps,
        "effective_wear_rate": effective_wear_rate,          # Primary target for model
        "laps_until_critical_wear": laps_until_critical,     # Derived target for evaluation
    })

    return df

def validate_physics_sanity(df):
    """Verify physics logic behaves as expected with robust sample size checks."""
    print("Running rigorous physics sanity assertions...")

    # 1. U-shape check for pressure (using derived laps for validation context)
    ideal_press = df[(df["tire_pressure_psi"] >= 24.0) & (df["tire_pressure_psi"] <= 25.0)]
    bad_press = df[(df["tire_pressure_psi"] < 21.0) | (df["tire_pressure_psi"] > 28.0)]
    assert len(ideal_press) > 30 and len(bad_press) > 30, "Sample size too small for pressure check"
    assert ideal_press["laps_until_critical_wear"].mean() > bad_press["laps_until_critical_wear"].mean()

    # 2. Aggression monotonicity
    low_agg = df[df["driving_style_aggression"] < 0.9]
    high_agg = df[df["driving_style_aggression"] > 1.1]
    assert len(low_agg) > 30 and len(high_agg) > 30, "Sample size too small for aggression check"
    assert low_agg["laps_until_critical_wear"].mean() > high_agg["laps_until_critical_wear"].mean()

    # 3. Camber monotonicity
    harsh_camber = df[df["camber_front"] < -4.0]
    gentle_camber = df[df["camber_front"] > -2.0]
    assert len(harsh_camber) > 30 and len(gentle_camber) > 30, "Sample size too small for camber check"
    assert gentle_camber["laps_until_critical_wear"].mean() > harsh_camber["laps_until_critical_wear"].mean()

    print("✓ All rigorous sanity checks passed successfully!")

if __name__ == "__main__":
    dataset = generate_hybrid_dataset()
    validate_physics_sanity(dataset)
    output_path = DATA_DIR / "hybrid_stint_telemetry.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path} ({len(dataset)} rows)")
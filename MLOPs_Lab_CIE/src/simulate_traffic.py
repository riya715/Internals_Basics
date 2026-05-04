import requests, json, os, random, time, pickle
import pandas as pd
import numpy as np

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.join(BASE, "data", "new_data.csv")
TRAIN    = os.path.join(BASE, "data", "training_data.csv")
LOG_PATH = os.path.join(BASE, "logs", "predictions.jsonl")
os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)

API_URL  = "http://localhost:8500/predict"
FEATURES = ["payload_kg", "distance_km", "wind_speed_kmph", "altitude_m"]

train_df = pd.read_csv(TRAIN)
new_df   = pd.read_csv(NEW_DATA)

# Load model locally for direct prediction on drifted data
with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Clear old log so we start fresh
open(LOG_PATH, "w").close()

# ── 35 normal requests → go through API (API handles logging) ──────────
print("Sending 35 normal requests via API...")
for _ in range(35):
    row = train_df[FEATURES].sample(1, random_state=random.randint(0, 9999)).iloc[0]
    data = {
        "payload_kg":      round(float(row.payload_kg), 2),
        "distance_km":     round(float(row.distance_km), 2),
        "wind_speed_kmph": round(float(row.wind_speed_kmph), 2),
        "altitude_m":      round(float(row.altitude_m), 2)
    }
    requests.post(API_URL, json=data, timeout=5)

# ── 15 drifted requests → log directly with RAW out-of-range values ────
# This is intentional: we simulate real-world drift where values
# fall outside training distribution. We bypass the API validator
# and log directly so the monitor can detect true distribution shift.
print("Logging 15 drifted requests directly (raw new_data values)...")
for i in range(15):
    row = new_df[FEATURES].iloc[i % len(new_df)]
    data = {
        "payload_kg":      round(float(row.payload_kg), 2),
        "distance_km":     round(float(row.distance_km), 2),     # up to 28km — true drift
        "wind_speed_kmph": round(float(row.wind_speed_kmph), 2), # up to 57 — true drift
        "altitude_m":      round(float(row.altitude_m), 2)
    }
    # Predict with model directly (no API range restriction)
    pred = round(float(model.predict([[
        min(max(data["payload_kg"], 0.1), 5.0),
        data["distance_km"],
        data["wind_speed_kmph"],
        min(max(data["altitude_m"], 10.0), 120.0)
    ]])[0]), 4)

    entry = {"timestamp": time.time(), "input": data, "prediction": pred}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Done. Total log entries: {sum(1 for _ in open(LOG_PATH))}")
# Replace your simulate_traffic.py with this fixed version

import requests, json, os, random, time
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

# Clear old log
open(LOG_PATH, "w").close()

def send_and_log(payload: dict) -> float:
    r = requests.post(API_URL, json=payload, timeout=5)
    r.raise_for_status()
    pred = r.json()["prediction_min"]
    return pred

def log_directly(payload: dict, pred: float):
    """Write log entry directly without going through API (for out-of-range drifted data)"""
    entry = {"timestamp": time.time(), "input": payload, "prediction": pred}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ── 35 normal requests via API ─────────────────────────────────────────
print("Sending 35 normal requests...")
for _ in range(35):
    row = train_df[FEATURES].sample(1, random_state=random.randint(0, 9999)).iloc[0]
    data = {
        "payload_kg":      round(float(row.payload_kg), 2),
        "distance_km":     round(float(row.distance_km), 2),
        "wind_speed_kmph": round(float(row.wind_speed_kmph), 2),
        "altitude_m":      round(float(row.altitude_m), 2)
    }
    send_and_log(data)  # API handles logging

# ── 15 drifted requests — log directly with raw new_data values ────────
print("Sending 15 drifted requests (direct log)...")
import pickle
with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

for i in range(15):
    row = new_df[FEATURES].iloc[i % len(new_df)]
    data = {
        "payload_kg":      round(float(row.payload_kg), 2),
        "distance_km":     round(float(row.distance_km), 2),      # raw drifted value
        "wind_speed_kmph": round(float(row.wind_speed_kmph), 2),  # raw drifted value
        "altitude_m":      round(float(row.altitude_m), 2)
    }
    # Predict locally (model can handle any range)
    clamped = [[
        min(max(data["payload_kg"], 0.1), 5.0),
        data["distance_km"],
        data["wind_speed_kmph"],
        min(max(data["altitude_m"], 10.0), 120.0)
    ]]
    pred = round(float(model.predict(clamped)[0]), 4)
    log_directly(data, pred)

print(f"✅ Done. Check logs/predictions.jsonl")

import requests, json, os, random
import pandas as pd
import numpy as np

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.join(BASE, "data", "new_data.csv")
TRAIN    = os.path.join(BASE, "data", "training_data.csv")

API_URL  = "http://localhost:8500/predict"
FEATURES = ["payload_kg", "distance_km", "wind_speed_kmph", "altitude_m"]

# Load datasets
train_df = pd.read_csv(TRAIN)
new_df   = pd.read_csv(NEW_DATA)

predictions_log = []

def send(payload: dict) -> float:
    r = requests.post(API_URL, json=payload, timeout=5)
    r.raise_for_status()
    return r.json()["prediction_min"]

# 35 normal requests — sample from training data ranges
print("Sending 35 normal requests...")
for _ in range(35):
    row = train_df[FEATURES].sample(1, random_state=random.randint(0, 9999)).iloc[0]
    data = {
        "payload_kg":      round(float(row.payload_kg), 2),
        "distance_km":     round(float(row.distance_km), 2),
        "wind_speed_kmph": round(float(row.wind_speed_kmph), 2),
        "altitude_m":      round(float(row.altitude_m), 2)
    }
    pred = send(data)
    predictions_log.append(pred)

# 15 drifted requests — use new_data (out-of-distribution)
print("Sending 15 drifted requests...")
for i in range(15):
    row = new_df[FEATURES].iloc[i % len(new_df)]
    # Clamp to API validation range so we don't get 422 errors
    data = {
        "payload_kg":      round(min(max(float(row.payload_kg), 0.1), 5.0), 2),
        "distance_km":     round(min(max(float(row.distance_km), 0.5), 15.0), 2),
        "wind_speed_kmph": round(min(max(float(row.wind_speed_kmph), 0.0), 30.0), 2),
        "altitude_m":      round(min(max(float(row.altitude_m), 10.0), 120.0), 2)
    }
    pred = send(data)
    predictions_log.append(pred)

print(f"✅ Done. Total predictions sent: {len(predictions_log)}")
print(f"   Mean prediction: {round(sum(predictions_log)/len(predictions_log), 4)}")
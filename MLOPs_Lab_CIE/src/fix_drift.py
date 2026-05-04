
# save as src/fix_drift.py and run it once (API must be running)
import json, os, time, pickle
import pandas as pd
import numpy as np

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW_DATA = os.path.join(BASE, "data", "new_data.csv")
TRAIN    = os.path.join(BASE, "data", "training_data.csv")
LOG_PATH = os.path.join(BASE, "logs", "predictions.jsonl")

train_df = pd.read_csv(TRAIN)
new_df   = pd.read_csv(NEW_DATA)

with open(os.path.join(BASE, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

import requests, random
open(LOG_PATH, "w").close()  # clear log

FEATURES = ["payload_kg", "distance_km", "wind_speed_kmph", "altitude_m"]

# 30 normal requests via API
print("Sending 30 normal requests...")
for _ in range(30):
    row = train_df[FEATURES].sample(1, random_state=random.randint(0,9999)).iloc[0]
    data = {k: round(float(row[k]), 2) for k in FEATURES}
    requests.post("http://localhost:8500/predict", json=data, timeout=5)

# 20 drifted requests logged directly with raw new_data values
print("Logging 20 drifted requests...")
for i in range(20):
    row = new_df[FEATURES].iloc[i % len(new_df)]
    data = {k: round(float(row[k]), 2) for k in FEATURES}
    pred = round(float(model.predict([[
        min(max(data["payload_kg"], 0.1), 5.0),
        data["distance_km"],
        data["wind_speed_kmph"],
        min(max(data["altitude_m"], 10.0), 120.0)
    ]])[0]), 4)
    entry = {"timestamp": time.time(), "input": data, "prediction": pred}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

print(f"Done. Total: {sum(1 for _ in open(LOG_PATH))} entries")
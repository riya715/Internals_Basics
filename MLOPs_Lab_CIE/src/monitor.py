
import json, os
import pandas as pd
import numpy as np

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH  = os.path.join(BASE, "logs", "predictions.jsonl")
TRAIN     = os.path.join(BASE, "data", "training_data.csv")
RESULT    = os.path.join(BASE, "results", "step3_s5.json")
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

# ── Load training reference stats ─────────────────────────────────────
train_df = pd.read_csv(TRAIN)
train_distance_mean    = round(train_df["distance_km"].mean(), 2)
train_wind_mean        = round(train_df["wind_speed_kmph"].mean(), 2)

# ── Load prediction logs ───────────────────────────────────────────────
records = []
with open(LOG_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

total_predictions = len(records)
all_preds     = [r["prediction"] for r in records]
mean_pred     = round(float(np.mean(all_preds)), 4)

live_distances = [r["input"]["distance_km"]     for r in records]
live_winds     = [r["input"]["wind_speed_kmph"]  for r in records]

live_distance_mean = round(float(np.mean(live_distances)), 2)
live_wind_mean     = round(float(np.mean(live_winds)), 2)

# ── Drift thresholds (from question paper) ────────────────────────────
DIST_THRESHOLD = 3.97
WIND_THRESHOLD = 6.14

dist_shift = round(abs(live_distance_mean - train_distance_mean), 2)
wind_shift = round(abs(live_wind_mean     - train_wind_mean),     2)

dist_alert = dist_shift > DIST_THRESHOLD
wind_alert = wind_shift > WIND_THRESHOLD
drift_detected = bool(dist_alert or wind_alert)
alerts = []
if dist_alert:
    alerts.append({
        "feature":    "distance_km",
        "train_mean": train_distance_mean,
        "live_mean":  live_distance_mean,
        "shift":      dist_shift,
        "threshold":  DIST_THRESHOLD,
        "status":     "ALERT"
    })
if wind_alert:
    alerts.append({
        "feature":    "wind_speed_kmph",
        "train_mean": train_wind_mean,
        "live_mean":  live_wind_mean,
        "shift":      wind_shift,
        "threshold":  WIND_THRESHOLD,
        "status":     "ALERT"
    })

result = {
    "total_predictions": total_predictions,
    "mean_prediction":   mean_pred,
    "drift_detected":    drift_detected,
    "alerts":            alerts
}

with open(RESULT, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"\n📄 Saved → {RESULT}")
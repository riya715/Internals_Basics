# run_test.py  (run once manually)
import requests, json, os

BASE   = os.path.dirname(os.path.abspath(__file__)) + "/.."
RESULT = os.path.join(BASE, "results", "step2_s4.json")

test_input = {"payload_kg": 3.0, "distance_km": 5.4, "wind_speed_kmph": 16.4, "altitude_m": 58.0}
resp = requests.post("http://localhost:8500/predict", json=test_input)
pred = resp.json()["prediction_min"]

result = {
    "health_endpoint": "/heartbeat",
    "predict_endpoint": "/predict",
    "port": 8500,
    "health_response": {"status": "operational", "service": "SkyDrop API"},
    "test_input": test_input,
    "prediction": pred
}
with open(RESULT, "w") as f:
    json.dump(result, f, indent=2)
print(f"✅ step2_s4.json saved. Prediction = {pred}")
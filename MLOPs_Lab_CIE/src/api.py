
import os, json, pickle, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "best_model.pkl")
LOG_PATH   = os.path.join(BASE, "logs", "predictions.jsonl")
RESULT     = os.path.join(BASE, "results", "step2_s4.json")
os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

# ── global model holder ────────────────────────────────────────────────
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model ONCE at startup — efficient, production-correct pattern
    with open(MODEL_PATH, "rb") as f:
        model_store["model"] = pickle.load(f)
    print("✅ Model loaded into memory")
    yield
    model_store.clear()

app = FastAPI(
    title="SkyDrop Flight Time API",
    description="Predicts drone flight time for medical supply delivery",
    version="1.0.0",
    lifespan=lifespan
)

# ── Pydantic schema with range validation ──────────────────────────────
class FlightInput(BaseModel):
    payload_kg:       float = Field(..., ge=0.1,  le=5.0,  description="Payload in kg (0.1–5)")
    distance_km:      float = Field(..., ge=0.5,  le=15.0, description="Distance in km (0.5–15)")
    wind_speed_kmph:  float = Field(..., ge=0.0,  le=30.0, description="Wind speed kmph (0–30)")
    altitude_m:       float = Field(..., ge=10.0, le=120.0,description="Altitude in meters (10–120)")

class PredictionResponse(BaseModel):
    prediction_min: float
    model_used:     str
    timestamp:      float

# ── endpoints ──────────────────────────────────────────────────────────
@app.get("/heartbeat")
def heartbeat():
    return {"status": "operational", "service": "SkyDrop API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: FlightInput):
    model = model_store.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = [[data.payload_kg, data.distance_km, data.wind_speed_kmph, data.altitude_m]]
    prediction = float(round(model.predict(features)[0], 4))
    ts = time.time()

    # ── log every prediction to JSONL ──────────────────────────────────
    log_entry = {
        "timestamp": ts,
        "input": data.model_dump(),
        "prediction": prediction
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return PredictionResponse(
        prediction_min=prediction,
        model_used="best_model",
        timestamp=ts
    )

# ── generate step2 result (called once after first test request) ───────
def save_step2_result(test_input, prediction):
    result = {
        "health_endpoint": "/heartbeat",
        "predict_endpoint": "/predict",
        "port": 8500,
        "health_response": {"status": "operational", "service": "SkyDrop API"},
        "test_input": test_input,
        "prediction": prediction
    }
    with open(RESULT, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8500, reload=False)
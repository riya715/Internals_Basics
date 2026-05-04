
import mlflow, json, os
from mlflow import MlflowClient

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META      = os.path.join(BASE, "models", "best_model_meta.json")
RESULT    = os.path.join(BASE, "results", "step4_s6.json")
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

mlflow.set_tracking_uri(f"file://{os.path.join(BASE, 'mlruns')}")

# ── Load best model metadata saved by train.py ─────────────────────────
with open(META) as f:
    meta = json.load(f)

run_id     = meta["run_id"]
model_name = "skydrop-flight-time-min-predictor"
model_uri  = f"runs:/{run_id}/model"

print(f"Registering model from run: {run_id}")

# ── Register model in MLflow Model Registry ────────────────────────────
client = MlflowClient()
reg    = mlflow.register_model(model_uri=model_uri, name=model_name)
version = int(reg.version)

print(f"✅ Registered '{model_name}' as version {version}")

# ── Save result ────────────────────────────────────────────────────────
result = {
    "registered_model_name": model_name,
    "version":               version,
    "run_id":                run_id,
    "source_metric":         "mae",
    "source_metric_value":   meta["mae"]
}

with open(RESULT, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print(f"📄 Saved → {RESULT}")
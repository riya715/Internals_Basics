
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json, os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── paths ──────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data", "training_data.csv")
MODEL_DIR = os.path.join(BASE, "models")
RESULT = os.path.join(BASE, "results", "step1_s1.json")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

FEATURES = ["payload_kg", "distance_km", "wind_speed_kmph", "altitude_m"]
TARGET   = "flight_time_min"
EXP_NAME = "skydrop-flight-time-min"

# ── data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── MLflow setup ───────────────────────────────────────────────────────
mlflow.set_tracking_uri(f"file://{os.path.join(BASE, 'mlruns')}")
mlflow.set_experiment(EXP_NAME)

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return round(mae, 4), round(rmse, 4)

results = []

# ── Model 1: SVR inside a scaling Pipeline ─────────────────────────────
# SVR is sensitive to feature scale — wrapping in Pipeline is correct MLOps practice
svr_params = {"svr__C": 10.0, "svr__epsilon": 0.5, "svr__kernel": "rbf"}
svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(C=10.0, epsilon=0.5, kernel="rbf"))])

with mlflow.start_run(run_name="SVR_Pipeline") as run:
    mlflow.set_tag("priority", "high")
    mlflow.log_params(svr_params)
    svr_pipe.fit(X_train, y_train)
    mae, rmse = evaluate(svr_pipe, X_test, y_test)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(svr_pipe, "model")
    svr_run_id = run.info.run_id
    results.append({"name": "SVR", "mae": mae, "rmse": rmse, "run_id": svr_run_id})
    print(f"SVR  → MAE={mae}  RMSE={rmse}")

# ── Model 2: RandomForest (tuned, no scaling needed) ───────────────────
rf_params = {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2, "random_state": 42}
rf_model = RandomForestRegressor(**rf_params)

with mlflow.start_run(run_name="RandomForest_Tuned") as run:
    mlflow.set_tag("priority", "high")
    mlflow.log_params(rf_params)
    rf_model.fit(X_train, y_train)
    mae, rmse = evaluate(rf_model, X_test, y_test)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(rf_model, "model")
    rf_run_id = run.info.run_id
    results.append({"name": "RandomForest", "mae": mae, "rmse": rmse, "run_id": rf_run_id})
    print(f"RF   → MAE={mae}  RMSE={rmse}")

# ── Pick best model & save it to disk ─────────────────────────────────
best = min(results, key=lambda x: x["mae"])
print(f"\n✅ Best model: {best['name']} (MAE={best['mae']})")

import mlflow.pyfunc, pickle
# Save best model artifact locally for API to load
best_model_obj = svr_pipe if best["name"] == "SVR" else rf_model
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model_obj, f)

# Save best model name for other scripts to reference
with open(os.path.join(MODEL_DIR, "best_model_meta.json"), "w") as f:
    json.dump({"name": best["name"], "run_id": best["run_id"], "mae": best["mae"]}, f)

# ── Write step1 result ─────────────────────────────────────────────────
step1 = {
    "experiment_name": EXP_NAME,
    "models": [{"name": r["name"], "mae": r["mae"], "rmse": r["rmse"]} for r in results],
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}
with open(RESULT, "w") as f:
    json.dump(step1, f, indent=2)

print(f"📄 Saved → {RESULT}")
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "delivery_time_model.joblib"
MLRUNS = ROOT / "mlruns"


def make_training_examples(seed: int = 42):
    """Generate a small deterministic training set inside the project."""
    rng = np.random.default_rng(seed)
    distance = rng.uniform(1, 30, 180)
    weight = rng.uniform(0.2, 20, 180)
    priority = rng.integers(0, 2, 180)
    noise = rng.normal(0, 2.0, 180)
    target = 18 + 1.8 * distance + 0.35 * weight - 7 * priority + noise
    X = np.column_stack([distance, weight, priority])
    return X, target


def train_model():
    X, y = make_training_examples()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"mae": float(mean_absolute_error(y_test, pred)), "r2": float(r2_score(y_test, pred))}
    try:
        import mlflow
        import mlflow.sklearn
        MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{MLRUNS}")
        mlflow.set_experiment("Experiment_8_Simple_MLOps")
        with mlflow.start_run(run_name="DeliveryTimeRandomForest"):
            mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 120, "max_depth": 8})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
    except ImportError:
        pass
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return metrics


def load_model():
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)

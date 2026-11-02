from flask import Flask, jsonify, request
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
from model import load_model


app = Flask(__name__)
REQUESTS = Counter("delivery_prediction_requests_total", "Prediction requests")
LATENCY = Histogram("delivery_prediction_latency_seconds", "Prediction latency")
ERRORS = Counter("delivery_prediction_errors_total", "Invalid prediction requests")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    started = time.perf_counter()
    try:
        data = request.get_json(silent=True) or {}
        distance, weight, priority = float(data["distance"]), float(data["weight"]), int(data.get("priority", 0))
        if not (1 <= distance <= 30 and 0.2 <= weight <= 20 and priority in (0, 1)):
            raise ValueError("Inputs are outside the supported range")
        value = float(load_model().predict([[distance, weight, priority]])[0])
        REQUESTS.inc()
        return jsonify({"delivery_time_minutes": round(value, 2)})
    except (KeyError, TypeError, ValueError) as exc:
        ERRORS.inc()
        return jsonify({"error": str(exc)}), 400
    finally:
        LATENCY.observe(time.perf_counter() - started)


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

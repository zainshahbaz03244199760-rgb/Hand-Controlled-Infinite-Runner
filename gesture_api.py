"""
gesture_api.py
----------------
REST API for gesture prediction.

Input:
{
    "landmarks": [63 values]
}

Output:
{
    "gesture_id": 2
}

Run:
python gesture_api.py
"""

import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

from config import (
    MODEL_PATH,
    LABEL_MAP_PATH,
    CONFIDENCE_THRESHOLD,
    SMOOTHING_WINDOW
)

from realtime_classifier import GesturePredictor

app = Flask(__name__)

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = {
        int(k): v
        for k, v in json.load(f).items()
    }

predictor = GesturePredictor(
    model,
    label_map,
    window=SMOOTHING_WINDOW,
    threshold=CONFIDENCE_THRESHOLD
)

print("Model loaded.")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if "landmarks" not in data:
        return jsonify(
            {
                "error":
                "Missing landmarks"
            }
        ), 400

    landmarks = np.array(
        data["landmarks"],
        dtype=np.float32
    )

    if len(landmarks) != 63:
        return jsonify(
            {
                "error":
                "Expected 63 landmark values"
            }
        ), 400

    class_id, confidence = predictor.predict(
        landmarks
    )

    if class_id is None:
        return jsonify(
            {
                "gesture_id": None,
                "confidence": confidence
            }
        )

    return jsonify(
        {
            "gesture_id": int(class_id),
            "confidence": confidence
        }
    )


@app.route("/reset", methods=["POST"])
def reset():

    predictor.reset()

    return jsonify(
        {
            "status": "reset"
        }
    )

@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "api_running": True
        }
    ), 200


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
import base64
import json
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from io import BytesIO

from flask import Flask, request, Response


MODEL_DIR = "/opt/ml/model"

app = Flask(__name__)


# =========================================================
# Load models ONCE when the container starts
# =========================================================

print("Loading models...")

raw_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "raw.keras")
)

filtered_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "filtered.keras")
)

edge_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "edges.keras")
)


with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "r"
) as f:
    CLASS_NAMES = json.load(f)


print("Models loaded successfully.")
print("Classes:", CLASS_NAMES)

print("Raw model input shape:", raw_model.input_shape)
print("Filtered model input shape:", filtered_model.input_shape)
print("Edge model input shape:", edge_model.input_shape)


# =========================================================
# Image decoding
# =========================================================

def decode_base64_image(image_b64):

    image_bytes = base64.b64decode(image_b64)

    # Keep RGB here because the raw and filtered models
    # require 3-channel input.
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    return np.array(image)


# =========================================================
# Prepare RGB input
# =========================================================

def prepare_rgb(image):

    image = Image.fromarray(image)

    image = image.resize(
        (200, 200),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # (200, 200, 3)
    #       ↓
    # (1, 200, 200, 3)
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# Prepare edge input
# =========================================================

def prepare_edge(image):

    # The image arrives from decode_base64_image()
    # as RGB. Convert it back to grayscale.
    image = Image.fromarray(
        image
    ).convert("L")

    image = image.resize(
        (200, 200),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # (200, 200)
    #     ↓
    # (200, 200, 1)
    image = np.expand_dims(
        image,
        axis=-1
    )

    # (200, 200, 1)
    #       ↓
    # (1, 200, 200, 1)
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# Top 3 predictions
# =========================================================

def top3_predictions(probabilities):

    probabilities = probabilities[0]

    indices = np.argsort(
        probabilities
    )[::-1][:3]

    results = []

    for index in indices:

        results.append({
            "class": CLASS_NAMES[int(index)],
            "probability": float(
                probabilities[index]
            )
        })

    return {
        "prediction": results[0]["class"],
        "top3": results
    }


# =========================================================
# Run all three models
# =========================================================

def predict_all(
    raw_image,
    filtered_image,
    edge_image
):

    # -----------------------------------------------------
    # Prepare inputs
    # -----------------------------------------------------

    raw_input = prepare_rgb(
        raw_image
    )

    filtered_input = prepare_rgb(
        filtered_image
    )

    edge_input = prepare_edge(
        edge_image
    )

    # -----------------------------------------------------
    # Print shapes for debugging
    # -----------------------------------------------------

    print(
        "Raw input shape:",
        raw_input.shape
    )

    print(
        "Filtered input shape:",
        filtered_input.shape
    )

    print(
        "Edge input shape:",
        edge_input.shape
    )

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    raw_probabilities = raw_model.predict(
        raw_input,
        verbose=0
    )

    filtered_probabilities = filtered_model.predict(
        filtered_input,
        verbose=0
    )

    edge_probabilities = edge_model.predict(
        edge_input,
        verbose=0
    )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    return {

        "raw": top3_predictions(
            raw_probabilities
        ),

        "filtered": top3_predictions(
            filtered_probabilities
        ),

        "edges": top3_predictions(
            edge_probabilities
        )
    }


# =========================================================
# Health check
# =========================================================

@app.route(
    "/ping",
    methods=["GET"]
)
def ping():

    return Response(
        response="OK",
        status=200,
        mimetype="text/plain"
    )


# =========================================================
# SageMaker inference endpoint
# =========================================================

@app.route(
    "/invocations",
    methods=["POST"]
)
def invocations():

    try:

        data = request.get_json()

        # -------------------------------------------------
        # Decode images
        # -------------------------------------------------

        raw_image = decode_base64_image(
            data["raw"]
        )

        filtered_image = decode_base64_image(
            data["filtered"]
        )

        edge_image = decode_base64_image(
            data["edges"]
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        predictions = predict_all(
            raw_image,
            filtered_image,
            edge_image
        )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return Response(
            response=json.dumps(
                predictions
            ),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:

        print(
            "Inference error:",
            repr(e)
        )

        return Response(
            response=json.dumps({
                "error": str(e)
            }),
            status=500,
            mimetype="application/json"
        )


# =========================================================
# Local/container startup
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080
    )
import os
import json
import logging
import joblib
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir):
    """
    Load model artifacts from /opt/ml/model.
    SageMaker extracts model.tar.gz into this directory.
    """
    model_path = os.path.join(model_dir, "churn_pipeline.joblib")
    threshold_path = os.path.join(model_dir, "threshold.json")

    model = joblib.load(model_path)

    with open(threshold_path, "r") as f:
        threshold_data = json.load(f)

    threshold = threshold_data.get("threshold", 0.5)

    logger.info("Model and threshold loaded successfully.")

    return {
        "model": model,
        "threshold": threshold,
    }


def input_fn(request_body, request_content_type):
    """
    Convert incoming JSON request into a pandas DataFrame.
    """

    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    data = json.loads(request_body)

    # Format 1: {"instances": [{...}, {...}]}
    if isinstance(data, dict) and "instances" in data:
        return pd.DataFrame(data["instances"])

    # Format 2: single record {"tenure": 12, "MonthlyCharges": 70.5, ...}
    if isinstance(data, dict):
        return pd.DataFrame([data])

    # Format 3: list of records [{...}, {...}]
    if isinstance(data, list):
        return pd.DataFrame(data)

    raise ValueError("Invalid input format")


def predict_fn(input_data, model_bundle):
    """
    Generate churn probability and final class prediction.
    """

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]

    logger.info(f"Received prediction request with {len(input_data)} rows.")

    probabilities = model.predict_proba(input_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = []

    for prob, pred in zip(probabilities, predictions):
        results.append(
            {
                "churn_probability": float(prob),
                "prediction": int(pred),
                "threshold": float(threshold),
                "label": "Churn" if int(pred) == 1 else "No Churn",
            }
        )

    return results


def output_fn(prediction, response_content_type):
    """
    Convert prediction result into JSON response.
    """

    if response_content_type != "application/json":
        raise ValueError(f"Unsupported response content type: {response_content_type}")

    return json.dumps({"predictions": prediction})

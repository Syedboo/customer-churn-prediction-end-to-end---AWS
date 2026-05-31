from __future__ import annotations

import joblib
import pandas as pd

from churn_prediction.config import EXPLAINER_BACKGROUND_PATH, METRICS_PATH, MODEL_PATH, THRESHOLD_PATH
from churn_prediction.data import normalize_columns
from churn_prediction.explain import (
    format_factor_details,
    heuristic_factor_details,
    heuristic_local_explanation,
    shap_local_explanation,
)
from churn_prediction.utils import read_json


def load_artifacts(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python -m churn_prediction.train` first."
        )
    model = joblib.load(model_path)
    threshold = read_json(THRESHOLD_PATH).get("threshold", 0.5) if THRESHOLD_PATH.exists() else 0.5
    metrics = read_json(METRICS_PATH) if METRICS_PATH.exists() else {}
    return model, float(threshold), metrics


def load_explainer_background(path=EXPLAINER_BACKGROUND_PATH) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return normalize_columns(pd.read_csv(path))


def risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def predict_dataframe(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    if model is None or threshold is None:
        model, artifact_threshold, _ = load_artifacts()
        threshold = artifact_threshold if threshold is None else threshold
    normalized = normalize_columns(df)
    probabilities = model.predict_proba(normalized)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    output = normalized.copy()
    output["churn_probability"] = probabilities.round(4)
    output["prediction"] = predictions
    output["risk_band"] = [risk_band(float(prob)) for prob in probabilities]
    return output


def predict_one(payload: dict, model=None, threshold: float | None = None) -> dict:
    frame = pd.DataFrame([payload])
    normalized = normalize_columns(frame)
    result = predict_dataframe(frame, model=model, threshold=threshold).iloc[0].to_dict()
    probability = float(result["churn_probability"])
    background = load_explainer_background()
    explanation_method = "heuristic"
    try:
        if model is None:
            model, _, _ = load_artifacts()
        if background is not None:
            factor_details = shap_local_explanation(model, normalized, background)
            top_factors = format_factor_details(factor_details)
            explanation_method = "shap"
        else:
            factor_details = heuristic_factor_details(normalized.iloc[0].to_dict(), probability)
            top_factors = heuristic_local_explanation(normalized.iloc[0].to_dict(), probability)
    except Exception:
        factor_details = heuristic_factor_details(normalized.iloc[0].to_dict(), probability)
        top_factors = heuristic_local_explanation(normalized.iloc[0].to_dict(), probability)
    return {
        "churn_probability": probability,
        "prediction": int(result["prediction"]),
        "risk_band": result["risk_band"],
        "top_factors": top_factors,
        "top_factor_details": factor_details,
        "explanation_method": explanation_method,
    }

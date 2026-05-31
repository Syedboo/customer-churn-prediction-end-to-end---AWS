from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from churn_prediction.config import FEATURE_DESCRIPTIONS, RANDOM_STATE


def save_permutation_importance(model, x_test, y_test, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring="average_precision",
    )
    importances = (
        pd.DataFrame({"feature": x_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(12)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importances["feature"][::-1], importances["importance"][::-1], color="#326273")
    ax.set_title("Permutation Importance")
    ax.set_xlabel("Decrease in average precision")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def heuristic_local_explanation(row: dict, probability: float) -> list[str]:
    """Stakeholder-friendly local explanation used when SHAP is not available."""
    factors: list[str] = []
    if row.get("complain", 0) == 1:
        factors.append("recent complaint raises churn risk")
    if row.get("tenure") is not None and float(row["tenure"]) <= 6:
        factors.append("short customer tenure raises churn risk")
    if row.get("satisfaction_score") is not None and float(row["satisfaction_score"]) <= 2:
        factors.append("low satisfaction score raises churn risk")
    if row.get("day_since_last_order") is not None and float(row["day_since_last_order"]) <= 3:
        factors.append("very recent low-order engagement pattern may indicate instability")
    if row.get("order_count") is not None and float(row["order_count"]) <= 2:
        factors.append("low order count limits relationship depth")
    if row.get("cashback_amount") is not None and float(row["cashback_amount"]) >= 220:
        factors.append("higher cashback history may reduce churn risk")
    if not factors:
        risk = "high" if probability >= 0.65 else "moderate" if probability >= 0.35 else "low"
        factors.append(f"overall feature pattern maps to {risk} churn risk")
    return factors[:4]


def describe_feature(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())


def base_feature_name(encoded_feature: str, original_features: list[str]) -> str:
    if encoded_feature in original_features:
        return encoded_feature
    matches = [
        feature
        for feature in original_features
        if encoded_feature == feature or encoded_feature.startswith(f"{feature}_")
    ]
    return max(matches, key=len) if matches else encoded_feature


def shap_local_explanation(model, row: pd.DataFrame, background: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Return local SHAP contributions for the positive churn class.

    This explains the fitted estimator after the pipeline preprocessor has transformed
    numeric and categorical inputs. One-hot columns are aggregated back to source
    feature names for stakeholder readability.
    """
    try:
        import shap
    except Exception as exc:
        raise ImportError("Install shap to enable SHAP local explanations.") from exc

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed_background = preprocessor.transform(background)
    transformed_row = preprocessor.transform(row)
    transformed_feature_names = preprocessor.get_feature_names_out()
    original_features = row.columns.tolist()

    explainer = shap.Explainer(estimator, transformed_background, feature_names=transformed_feature_names)
    explanation = explainer(transformed_row)
    values = explanation.values
    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        values = values[:, :, class_index]
    values = np.asarray(values)[0]

    aggregated: dict[str, float] = {}
    for encoded_name, contribution in zip(transformed_feature_names, values):
        feature = base_feature_name(str(encoded_name), original_features)
        aggregated[feature] = aggregated.get(feature, 0.0) + float(contribution)

    ranked = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    details = []
    for feature, contribution in ranked:
        direction = "increases churn risk" if contribution > 0 else "decreases churn risk"
        details.append(
            {
                "feature": feature,
                "description": describe_feature(feature),
                "direction": direction,
                "contribution": round(contribution, 4),
            }
        )
    return details


def format_factor_details(details: list[dict]) -> list[str]:
    return [
        f"{item['description']} {item['direction']} ({item['contribution']:+.4f})"
        for item in details
    ]


def heuristic_factor_details(row: dict, probability: float) -> list[dict]:
    details = []
    for factor in heuristic_local_explanation(row, probability):
        details.append(
            {
                "feature": "rule_based_signal",
                "description": factor,
                "direction": "increases churn risk" if "raises" in factor or "high" in factor else "mixed",
                "contribution": None,
            }
        )
    return details

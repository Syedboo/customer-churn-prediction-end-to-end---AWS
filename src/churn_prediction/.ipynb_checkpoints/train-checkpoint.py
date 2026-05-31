from __future__ import annotations

import argparse
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_prediction.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    THRESHOLD_PATH,
)
from churn_prediction.data import load_dataset, split_features_target
from churn_prediction.evaluate import save_confusion_matrix, save_roc_pr_curves
from churn_prediction.explain import save_permutation_importance
from churn_prediction.features import build_preprocessor, infer_feature_types
from churn_prediction.utils import ensure_dirs, write_json


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    threshold: float
    metrics: dict


def get_candidate_models(y_train: pd.Series) -> dict:
    negative = max(int((y_train == 0).sum()), 1)
    positive = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negative / positive

    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        )
    except Exception:
        pass
    return candidates


def predict_proba_positive(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    decision = model.decision_function(x_test)
    return 1 / (1 + np.exp(-decision))


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.2, 0.8, 61):
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    preds = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, preds)), 4),
        "precision_churn": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall_churn": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1_churn": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(y_true, preds, output_dict=True, zero_division=0),
    }


def train(data_path: str | None = None) -> TrainedModel:
    ensure_dirs(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)
    df = load_dataset(data_path)
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    numeric_features, categorical_features = infer_feature_types(x_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    results: list[TrainedModel] = []
    for name, estimator in get_candidate_models(y_train).items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probabilities = predict_proba_positive(pipeline, x_test)
        threshold, _ = choose_threshold(y_test, probabilities)
        metrics = evaluate_predictions(y_test, probabilities, threshold)
        results.append(TrainedModel(name=name, pipeline=pipeline, threshold=threshold, metrics=metrics))

    best = max(results, key=lambda item: item.metrics["average_precision"])
    joblib.dump(best.pipeline, MODEL_PATH)
    write_json(THRESHOLD_PATH, {"threshold": best.threshold})
    write_json(
        METRICS_PATH,
        {
            "best_model": best.name,
            "feature_columns": x.columns.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "models": {result.name: result.metrics for result in results},
        },
    )
    best_probabilities = predict_proba_positive(best.pipeline, x_test)
    best_predictions = (best_probabilities >= best.threshold).astype(int)
    save_confusion_matrix(y_test, best_predictions, FIGURES_DIR / "confusion_matrix.png")
    save_roc_pr_curves(y_test, best_probabilities, FIGURES_DIR / "roc_pr_curves.png")
    save_permutation_importance(best.pipeline, x_test, y_test, FIGURES_DIR / "permutation_importance.png")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn prediction models.")
    parser.add_argument("--data", default=None, help="Optional path to a CSV/XLSX churn dataset.")
    args = parser.parse_args()
    best = train(args.data)
    print(f"Best model: {best.name}")
    print(f"Average precision: {best.metrics['average_precision']}")
    print(f"Recall churn: {best.metrics['recall_churn']}")


if __name__ == "__main__":
    main()

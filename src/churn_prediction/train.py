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
    EXPLAINER_BACKGROUND_PATH,
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
    excluded_features: list[str]


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


def artifact_path(base_path, artifact_suffix: str):
    if not artifact_suffix:
        return base_path
    return base_path.with_name(f"{base_path.stem}{artifact_suffix}{base_path.suffix}")


def save_explainer_background(x_train: pd.DataFrame, artifact_suffix: str = "") -> None:
    background_path = artifact_path(EXPLAINER_BACKGROUND_PATH, artifact_suffix)
    background = x_train.sample(n=min(100, len(x_train)), random_state=RANDOM_STATE)
    background.to_csv(background_path, index=False)


def remove_features(x: pd.DataFrame, excluded_features: list[str] | None = None) -> pd.DataFrame:
    if not excluded_features:
        return x
    missing = [feature for feature in excluded_features if feature not in x.columns]
    if missing:
        raise ValueError(f"Cannot exclude missing feature(s): {missing}. Available columns: {x.columns.tolist()}")
    return x.drop(columns=excluded_features)


def train(
    data_path: str | None = None,
    excluded_features: list[str] | None = None,
    artifact_suffix: str = "",
) -> TrainedModel:
    ensure_dirs(MODELS_DIR, REPORTS_DIR, FIGURES_DIR)
    df = load_dataset(data_path)
    x, y = split_features_target(df)
    x = remove_features(x, excluded_features)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    numeric_features, categorical_features = infer_feature_types(x_train)

    results: list[TrainedModel] = []
    for name, estimator in get_candidate_models(y_train).items():
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probabilities = predict_proba_positive(pipeline, x_test)
        threshold, _ = choose_threshold(y_test, probabilities)
        metrics = evaluate_predictions(y_test, probabilities, threshold)
        results.append(
            TrainedModel(
                name=name,
                pipeline=pipeline,
                threshold=threshold,
                metrics=metrics,
                excluded_features=excluded_features or [],
            )
        )

    best = max(results, key=lambda item: item.metrics["average_precision"])
    model_path = artifact_path(MODEL_PATH, artifact_suffix)
    threshold_path = artifact_path(THRESHOLD_PATH, artifact_suffix)
    metrics_path = artifact_path(METRICS_PATH, artifact_suffix)
    joblib.dump(best.pipeline, model_path)
    save_explainer_background(x_train, artifact_suffix)
    write_json(threshold_path, {"threshold": best.threshold})
    write_json(
        metrics_path,
        {
            "best_model": best.name,
            "artifact_suffix": artifact_suffix,
            "excluded_features": excluded_features or [],
            "feature_columns": x.columns.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "models": {result.name: result.metrics for result in results},
        },
    )
    best_probabilities = predict_proba_positive(best.pipeline, x_test)
    best_predictions = (best_probabilities >= best.threshold).astype(int)
    save_confusion_matrix(
        y_test, best_predictions, artifact_path(FIGURES_DIR / "confusion_matrix.png", artifact_suffix)
    )
    save_roc_pr_curves(
        y_test, best_probabilities, artifact_path(FIGURES_DIR / "roc_pr_curves.png", artifact_suffix)
    )
    save_permutation_importance(
        best.pipeline,
        x_test,
        y_test,
        artifact_path(FIGURES_DIR / "permutation_importance.png", artifact_suffix),
    )
    return best


def train_without_tenure(data_path: str | None = None) -> TrainedModel:
    return train(data_path=data_path, excluded_features=["tenure"], artifact_suffix="_without_tenure")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer churn prediction models.")
    parser.add_argument("--data", default=None, help="Optional path to a CSV/XLSX churn dataset.")
    parser.add_argument(
        "--without-tenure",
        action="store_true",
        help="Run a sensitivity model that excludes the tenure feature.",
    )
    args = parser.parse_args()
    if args.without_tenure:
        best = train_without_tenure(args.data)
    else:
        best = train(args.data)
    print(f"Best model: {best.name}")
    if best.excluded_features:
        print(f"Excluded features: {', '.join(best.excluded_features)}")
    print(f"Average precision: {best.metrics['average_precision']}")
    print(f"Recall churn: {best.metrics['recall_churn']}")


if __name__ == "__main__":
    main()

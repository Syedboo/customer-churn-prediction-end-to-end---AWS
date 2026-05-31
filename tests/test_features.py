from churn_prediction.data import load_dataset, split_features_target
from churn_prediction.features import build_preprocessor, infer_feature_types
from churn_prediction.train import remove_features


def test_preprocessor_transforms_sample_data():
    df = load_dataset()
    x, _ = split_features_target(df)
    numeric_features, categorical_features = infer_feature_types(x)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    transformed = preprocessor.fit_transform(x)
    assert transformed.shape[0] == len(x)
    assert transformed.shape[1] >= len(numeric_features)


def test_remove_features_drops_tenure():
    df = load_dataset()
    x, _ = split_features_target(df)
    without_tenure = remove_features(x, ["tenure"])
    assert "tenure" in x.columns
    assert "tenure" not in without_tenure.columns

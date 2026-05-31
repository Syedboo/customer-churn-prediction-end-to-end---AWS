from churn_prediction.data import load_dataset, split_features_target
from churn_prediction.predict import predict_one, risk_band
from churn_prediction.train import train


def test_risk_band_thresholds():
    assert risk_band(0.2) == "Low"
    assert risk_band(0.5) == "Medium"
    assert risk_band(0.8) == "High"


def test_predict_one_with_trained_sample_model():
    trained = train()
    df = load_dataset()
    x, _ = split_features_target(df)
    result = predict_one(x.iloc[0].to_dict(), model=trained.pipeline, threshold=trained.threshold)
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_band"] in ["Low", "Medium", "High"]
    assert result["top_factors"]

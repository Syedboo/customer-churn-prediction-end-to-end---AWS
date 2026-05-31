from fastapi.testclient import TestClient

from api.main import app
from churn_prediction.data import load_dataset, split_features_target
from churn_prediction.train import train


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_after_training():
    train()
    with TestClient(app) as client:
        df = load_dataset()
        x, _ = split_features_target(df)
        response = client.post("/predict", json=x.iloc[0].to_dict())
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["churn_probability"] <= 1
    assert payload["risk_band"] in ["Low", "Medium", "High"]

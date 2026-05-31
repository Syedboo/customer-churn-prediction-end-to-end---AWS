from churn_prediction.data import load_dataset, normalize_column_name, split_features_target


def test_normalize_column_name_handles_medium_dataset_names():
    assert normalize_column_name("CustomerID") == "customer_id"
    assert normalize_column_name("HourSpendOnApp") == "hour_spend_on_app"
    assert normalize_column_name("CashbackAmount") == "cashback_amount"


def test_load_sample_dataset_and_split():
    df = load_dataset()
    x, y = split_features_target(df)
    assert "Churn" in df.columns
    assert "Churn" not in x.columns
    assert "customer_id" not in x.columns
    assert y.isin([0, 1]).all()
    assert len(x) == len(y)

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "ecommerce_churn_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"
EXPLAINER_BACKGROUND_PATH = MODELS_DIR / "explainer_background.csv"

TARGET_COLUMN = "Churn"
ID_COLUMNS = {"customer_id", "customerid"}

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_DESCRIPTIONS = {
    "tenure": "Months since the customer joined.",
    "preferred_login_device": "Customer's usual login device.",
    "city_tier": "Commercial tier of the customer's city.",
    "warehouse_to_home": "Distance between warehouse and home.",
    "preferred_payment_mode": "Payment mode selected most often.",
    "gender": "Recorded customer gender.",
    "hour_spend_on_app": "Average hours spent on the app.",
    "number_of_device_registered": "Registered devices for the account.",
    "prefered_order_cat": "Most preferred order category.",
    "satisfaction_score": "Customer satisfaction score.",
    "marital_status": "Recorded marital status.",
    "number_of_address": "Number of saved addresses.",
    "complain": "Whether the customer complained recently.",
    "order_amount_hike_fromlast_year": "Order amount increase from last year.",
    "coupon_used": "Number of coupons used.",
    "order_count": "Number of orders.",
    "day_since_last_order": "Days since the most recent order.",
    "cashback_amount": "Cashback received by the customer.",
}

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from churn_prediction.config import ID_COLUMNS, RAW_DATA_DIR, SAMPLE_DATA_PATH, TARGET_COLUMN


def normalize_column_name(name: str) -> str:
    """Convert dataset column names into stable snake_case names."""
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    replacements = {
        "churn": "Churn",
        "customerid": "customer_id",
        "preferredlogindevice": "preferred_login_device",
        "preferredpaymentmode": "preferred_payment_mode",
        "hourspendonapp": "hour_spend_on_app",
        "numberofdeviceregistered": "number_of_device_registered",
        "preferedordercat": "prefered_order_cat",
        "satisfactionscore": "satisfaction_score",
        "maritalstatus": "marital_status",
        "numberofaddress": "number_of_address",
        "orderamounthikefromlastyear": "order_amount_hike_fromlast_year",
        "couponused": "coupon_used",
        "ordercount": "order_count",
        "daysincelastorder": "day_since_last_order",
        "cashbackamount": "cashback_amount",
    }
    return replacements.get(value, value)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df


def discover_data_file(raw_dir: Path = RAW_DATA_DIR) -> Path:
    candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    if candidates:
        return candidates[0]
    return SAMPLE_DATA_PATH


def read_excel_dataset(data_path: Path, sheet_name: int | str = 1) -> pd.DataFrame:
    """Read the usable customer table from the Kaggle workbook's second sheet."""
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    return normalize_columns(df)



def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path else discover_data_file()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        df = read_excel_dataset(data_path, sheet_name=1)
    else:
        df = pd.read_csv(data_path)
        df = normalize_columns(df)

    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' after column normalization.")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    unique_targets = set(pd.Series(df[TARGET_COLUMN]).dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError("Target column must be binary encoded as 0/1.")


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    removable = [col for col in df.columns if col in ID_COLUMNS or col == TARGET_COLUMN]
    x = df.drop(columns=removable)
    y = df[TARGET_COLUMN].astype(int)
    return x, y

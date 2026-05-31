from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    tenure: float = Field(..., ge=0)
    preferred_login_device: str
    city_tier: int = Field(..., ge=1, le=3)
    warehouse_to_home: float = Field(..., ge=0)
    preferred_payment_mode: str
    gender: str
    hour_spend_on_app: float = Field(..., ge=0)
    number_of_device_registered: int = Field(..., ge=0)
    prefered_order_cat: str
    satisfaction_score: int = Field(..., ge=1, le=5)
    marital_status: str
    number_of_address: int = Field(..., ge=0)
    complain: int = Field(..., ge=0, le=1)
    order_amount_hike_fromlast_year: float
    coupon_used: int = Field(..., ge=0)
    order_count: int = Field(..., ge=0)
    day_since_last_order: float = Field(..., ge=0)
    cashback_amount: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    prediction: int
    risk_band: str
    top_factors: list[str]
    top_factor_details: list[dict] = Field(default_factory=list)
    explanation_method: str = "heuristic"

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn_prediction.config import METRICS_PATH
from churn_prediction.predict import load_artifacts, predict_dataframe, predict_one
from churn_prediction.utils import read_json

st.set_page_config(page_title="Customer Churn AI", layout="wide")
st.title("Customer Churn Prediction")

try:
    model, threshold, metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model artifact not found. Run `python -m churn_prediction.train` first.")
    st.stop()

tab_single, tab_batch, tab_model = st.tabs(["Single customer", "Batch scoring", "Model evidence"])

with tab_single:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tenure = st.number_input("Tenure", min_value=0.0, value=6.0)
        city_tier = st.selectbox("City tier", [1, 2, 3], index=2)
        warehouse_to_home = st.number_input("Warehouse to home", min_value=0.0, value=18.0)
        preferred_login_device = st.selectbox("Preferred login device", ["Mobile Phone", "Phone", "Computer"])
        preferred_payment_mode = st.selectbox("Payment mode", ["Debit Card", "Credit Card", "UPI", "E wallet", "COD"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col_b:
        hour_spend_on_app = st.number_input("Hours on app", min_value=0.0, value=3.0)
        number_of_device_registered = st.number_input("Registered devices", min_value=0, value=4)
        prefered_order_cat = st.selectbox(
            "Preferred order category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery"]
        )
        satisfaction_score = st.select_slider("Satisfaction score", options=[1, 2, 3, 4, 5], value=2)
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        number_of_address = st.number_input("Saved addresses", min_value=0, value=5)
    with col_c:
        complain = st.selectbox("Recent complaint", [0, 1], index=1)
        order_amount_hike_fromlast_year = st.number_input("Order amount hike from last year", value=17.0)
        coupon_used = st.number_input("Coupons used", min_value=0, value=1)
        order_count = st.number_input("Order count", min_value=0, value=2)
        day_since_last_order = st.number_input("Days since last order", min_value=0.0, value=3.0)
        cashback_amount = st.number_input("Cashback amount", min_value=0.0, value=120.0)

    payload = {
        "tenure": tenure,
        "preferred_login_device": preferred_login_device,
        "city_tier": city_tier,
        "warehouse_to_home": warehouse_to_home,
        "preferred_payment_mode": preferred_payment_mode,
        "gender": gender,
        "hour_spend_on_app": hour_spend_on_app,
        "number_of_device_registered": number_of_device_registered,
        "prefered_order_cat": prefered_order_cat,
        "satisfaction_score": satisfaction_score,
        "marital_status": marital_status,
        "number_of_address": number_of_address,
        "complain": complain,
        "order_amount_hike_fromlast_year": order_amount_hike_fromlast_year,
        "coupon_used": coupon_used,
        "order_count": order_count,
        "day_since_last_order": day_since_last_order,
        "cashback_amount": cashback_amount,
    }
    if st.button("Score customer", type="primary"):
        prediction = predict_one(payload, model=model, threshold=threshold)
        st.metric("Churn probability", f"{prediction['churn_probability']:.1%}", prediction["risk_band"])
        st.write(f"Top factors ({prediction.get('explanation_method', 'heuristic')})")
        for factor in prediction["top_factors"]:
            st.write(f"- {factor}")
        if prediction.get("top_factor_details"):
            st.dataframe(pd.DataFrame(prediction["top_factor_details"]), use_container_width=True)

with tab_batch:
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
        scored = predict_dataframe(batch, model=model, threshold=threshold)
        st.dataframe(scored, use_container_width=True)
        st.download_button(
            "Download scored CSV",
            scored.to_csv(index=False),
            file_name="churn_scored_customers.csv",
            mime="text/csv",
        )

with tab_model:
    if METRICS_PATH.exists():
        metrics = read_json(METRICS_PATH)
    st.subheader("Selected model")
    st.json({"best_model": metrics.get("best_model"), "threshold": threshold})
    if metrics.get("models"):
        summary = pd.DataFrame(metrics["models"]).T[
            ["roc_auc", "average_precision", "balanced_accuracy", "precision_churn", "recall_churn", "f1_churn"]
        ]
        st.dataframe(summary, use_container_width=True)
    st.image("reports/figures/confusion_matrix.png", caption="Confusion matrix")
    st.image("reports/figures/roc_pr_curves.png", caption="ROC and precision-recall curves")

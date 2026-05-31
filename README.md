# Customer Churn Prediction AI

Production-style customer churn prediction project for an applied AI / university collaboration centre portfolio. The project turns an e-commerce churn use case into a reproducible ML system with EDA, training, evaluation, explainability, API deployment, dashboarding, Docker, tests, and stakeholder-facing documentation.

The project is inspired by Allan Ouko's Medium walkthrough on customer churn prediction, but upgrades the work into a modular, leakage-aware, deployment-ready system.

## Stakeholder Problem

Customer churn reduces recurring revenue and makes growth expensive. This project predicts the probability that a customer will churn so commercial, customer success, and product teams can prioritise retention interventions.

Example use cases:

- identify high-risk customers before cancellation
- prioritise retention calls, discounts, or service recovery
- understand common churn drivers by segment
- monitor customer experience signals such as complaints and satisfaction

## Architecture

```text
data -> validation -> preprocessing pipeline -> model comparison -> evaluation
                                     |              |
                                     v              v
                              saved artifact   explainability
                                     |
                          FastAPI + Streamlit dashboard
```

The same saved pipeline is used for batch scoring, API prediction, and the dashboard. This avoids training-serving skew.

## Project Structure

```text
customer-churn-ai/
├── api/                         # FastAPI service
├── dashboard/                   # Streamlit stakeholder dashboard
├── data/
│   ├── raw/                     # place Kaggle XLSX/CSV here
│   └── sample/                  # small runnable sample dataset
├── models/                      # generated model artifacts
├── notebooks/                   # EDA notebook
├── reports/figures/             # generated plots
├── src/churn_prediction/        # reusable ML package
├── tests/                       # pytest test suite
├── Dockerfile
├── model_card.md
└── interview_questions.md
```

## Dataset

Use the e-commerce churn dataset referenced in the Medium article. Put the downloaded `.xlsx` or `.csv` file in:

```bash
data/raw/
```

If no raw dataset is present, the code uses `data/sample/ecommerce_churn_sample.csv` so the project remains runnable.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For a lighter install without optional SHAP/XGBoost:

```bash
pip install -e .
```

## Run EDA

Open:

```text
notebooks/01_eda_customer_churn.ipynb
```

The notebook covers data quality, churn imbalance, segment-level churn rates, numeric relationships, and stakeholder hypotheses.

## Train Models

```bash
python -m churn_prediction.train
```

Or specify a dataset:

```bash
python -m churn_prediction.train --data data/raw/ECommerce_Dataset.xlsx
```

Run the tenure sensitivity experiment:

```bash
python -m churn_prediction.train --data "data/raw/E Commerce Dataset.xlsx" --without-tenure
```

This trains and evaluates the same candidate models after removing the `tenure` feature. It saves separate artifacts so the production model is not overwritten:

- `models/churn_pipeline_without_tenure.joblib`
- `models/metrics_without_tenure.json`
- `models/threshold_without_tenure.json`
- `reports/figures/confusion_matrix_without_tenure.png`
- `reports/figures/roc_pr_curves_without_tenure.png`
- `reports/figures/permutation_importance_without_tenure.png`

The training pipeline compares:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier
- XGBoostClassifier, when installed

Artifacts created:

- `models/churn_pipeline.joblib`
- `models/metrics.json`
- `models/threshold.json`
- `models/explainer_background.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_pr_curves.png`
- `reports/figures/permutation_importance.png`

## Evaluation Strategy

Accuracy is not the primary metric because churn is typically imbalanced. The project reports:

- ROC-AUC
- average precision / PR-AUC
- balanced accuracy
- churn precision
- churn recall
- churn F1
- confusion matrix

The selected model is chosen by average precision, which is useful when the positive churn class is relatively rare.

## Model Results

Training on the full raw Excel dataset selected `HistGradientBoostingClassifier` as the best model.

| Metric | Value |
|---|---:|
| Best model | HistGradientBoostingClassifier |
| Decision threshold | 0.33 |
| ROC-AUC | 0.9991 |
| Average precision | 0.9947 |
| Churn recall | 0.9789 |
| Churn precision | 0.9738 |
| Churn F1 | 0.9764 |

The confusion matrix on the holdout test set was `[[931, 5], [4, 186]]`, meaning the model correctly identified 186 churners, missed 4 churners, and incorrectly flagged 5 retained customers as churn risks.

Permutation importance shows that `tenure` is the strongest model driver, followed by `complain`, `number_of_address`, `cashback_amount`, and `satisfaction_score`. These are useful stakeholder signals, but they should be interpreted as predictive associations rather than causal proof.

Because the scores are extremely high, the next validation step is to check feature timing, duplicates, and leakage risk, then evaluate on an out-of-time holdout dataset before treating the model as production-ready.

To test whether the model depends too strongly on `tenure`, run the no-tenure sensitivity pipeline and compare `models/metrics.json` with `models/metrics_without_tenure.json`. If performance remains strong, the churn signal is distributed across several customer behaviour features. If performance drops sharply, tenure is carrying most of the predictive signal and should be carefully validated for feature timing.

## Run FastAPI

Train the model first, then run:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "tenure": 3,
  "preferred_login_device": "Mobile Phone",
  "city_tier": 3,
  "warehouse_to_home": 22,
  "preferred_payment_mode": "Debit Card",
  "gender": "Male",
  "hour_spend_on_app": 2,
  "number_of_device_registered": 4,
  "prefered_order_cat": "Mobile Phone",
  "satisfaction_score": 2,
  "marital_status": "Single",
  "number_of_address": 5,
  "complain": 1,
  "order_amount_hike_fromlast_year": 21,
  "coupon_used": 0,
  "order_count": 1,
  "day_since_last_order": 2,
  "cashback_amount": 115
}
```

Example response:

```json
{
  "churn_probability": 0.82,
  "prediction": 1,
  "risk_band": "High",
  "explanation_method": "shap",
  "top_factors": [
    "Months since the customer joined. increases churn risk (+0.4123)",
    "Whether the customer complained recently. increases churn risk (+0.1911)"
  ],
  "top_factor_details": [
    {
      "feature": "tenure",
      "description": "Months since the customer joined.",
      "direction": "increases churn risk",
      "contribution": 0.4123
    }
  ]
}
```

During training, the pipeline saves a small background sample at `models/explainer_background.csv`. At inference time, the API uses SHAP to calculate local feature contributions for the positive churn class. If SHAP is not installed or the explainer fails, the API falls back to rule-based local factors so prediction remains available.

## Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard supports:

- single-customer scoring
- batch CSV upload
- risk bands
- top local factors
- model metric summary

## Docker

Train locally first so the `models/` folder exists, then:

```bash
docker build -t customer-churn-api .
docker run -p 8000:8000 customer-churn-api
```

## Tests

```bash
pytest
```

The tests cover:

- data loading and schema validation
- feature preprocessing
- model prediction contract
- FastAPI health and prediction endpoints

## Responsible AI Notes

This model should support human decision-making, not automatically deny service or benefits. Features such as gender, city tier, and marital status may act as sensitive or proxy variables. Before production use, evaluate fairness across customer segments, calibrate probabilities, and monitor drift.

## Future Work

- add campaign expected-value optimisation
- add drift monitoring
- add CI pipeline
- add model registry integration
- add fairness reporting by customer segment

# Model Card: Customer Churn Prediction

## Model Details

- Project: Customer Churn Prediction AI
- Task: Binary classification
- Target: `Churn`, encoded as 1 for churn and 0 for retained
- Model family: selected from logistic regression, random forest, gradient boosting, and optional XGBoost
- Artifact: `models/churn_pipeline.joblib`

## Intended Use

The model estimates customer churn probability for an e-commerce business. It is intended to help stakeholders prioritise retention actions, service recovery, and customer engagement campaigns.

Appropriate users:

- data science teams
- customer success teams
- marketing analysts
- product analytics teams

## Out-of-Scope Use

The model should not be used to:

- make fully automated decisions that materially harm customers
- deny service, support, or fair treatment
- infer causal relationships without additional causal analysis
- generalise to unrelated industries without retraining and validation

## Training Data

The project expects the e-commerce churn dataset referenced in the accompanying Medium article. A small sample dataset is included for demonstration and tests only. Production conclusions should be based on the full dataset.

Key feature groups:

- tenure and engagement
- complaints and satisfaction
- order history
- cashback and coupon behaviour
- device, city, payment, and demographic attributes

## Evaluation

Primary selection metric:

- average precision, because churn is usually an imbalanced positive class

Reported metrics:

- ROC-AUC
- average precision
- balanced accuracy
- churn precision
- churn recall
- churn F1
- confusion matrix

### Evaluation Results on Full Raw Dataset

| Metric | Value |
|---|---:|
| Best model | HistGradientBoostingClassifier |
| Decision threshold | 0.33 |
| ROC-AUC | 0.9991 |
| Average precision | 0.9947 |
| Balanced accuracy | 0.9868 |
| Churn precision | 0.9738 |
| Churn recall | 0.9789 |
| Churn F1 | 0.9764 |

Confusion matrix:

```text
[[931, 5],
 [4, 186]]
```

The model caught 186 of 190 churners in the holdout test set and incorrectly flagged 5 retained customers. Permutation importance indicates that tenure is the strongest driver, followed by complaint history, number of addresses, cashback amount, and satisfaction score.

## Inference-Time Explainability

The prediction service returns local feature explanations with each prediction. During training, a small background sample is saved to `models/explainer_background.csv`. During inference, SHAP explains the fitted pipeline for the positive churn class and returns top feature contributions. If SHAP is unavailable, the service falls back to rule-based local explanations.

These explanations are designed for decision support. A positive contribution means the feature pushed the individual prediction toward churn relative to the background sample; it does not prove the feature caused churn.

### Tenure Sensitivity Check

Because tenure dominates the permutation-importance plot, the project includes a sensitivity training path that excludes `tenure`:

```bash
python -m churn_prediction.train --data "data/raw/E Commerce Dataset.xlsx" --without-tenure
```

This creates separate no-tenure artifacts and metrics. The comparison helps determine whether strong model performance is broadly supported by customer behaviour features or overly dependent on tenure.

## Ethical Considerations

Some features may encode sensitive or proxy information, including gender, city tier, and marital status. These features can improve predictive performance but may also create fairness concerns.

Recommended mitigations:

- evaluate performance by segment
- compare models with and without sensitive/proxy variables
- use explanations to support review, not automatic action
- document intervention policies
- give customers beneficial interventions rather than punitive treatment

## Limitations

- Predictive explanations are not causal explanations.
- Historical churn patterns may change after new pricing, campaigns, or market events.
- The included sample dataset is too small for meaningful performance claims.
- Model performance depends on the quality and freshness of customer data.
- The very high holdout performance requires extra validation for leakage, duplicate records, and feature timing before production use.

## Monitoring Recommendations

Monitor:

- input feature drift
- churn base rate drift
- prediction distribution drift
- calibration drift
- precision and recall on newly labelled data
- complaint-heavy and low-tenure customer segments

Retrain when performance degrades, data definitions change, or the business introduces new retention strategies.

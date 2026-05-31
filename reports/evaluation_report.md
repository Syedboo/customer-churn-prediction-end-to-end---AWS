# Evaluation Report

Run `python -m churn_prediction.train` to regenerate the latest metrics and plots.

The generated `models/metrics.json` file records model comparison results, and `reports/figures/` contains confusion matrix, ROC/PR curves, and permutation importance.

## Current Results

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

Permutation importance shows that tenure is the dominant signal, followed by complaint history, number of addresses, cashback amount, and satisfaction score.

## Interpretation Guidance

- Prefer average precision and churn recall over raw accuracy for imbalanced churn data.
- Review false negatives carefully because they are customers the business failed to identify as churn risks.
- Review false positives in terms of intervention cost. A low-cost email campaign can tolerate more false positives than a costly one-to-one retention call.
- Use feature importance as decision support, not causal evidence.
- Because performance is very high, validate leakage risk and test on an out-of-time holdout before production use.

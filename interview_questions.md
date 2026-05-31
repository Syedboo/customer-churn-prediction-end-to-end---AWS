# Interview Questions and Answers

## 1. What business problem does this project solve?

It predicts which customers are likely to churn so the organisation can prioritise retention actions. The stakeholder value is not just prediction accuracy; it is earlier intervention, better campaign targeting, and more efficient use of customer success resources.

## 2. Why is accuracy not enough?

Churn is usually imbalanced. A model can achieve high accuracy by predicting that most customers will stay, while missing the customers who actually churn. I used average precision, recall, F1, balanced accuracy, and confusion matrices to better evaluate minority-class performance.

## 3. How did you avoid data leakage?

The train-test split happens before preprocessing is fitted. Imputation, encoding, scaling, and modelling are inside a pipeline that is fitted only on the training data. The saved pipeline is then reused at inference time.

## 4. Why compare multiple models?

A simple baseline helps establish whether complexity is justified. Logistic regression gives interpretability, random forest provides a nonlinear benchmark, and gradient boosting often performs strongly on tabular data. This makes the modelling decision evidence-based.

## 5. Why might XGBoost be a good final model?

XGBoost performs well on structured tabular data, handles nonlinear interactions, supports class imbalance controls, and usually provides strong ranking performance. However, I still compare it against simpler models and choose based on validation metrics.

## 6. What metric would you optimise for stakeholders?

It depends on intervention cost. If outreach is cheap, I would favour recall to catch more churners. If outreach is expensive, I would favour precision or an expected-value threshold based on campaign cost, churn probability, and customer lifetime value.

## 7. How would you explain predictions to non-technical users?

I would show a churn probability, a risk band, and a short list of factors such as complaint history, low tenure, low satisfaction, or low order count. I would clearly say these are predictive signals, not proven causes.

## 8. How would you productionise this?

I would serve the saved pipeline through FastAPI, validate inputs with Pydantic, containerise the service with Docker, add automated tests, monitor drift and performance, and retrain on a scheduled or trigger-based cadence.

## 9. What risks should be considered?

Risks include data leakage, poor calibration, drift, overfitting, biased outcomes for customer groups, and treating correlation as causation. The project addresses these with pipelines, imbalanced metrics, documentation, and responsible AI notes.

## 10. How does this fit a university AI collaboration centre role?

It demonstrates the full applied AI lifecycle: translating a stakeholder problem into a model, evaluating rigorously, explaining outputs, documenting limitations, and deploying a usable prototype. It also shows the ability to communicate across technical and non-technical audiences.

## 11. What would you improve with more time?

I would add SHAP-based local explanations, campaign uplift modelling, fairness analysis by segment, calibration monitoring, CI/CD, model registry integration, and a live feedback loop from retention campaign outcomes.

## 12. How would you validate real-world impact?

I would run an A/B test or stepped-wedge pilot where high-risk customers receive targeted interventions. The impact metrics would include churn reduction, incremental revenue retained, campaign cost, customer satisfaction, and false-positive intervention burden.

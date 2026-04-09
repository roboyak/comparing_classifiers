# Module 17: Comparing Classifiers — Bank Marketing Campaign

## Summary of Findings

This project compares four classification models (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines) to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's telephone marketing campaigns. The analysis follows the **CRISP-DM** methodology.

### Business Problem
The bank's marketing campaigns have an ~11% success rate, meaning nearly 9 out of 10 calls fail to convert. The goal is to build a predictive model that identifies high-probability subscribers so the bank can target its outreach more efficiently, reduce wasted calls, and increase ROI.

### Dataset
- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size:** 41,188 contacts across 17 campaigns (May 2008 - Nov 2010)
- **Features:** 20 input variables (client demographics, campaign details, socioeconomic indicators)
- **Target:** Binary — subscribed to term deposit (yes/no), ~11.3% positive class

### Key Results — Default Models

| Model | Train Time | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|---|
| Logistic Regression | 3.2s | 90.95% | 91.27% | Best generalization (0.3% gap) |
| KNN | 0.01s | 93.11% | 90.53% | Moderate overfitting |
| Decision Tree | 0.12s | 100.00% | 89.20% | Severe overfitting (10.8% gap) |
| SVM | 6.8s | 89.76% | 89.78% | Slowest, no overfitting |

- **Baseline (majority class):** 88.73% — predicts "no" every time, zero business value
- All models beat the baseline
- **Decision Tree** memorized the training data (100% train accuracy) — fixed via hyperparameter tuning

### Key Results — Tuned Models (GridSearchCV + StandardScaler)

| Model | Best Params | CV Accuracy | Test Accuracy |
|---|---|---|---|
| Logistic Regression | C=10, solver=liblinear | 90.96% | 91.48% |
| KNN | n_neighbors=15, uniform weights | 89.84% | 90.38% |
| Decision Tree | max_depth=5 | 91.22% | 91.67% |
| SVM | C=10, kernel=rbf | 91.06% | 91.39% |

- **Decision Tree** improved the most (+2.5pp) after constraining `max_depth`
- **Logistic Regression** selected as the recommended model for deployment

### Benchmark vs Production Model

| Configuration | Test Accuracy | Use Case |
|---|---|---|
| **Benchmark** (with `duration`) | 91.27% | Assignment comparison |
| **Production** (without `duration`) | 90.15% | Real-world deployment |

`duration` (call length) is the strongest single predictor but is only known after the call ends — it cannot be used to decide who to call next. The production model drops ~1.1pp in accuracy but represents realistic deployment performance.

### Evaluation Metrics
**Primary metric: AUC-ROC** — because the bank's real task is ranking clients by subscription probability before each campaign. AUC-ROC measures this ranking quality independent of decision threshold.

**Secondary metric: F1-Score** — balances precision (don't waste calls on false leads) and recall (don't miss potential subscribers).

Accuracy alone is misleading with 88.7% class imbalance — a model predicting "no" every time achieves 88.7% accuracy with zero business value.

### Feature Engineering Insights
- **Subscription rate by category:** `retired` and `student` clients convert at much higher rates; `month=mar` and `poutcome=success` are strong positive signals; `cellular` contact outperforms `telephone`
- **Feature distributions:** `duration` separates classes well; `pdays=999` dominates ~96% of data (never previously contacted)
- **Socioeconomic columns** renamed for readability (e.g., `emp.var.rate` -> `employment_var_rate`)

### Logistic Regression Coefficient Interpretation
Top positive predictors: `month_mar` (March campaigns), `consumer_price_idx`, `month_aug`, `job_retired` — the bank should time campaigns in March/August and prioritize retired clients.

Top negative predictors: `euribor_3m_rate` (higher interbank rates), `num_employed`, `contact_telephone` — less favorable economic conditions and telephone contact reduce subscription likelihood.

### Decision Tree Overfitting Analysis (Extra Credit)
The default Decision Tree hit 100% train / 89.2% test accuracy. Analysis of accuracy vs `max_depth` revealed:
- Optimal depth is **7** — achieves 91.9% test accuracy with only 0.3% train/test gap
- Beyond depth 7, the tree creates exponentially more leaves that memorize noise
- Constraining depth converts a lookup table into a generalizable model

### Actionable Insights
1. **Prioritize warm leads** — clients with previous campaign success are far more likely to subscribe
2. **Time campaigns strategically** — March and August show the highest coefficient values; end-of-quarter months perform well
3. **Train agents for engagement** — longer, meaningful conversations correlate strongly with conversion
4. **Use cellular over telephone** — cellular contact has a significantly higher conversion rate
5. **Monitor economic conditions** — favorable employment variation rates and low euribor rates increase receptivity
6. **Target retirement-age clients** — `retired` is one of the strongest positive predictors

### Recommendations and Next Steps
- Deploy **Logistic Regression** for its interpretability, speed, and competitive accuracy
- Remove `duration` for production deployment (data leakage)
- Address class imbalance with SMOTE or `class_weight='balanced'` to improve recall
- Engineer new features (e.g., `was_previously_contacted` from `pdays`, age bins)
- Try ensemble methods (Random Forest, Gradient Boosting) for further improvement
- A/B test the model in a pilot campaign measuring actual conversion lift
- Retrain periodically as economic conditions and client demographics shift

## Jupyter Notebook

[View the full analysis notebook](module17_starter/prompt_III.ipynb)

## Repository Structure
```
├── README.md                          # This file — summary of findings
└── module17_starter/
    └── prompt_III.ipynb               # Main analysis notebook
```

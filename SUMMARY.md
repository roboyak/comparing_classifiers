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

### Key Results

| Model | Default Test Accuracy | Tuned Test Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 91.3% | ~91% | Best speed/accuracy/interpretability balance |
| KNN | 90.5% | ~90% | Moderate, no explicit training phase |
| Decision Tree | 89.2% | ~90% | 100% train accuracy (overfit) before tuning |
| SVM | 89.8% | ~91% | Slowest to train by far |

- **Baseline (majority class):** 88.7% accuracy — predicts "no" every time, zero business value
- All tuned models beat the baseline
- **Decision Tree** benefited most from hyperparameter tuning — overfitting reduced dramatically via `max_depth` constraint
- **Logistic Regression** selected as the recommended model

### Benchmark vs Production Model

We trained Logistic Regression in two configurations to separate benchmark performance from realistic deployment:

| Configuration | Test Accuracy | AUC-ROC | Use Case |
|---|---|---|---|
| **Benchmark** (with `duration`) | ~91% | High | Assignment comparison |
| **Production** (without `duration`) | Lower | Moderate | Real-world deployment |

`duration` (call length) is the strongest single predictor but is only known *after* the call ends — it cannot be used to decide who to call next. The production model shows what the bank can realistically expect.

### Feature Engineering Insights
- **Subscription rate by category:** `retired` and `student` clients convert at much higher rates; `month=mar` and `poutcome=success` are strong positive signals; `cellular` contact outperforms `telephone`
- **Feature distributions:** `duration` separates classes well; `pdays=999` dominates ~96% of data (never previously contacted)
- **Socioeconomic columns** renamed for readability (e.g., `emp.var.rate` -> `employment_var_rate`)

### Logistic Regression Coefficient Interpretation
The model's coefficients reveal which features most strongly influence subscription likelihood:
- **Positive predictors:** `duration`, `poutcome_success`, certain months (March), and client segments (retired, student)
- **Negative predictors:** High `euribor_3m_rate`, `num_employed`, and certain contact patterns

### Decision Tree Overfitting Analysis (Extra Credit)
The default Decision Tree achieved 100% training accuracy but only ~89% on test data. Analysis of accuracy vs `max_depth` revealed the classic bias-variance tradeoff — an unrestricted tree creates thousands of hyper-specific rules that memorize training data rather than learning generalizable patterns. Constraining depth to the optimal value closes most of the overfitting gap.

### Actionable Insights
1. **Prioritize warm leads** — clients with previous campaign success are far more likely to subscribe
2. **Time campaigns strategically** — end-of-quarter months (March, June, September, December) show higher success rates
3. **Train agents for engagement** — longer, meaningful conversations correlate strongly with conversion
4. **Monitor economic conditions** — favorable employment variation rates and low euribor rates increase receptivity
5. **Use the model to score prospects** — rank clients by subscription probability before each campaign to focus on the top deciles

### Recommendations and Next Steps
- Deploy **Logistic Regression** for its interpretability, speed, and competitive accuracy
- Remove `duration` for production (data leakage)
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

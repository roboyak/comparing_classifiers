# Module 17: Comparing Classifiers — Bank Marketing Campaign

## Business Problem
The bank's marketing campaigns have an ~11% success rate, meaning nearly 9 out of 10 calls fail to convert. The goal is to build a predictive model that identifies high-probability subscribers so the bank can target its outreach more efficiently, reduce wasted calls, and increase ROI.

### Methodolgy

This project compares four classification models (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines) to determine the best model for predicting whether a client will subscribe to a term deposit based on data from a Portuguese bank's telephone marketing campaigns. The analysis follows the **CRISP-DM** methodology.

### Dataset
- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Size:** 41,188 contacts across 17 campaigns (May 2008 - Nov 2010)
- **Features:** 20 input variables (client demographics, campaign details, socioeconomic indicators)
- **Target:** Binary — subscribed to term deposit (yes/no), ~11.3% positive class

## Jupyter Notebook

[View the full analysis notebook](prompt_III.ipynb)

### Key Findings

**1. All four tuned models outperform the 88.7% majority-class baseline** on production-safe features (no duration leakage).

**2. Decision Tree overfits severely without regularization** — ~100% train accuracy with default settings. Constraining `max_depth` is essential.

**3. Removing `duration` reduces performance** but the models remain useful. The production model represents what the bank can actually deploy.

**4. Average Precision is the right primary metric** — accuracy is misleading with 88.7% class imbalance.

### Actionable Insights for the Marketing Team

1. **Prioritize warm leads** — clients with previous campaign success (`poutcome=success`) are the strongest positive signal
2. **Target retirement-age and student segments** — these groups convert at much higher rates than average
3. **Use cellular over telephone** — cellular contact has significantly higher conversion
4. **Time campaigns strategically** — certain months show consistently higher conversion rates
5. **Monitor economic conditions** — euribor rate and employment indicators significantly influence outcomes
6. **Use the model to rank prospects** — score all clients before each campaign and focus resources on the top deciles

### Recommended Model
**Logistic Regression** — best balance of accuracy, speed, interpretability, and stability. Marketing managers can directly read coefficient weights to understand what drives subscriptions.

### Next Steps
1. **Deploy production model** (without duration) for campaign targeting
2. **Address class imbalance** — SMOTE, `class_weight='balanced'`, or threshold tuning
3. **Engineer additional features** — age bins, economic trend indicators, interaction terms
4. **Try ensemble methods** — Random Forest and Gradient Boosting for nonlinear patterns
5. **A/B test** — pilot campaign comparing model-selected vs random contacts
6. **Retrain periodically** — economic conditions and client demographics shift over time

## Repository Structure
```
├── README.md                          # This file — summary of findings
├── prompt_III.ipynb                          # Main analysis notebook
└── data/
    └── bank-additional-full.csv              # Data set being studied
```

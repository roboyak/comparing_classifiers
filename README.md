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

- [View the full analysis](prompt_III.ipynb)

### Key Findings

**1. All four tuned models edge past the 88.7% majority-class baseline** on production-safe features (no duration leakage), but the accuracy gains are marginal (~1.1-1.5 pp). The real differentiation shows up in Average Precision (~0.40-0.47), which captures how well models identify the minority positive class.

**2. Decision Tree overfits severely without regularization** — ~100% train accuracy but test accuracy (84.2%) actually falls *below* baseline with default settings. Constraining `max_depth` is essential.

**3. Removing `duration` reduces performance** but the models remain useful. The production model represents what the bank can actually deploy.

**4. In the expanded SVC search, `rbf` beats `linear` decisively**. The best SVM in the latest run is `C=0.1, kernel='rbf'`, but it still trails Logistic Regression on tuned Average Precision.

**5. Average Precision is the right primary metric** — accuracy is misleading with 88.7% class imbalance. A model that predicts "no" every time scores 88.7% accuracy with zero business value.

### What We Learned

- **Accuracy is deceptive under class imbalance.** With only ~11% positive cases, every model "looks good" at ~90% accuracy — but that's barely above always predicting "no." Average Precision and the precision-recall tradeoff are essential for evaluating models on imbalanced business problems like this one.
- **Feature leakage can inflate results dramatically.** `duration` (call length) is only known *after* a call ends, so including it in a predictive model is cheating. Removing it drops AUC-ROC from 0.94 → 0.81 and AP from 0.62 → 0.47 — a sobering reminder to think carefully about what information is available at prediction time.
- **Simple models can still win.** In the SVC variant, SVM gets a tiny bump in untuned accuracy, but Logistic Regression remains the better production model overall: it is dramatically faster, more interpretable, and still leads on the tuned ranking metric (Average Precision). When we explicitly tune `kernel in ['rbf', 'linear']`, `rbf` wins and `linear` falls off sharply, so the extra SVM complexity still does not pay off here.
- **Overfitting is not just a textbook concept.** The default Decision Tree memorized the training data (~100% train accuracy) and performed *worse than the baseline* on test data (84.2% vs 88.7%). Regularization via `max_depth` and `min_samples_leaf` turned it into a competitive model.

### How This Helps the Bank

The bank currently converts ~11% of marketing calls — roughly 1 in 9. Even a modest improvement in targeting has significant ROI impact at scale.

1. **Score and rank prospects before each campaign** — instead of calling clients at random, use the model to prioritize the top deciles. Even with an AP of 0.47, concentrating effort on high-probability leads can substantially reduce wasted calls.
2. **Prioritize warm leads** — clients with previous campaign success (`poutcome=success`) are the strongest positive signal. The bank should re-engage past converters first.
3. **Target high-conversion segments** — retirement-age and student demographics convert at higher rates. Tailored messaging for these groups can amplify results.
4. **Use cellular over telephone** — cellular contact shows significantly higher conversion rates. Shift call allocation accordingly.
5. **Time campaigns to economic conditions** — euribor rate and employment indicators significantly influence outcomes. Launching campaigns during favorable economic windows improves response rates.
6. **Adopt Logistic Regression as the production model** — it's fast, interpretable (marketing managers can read the coefficient weights directly), and remains the strongest overall choice once we prioritize tuned Average Precision over raw untuned accuracy. Interpretability matters: the team needs to understand *why* a client is flagged, not just that they are.

### Next Steps
1. **Deploy production model** (without duration) for campaign targeting
2. **Address class imbalance** — SMOTE, `class_weight='balanced'`, or threshold tuning to improve recall on the minority class
3. **Engineer additional features** — age bins, economic trend indicators, interaction terms
4. **Try ensemble methods** — Random Forest and Gradient Boosting for nonlinear patterns
5. **A/B test** — pilot campaign comparing model-selected vs random contacts to measure real-world lift
6. **Retrain periodically** — economic conditions and client demographics shift over time

## Repository Structure
```
├── README.md                          # This file — summary of findings
├── prompt_III.ipynb                   # Jupyter notebook with detailed findings
└── data/
    └── bank-additional-full.csv       # Data set being studied
```

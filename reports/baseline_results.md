# Baseline Model Results — Member 2 (W1D2)
## FactoryGuard AI - NASA CMAPSS Dataset

## Models Trained
Both models use class_weight='balanced' to handle class imbalance.

## PR-AUC Scores (Key Metric)
| Model | PR-AUC Score |
|-------|-------------|
| Logistic Regression | 0.1298 |
| Random Forest | 0.2647 |
| **Winner** | **Random Forest** |

## What is PR-AUC?
- PR-AUC = Precision-Recall Area Under Curve
- We use this NOT accuracy because data is imbalanced (87% healthy, 12% failure)
- Score ranges 0 to 1 -- higher is better
- These baseline scores are the BENCHMARK
- Member 4 (Chemala) must beat these with XGBoost/LightGBM

## Train/Test Split
- Training rows : 19,712
- Testing rows  : 4,928
- Features used : 24

## Next Step
- Member 4 (Chemala): Use XGBoost or LightGBM to beat PR-AUC of 0.2647
- Shariq: Load Random Forest model from models/rf_baseline.pkl for SHAP analysis

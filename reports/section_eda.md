# EDA Findings - Member 2
## FactoryGuard AI - NASA CMAPSS Dataset

- Dataset: 24,640 readings from 100 engines
- failure = 1 when RUL <= 30 cycles
- failure = 0 when RUL > 30 cycles
- Healthy: 21,540 (87.4%), Near-failure: 3,100 (12.6%)
- Dataset is imbalanced - use PR-AUC not accuracy
- Key sensors: s2, s3, s4, s7, s11, s15

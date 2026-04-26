"""
FactoryGuard AI — Member 2: Baseline Models
============================================
Task W1D2: Train two baseline models and evaluate with PR-AUC

Models:
  1. Logistic Regression  (class_weight='balanced')
  2. Random Forest        (class_weight='balanced')

Metric: PR-AUC (Precision-Recall Area Under Curve)
       NOT accuracy — because data is imbalanced!

HOW TO RUN:
  python baseline_models.py

OUTPUT:
  - PR-AUC scores for both models
  - reports/plots/member2_baseline_comparison.png
  - reports/plots/member2_pr_curves.png
  - reports/baseline_results.md  (share with team)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

os.makedirs("reports/plots", exist_ok=True)
os.makedirs("models",        exist_ok=True)

print("=" * 60)
print("  MEMBER 2 — Baseline Models (W1D2)")
print("  Logistic Regression vs Random Forest")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD FEATURED DATA (from your EDA work)
# ══════════════════════════════════════════════════════════════════════════════
print("\n📂 STEP 1: Loading data...")

df = pd.read_csv("data/train_FD001_with_RUL.csv")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {df.shape[1]}")
print(f"   Failure rate: {df['failure'].mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: PREPARE FEATURES AND LABEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n⚙️  STEP 2: Preparing features...")

# Drop columns that are NOT features
# unit_nr, time_cycles, max_cycles, RUL are not sensor readings
drop_cols = ['unit_nr', 'time_cycles', 'max_cycles', 'RUL', 'failure']
drop_cols = [c for c in drop_cols if c in df.columns]

# X = all sensor columns (s1 to s21 + op settings)
# y = failure label (0 or 1)
X = df.drop(columns=drop_cols)
y = df['failure']

print(f"   Features : {X.shape[1]} columns")
print(f"   Feature names: {list(X.columns[:5])} ...")
print(f"   Label    : failure (0=healthy, 1=near-failure)")
print(f"   Class split: {y.value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n✂️  STEP 3: Splitting data...")

# stratify=y makes sure both splits have same % of failures
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.2,      # 80% train, 20% test
    random_state= 42,
    stratify    = y,        # keep same failure ratio in both splits
)

print(f"   Train : {X_train.shape[0]:,} rows")
print(f"   Test  : {X_test.shape[0]:,} rows")
print(f"   Train failure rate: {y_train.mean()*100:.1f}%")
print(f"   Test  failure rate: {y_test.mean()*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: SCALE FEATURES (needed for Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════
print("\n📏 STEP 4: Scaling features...")

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("   StandardScaler applied ✅")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: MODEL 1 — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n🤖 STEP 5: Training Model 1 — Logistic Regression...")
print("   class_weight='balanced' handles the imbalance automatically")

lr_model = LogisticRegression(
    class_weight = 'balanced',   # gives more weight to rare failures
    max_iter     = 1000,
    random_state = 42,
)
lr_model.fit(X_train_scaled, y_train)

# Get probability predictions (not just 0/1)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
lr_preds = lr_model.predict(X_test_scaled)

# PR-AUC score — this is the key metric
lr_prauc = average_precision_score(y_test, lr_probs)

print(f"\n   ── Logistic Regression Results ──")
print(f"   PR-AUC Score : {lr_prauc:.4f}  ← KEY METRIC")
print(f"\n   Classification Report:")
print(classification_report(y_test, lr_preds,
      target_names=['Healthy(0)', 'Near-Failure(1)']))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: MODEL 2 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("\n🌲 STEP 6: Training Model 2 — Random Forest...")
print("   This may take 1-2 minutes...")

rf_model = RandomForestClassifier(
    n_estimators = 100,           # 100 decision trees
    class_weight = 'balanced',    # handles imbalance
    random_state = 42,
    n_jobs       = -1,            # use all CPU cores
)
rf_model.fit(X_train, y_train)    # Random Forest doesn't need scaling

# Get probability predictions
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = rf_model.predict(X_test)

# PR-AUC score
rf_prauc = average_precision_score(y_test, rf_probs)

print(f"\n   ── Random Forest Results ──")
print(f"   PR-AUC Score : {rf_prauc:.4f}  ← KEY METRIC")
print(f"\n   Classification Report:")
print(classification_report(y_test, rf_preds,
      target_names=['Healthy(0)', 'Near-Failure(1)']))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: COMPARE BOTH MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  📊 MODEL COMPARISON")
print("=" * 60)
print(f"  Logistic Regression PR-AUC : {lr_prauc:.4f}")
print(f"  Random Forest PR-AUC       : {rf_prauc:.4f}")

winner = "Random Forest" if rf_prauc > lr_prauc else "Logistic Regression"
print(f"\n  🏆 Better baseline model: {winner}")
print(f"  ➡ Tell Member 4 (Chemala) to beat this with XGBoost!")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8: CHART 1 — PR CURVES FOR BOTH MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 STEP 8: Saving charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Baseline Models — Precision-Recall Curves", fontsize=14)

models_info = [
    ("Logistic Regression", lr_probs, lr_prauc, "steelblue"),
    ("Random Forest",       rf_probs, rf_prauc, "seagreen"),
]

for ax, (name, probs, prauc, color) in zip(axes, models_info):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ax.plot(recall, precision, color=color, lw=2,
            label=f"PR-AUC = {prauc:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color=color)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("reports/plots/member2_pr_curves.png", dpi=130)
plt.close()
print("   ✅  reports/plots/member2_pr_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9: CHART 2 — BAR COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

models  = ["Logistic\nRegression", "Random\nForest"]
scores  = [lr_prauc, rf_prauc]
colors  = ["steelblue", "seagreen"]

bars = ax.bar(models, scores, color=colors, width=0.4,
              edgecolor="white", linewidth=1.2)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{score:.4f}", ha="center",
            fontsize=13, fontweight="bold")

ax.set_title("Baseline Model Comparison — PR-AUC Score\n"
             "(Higher is better | XGBoost should beat these)",
             fontsize=12)
ax.set_ylabel("PR-AUC Score")
ax.set_ylim([0, 1.1])
ax.axhline(y=max(scores), color="red", linestyle="--",
           alpha=0.5, label=f"Best baseline: {max(scores):.4f}")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("reports/plots/member2_baseline_comparison.png", dpi=130)
plt.close()
print("   ✅  reports/plots/member2_baseline_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10: SAVE RESULTS FOR TEAM
# ══════════════════════════════════════════════════════════════════════════════
results = f"""# Baseline Model Results — Member 2 (W1D2)
## FactoryGuard AI - NASA CMAPSS Dataset

## Models Trained
Both models use class_weight='balanced' to handle class imbalance.

## PR-AUC Scores (Key Metric)
| Model | PR-AUC Score |
|-------|-------------|
| Logistic Regression | {lr_prauc:.4f} |
| Random Forest | {rf_prauc:.4f} |
| **Winner** | **{winner}** |

## What is PR-AUC?
- PR-AUC = Precision-Recall Area Under Curve
- We use this NOT accuracy because data is imbalanced (87% healthy, 12% failure)
- Score ranges 0 to 1 -- higher is better
- These baseline scores are the BENCHMARK
- Member 4 (Chemala) must beat these with XGBoost/LightGBM

## Train/Test Split
- Training rows : {X_train.shape[0]:,}
- Testing rows  : {X_test.shape[0]:,}
- Features used : {X_train.shape[1]}

## Next Step
- Member 4 (Chemala): Use XGBoost or LightGBM to beat PR-AUC of {max(scores):.4f}
- Shariq: Load Random Forest model from models/rf_baseline.pkl for SHAP analysis
"""

with open("reports/baseline_results.md", "w", encoding="utf-8") as f:
    f.write(results)
print("   ✅  reports/baseline_results.md")

# Save both models for the team to use
joblib.dump(lr_model, "models/lr_baseline.pkl")
joblib.dump(rf_model, "models/rf_baseline.pkl")
joblib.dump(scaler,   "models/scaler.pkl")
print("   ✅  models/lr_baseline.pkl")
print("   ✅  models/rf_baseline.pkl")
print("   ✅  models/scaler.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ✅  W1D2 COMPLETE!")
print("=" * 60)
print(f"""
  Results Summary:
  ─────────────────────────────────────────
  Logistic Regression PR-AUC : {lr_prauc:.4f}
  Random Forest PR-AUC       : {rf_prauc:.4f}
  Best baseline              : {winner}
  ─────────────────────────────────────────

  Files saved:
  reports/plots/member2_pr_curves.png
  reports/plots/member2_baseline_comparison.png
  reports/baseline_results.md
  models/lr_baseline.pkl
  models/rf_baseline.pkl
  models/scaler.pkl

  NOW RUN:
  git add .
  git commit -m "Add baseline models PR-AUC results — Member 2 W1D2"
  git push
""")

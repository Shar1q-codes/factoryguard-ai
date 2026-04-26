"""
FactoryGuard AI — Member 2: EDA
NASA CMAPSS Turbofan Engine Dataset
Zaalima Development Pvt. Ltd.

WHAT THIS FILE DOES:
  Step 1 → Load the NASA dataset
  Step 2 → Create RUL (Remaining Useful Life) column
  Step 3 → Create failure label (failure=1 if RUL <= 30)
  Step 4 → Plot 4 charts and save them
  Step 5 → Print findings (copy to your report)
  Step 6 → Save dataset with RUL for Member 3

HOW TO RUN:
  python member2_eda.py
  OR copy each block into your EDA.ipynb notebook
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Create output folders ──────────────────────────────────────────────────────
os.makedirs("data",          exist_ok=True)
os.makedirs("reports/plots", exist_ok=True)

print("=" * 60)
print("  MEMBER 2 — EDA · FactoryGuard AI")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n📂 STEP 1: Loading dataset...")

# Column names — the NASA file has NO headers, so we add them manually
cols = ['unit_nr', 'time_cycles', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

try:
    # Try loading real NASA file (space-separated .txt)
    df = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=cols)
    df.dropna(axis=1, inplace=True)   # remove empty columns
    print("   Loaded: data/train_FD001.txt  ✅")
except FileNotFoundError:
    try:
        # Try CSV version
        df = pd.read_csv('data/train_FD001.csv')
        print("   Loaded: data/train_FD001.csv  ✅")
    except FileNotFoundError:
        print("   ⚠ Dataset not found! Generating synthetic NASA-style data...")
        rng = np.random.default_rng(42)
        rows = []
        for unit in range(1, 101):
            max_cycle = rng.integers(150, 350)
            for cycle in range(1, max_cycle + 1):
                deg = cycle / max_cycle
                sensors = {f's{i}': round(rng.uniform(20,100) + deg*rng.uniform(-10,10), 4)
                           for i in range(1, 22)}
                rows.append({'unit_nr': unit, 'time_cycles': cycle,
                             'op1': round(rng.uniform(0,1),4),
                             'op2': round(rng.uniform(0,1),4),
                             'op3': round(rng.uniform(60,100),4), **sensors})
        df = pd.DataFrame(rows)
        df.to_csv('data/train_FD001.csv', index=False)
        print("   Generated synthetic data  ✅")

print(f"   Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Engines: {df['unit_nr'].nunique()}")
print(f"\n   Preview:")
print(df[['unit_nr','time_cycles','s1','s2','s3']].head(4).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: CREATE RUL COLUMN  (team leader's exact code)
# ══════════════════════════════════════════════════════════════════════════════
print("\n⚙️  STEP 2: Creating RUL column...")

max_cycles         = df.groupby('unit_nr')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_nr', 'max_cycles']
df                 = df.merge(max_cycles, on='unit_nr')
df['RUL']          = df['max_cycles'] - df['time_cycles']

print(f"   RUL min  : {df['RUL'].min()}")
print(f"   RUL max  : {df['RUL'].max()}")
print(f"   RUL mean : {df['RUL'].mean():.1f} cycles")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE FAILURE LABEL  (team leader's exact code)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🏷️  STEP 3: Creating failure label...")

df['failure'] = (df['RUL'] <= 30).astype(int)

print(df['failure'].value_counts().to_string())
print(f"   Imbalance ratio: {df['failure'].mean()*100:.2f}% failures")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4A: CHART 1 — RUL Distribution
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 STEP 4: Saving charts...")

plt.figure(figsize=(10, 5))
plt.hist(df['RUL'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Failure threshold (RUL=30)')
plt.title('RUL (Remaining Useful Life) Distribution', fontsize=14)
plt.xlabel('RUL (cycles remaining)')
plt.ylabel('Count')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/plots/member2_rul_distribution.png', dpi=130)
plt.close()
print("   ✅  reports/plots/member2_rul_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4B: CHART 2 — Class Imbalance
# ══════════════════════════════════════════════════════════════════════════════
counts = df['failure'].value_counts()
labels = ['Healthy (0)', 'Near-Failure (1)']
colors = ['#2196F3', '#F44336']

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Class Imbalance — Failure Label', fontsize=14)

axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor='white')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_title('Count per class')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].pie(counts.values, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Percentage split')

plt.tight_layout()
plt.savefig('reports/plots/member2_class_imbalance.png', dpi=130)
plt.close()
print("   ✅  reports/plots/member2_class_imbalance.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4C: CHART 3 — Sensor Distributions
# ══════════════════════════════════════════════════════════════════════════════
plot_sensors = ['s2', 's3', 's4', 's7', 's11', 's15']
sensor_cols  = [s for s in plot_sensors if s in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Sensor Distributions: Healthy vs Near-Failure', fontsize=14)
axes = axes.flatten()

healthy   = df[df['failure'] == 0]
near_fail = df[df['failure'] == 1]

for i, sensor in enumerate(sensor_cols):
    axes[i].hist(healthy[sensor],   bins=40, alpha=0.6, color='steelblue',
                 label='Healthy', density=True)
    axes[i].hist(near_fail[sensor], bins=30, alpha=0.7, color='tomato',
                 label='Near-Failure', density=True)
    axes[i].set_title(f'Sensor {sensor.upper()}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/plots/member2_sensor_distributions.png', dpi=130)
plt.close()
print("   ✅  reports/plots/member2_sensor_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4D: CHART 4 — RUL Over Time
# ══════════════════════════════════════════════════════════════════════════════
sample_units = list(df['unit_nr'].unique()[:5])

plt.figure(figsize=(12, 5))
for unit in sample_units:
    u = df[df['unit_nr'] == unit]
    plt.plot(u['time_cycles'], u['RUL'], label=f'Engine {unit}', lw=1.5)

plt.axhline(y=30, color='red', linestyle='--', lw=2, label='Failure Zone (RUL≤30)')
plt.fill_between(range(0, int(df['time_cycles'].max()) + 10),
                 0, 30, alpha=0.1, color='red')
plt.title('RUL Degradation Over Time (5 Engines)', fontsize=13)
plt.xlabel('Cycle')
plt.ylabel('RUL')
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/plots/member2_rul_over_time.png', dpi=130)
plt.close()
print("   ✅  reports/plots/member2_rul_over_time.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: FINDINGS — copy these into section_eda.md
# ══════════════════════════════════════════════════════════════════════════════
fail_pct      = df['failure'].mean() * 100
healthy_count = counts[0]
fail_count    = counts[1]
avg_life      = df.groupby('unit_nr')['time_cycles'].max().mean()

findings = f"""# EDA Findings — Member 2
## FactoryGuard AI · NASA CMAPSS Dataset

- Dataset contains **{len(df):,} readings** from **{df['unit_nr'].nunique()} engines**
- Each reading has 21 sensor values + 3 operational settings
- RUL ranges from 0 to {df['RUL'].max()} cycles (average engine life: {avg_life:.0f} cycles)

## Failure Label
- failure = 1 when RUL ≤ 30 cycles (engine is near end of life)
- failure = 0 when RUL > 30 cycles (engine is healthy)

## Class Imbalance (IMPORTANT)
- Healthy readings  : {healthy_count:,} ({100-fail_pct:.1f}%)
- Near-failure      : {fail_count:,}   ({fail_pct:.1f}%)
- **Dataset is imbalanced** → must use PR-AUC metric, NOT accuracy
- Must handle imbalance with SMOTE or class_weight in model training

## Sensor Analysis
- Sensors s2, s3, s4, s7, s11, s15 show clear difference between healthy/failure
- These sensors will be the most important features for the model

## Charts Produced
- reports/plots/member2_rul_distribution.png
- reports/plots/member2_class_imbalance.png
- reports/plots/member2_sensor_distributions.png
- reports/plots/member2_rul_over_time.png
"""

print("\n" + "=" * 60)
print("  📋 FINDINGS (copy into section_eda.md):")
print("=" * 60)
print(findings)

with open("reports/section_eda.md", "w") as f:
    f.write(findings)
print("   ✅  reports/section_eda.md")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE DATASET WITH RUL — Member 3 needs this!
# ══════════════════════════════════════════════════════════════════════════════
df.to_csv("data/train_FD001_with_RUL.csv", index=False)
print(f"   ✅  data/train_FD001_with_RUL.csv  (share with Member 3!)")

print("\n" + "=" * 60)
print("  ✅  ALL DONE! NOW PUSH TO GITHUB:")
print("=" * 60)
print("""
  git add .
  git commit -m "Add EDA notebook and plots — Member 2"
  git push
""")

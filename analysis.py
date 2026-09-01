"""
AI-Assisted Mathematical Reasoning and Statistical Outlier Detection
Independent Project — Venkata Srikanth Kompala
Reproducible analysis using synthetic educational data.

Requirements:
    pip install numpy pandas matplotlib scikit-learn
Run:
    python analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# 1. Generate synthetic data
n = 120
study_hours = rng.normal(6.0, 1.35, n).clip(2.5, 9.5)
practice_score = (52 + 5.2 * study_hours + rng.normal(0, 6.5, n)).clip(40, 100)
accuracy = (0.66 + 0.032 * (practice_score - 50) + rng.normal(0, 0.055, n)).clip(0.45, 0.99)

# Deliberate educational anomalies (not real people)
outlier_idx = np.array([8, 27, 54, 71, 93, 111])
study_hours[outlier_idx] = [2.7, 9.4, 3.0, 9.2, 4.0, 8.9]
practice_score[outlier_idx] = [96, 45, 98, 48, 42, 44]
accuracy[outlier_idx] = [0.48, 0.94, 0.46, 0.96, 0.49, 0.95]

df = pd.DataFrame({
    "record_id": np.arange(1, n + 1),
    "study_hours": study_hours,
    "practice_score": practice_score,
    "accuracy": accuracy
})

# 2. Z-score screening
features = ["study_hours", "practice_score", "accuracy"]
z = (df[features] - df[features].mean()) / df[features].std(ddof=0)
df["z_max_abs"] = z.abs().max(axis=1)
df["z_flag"] = df["z_max_abs"] > 2.5

# 3. Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=RANDOM_SEED
)
df["if_label"] = model.fit_predict(df[features])
df["if_score"] = model.decision_function(df[features])
df["if_flag"] = df["if_label"].eq(-1)

# 4. Combined decision rule
df["outlier_flag"] = df["z_flag"] | df["if_flag"]

# 5. Export results
df.to_csv("outlier_analysis_results.csv", index=False)

print("\nSummary statistics:")
print(df[features].describe())

print("\nFlagged records:")
print(df.loc[df["outlier_flag"], ["record_id"] + features + ["z_max_abs", "if_score"]])

print("\nNumber flagged:", int(df["outlier_flag"].sum()))

# 6. Visualisation
plt.scatter(df["study_hours"], df["practice_score"], s=25, alpha=0.65)
flagged = df[df["outlier_flag"]]
plt.scatter(flagged["study_hours"], flagged["practice_score"], s=55, marker="x")
plt.xlabel("Study hours / week")
plt.ylabel("Practice score")
plt.title("Study Hours vs. Practice Score — Flagged Records")
plt.tight_layout()
plt.savefig("study_vs_score.png", dpi=160)
plt.show()

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import argparse
import os
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Random Forest prediction: MGE detection")
parser.add_argument("--input", required=True, help="Input CSV with RF features")
parser.add_argument("--model", required=True, help="Pickle file for Random Forest model")
parser.add_argument("--output", required=True, help="Output CSV with RF predictions")
args = parser.parse_args()

# ----------------------------
# 2. Load RF model
# ----------------------------
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Random Forest model not found at {args.model}")
with open(args.model, "rb") as f:
    rf_model = pickle.load(f)

# ----------------------------
# 3. Load RF features
# ----------------------------
df = pd.read_csv(args.input)

rf_features = ["chr_count", "plasmid_count", "phage_count", "p_value"]
for col in rf_features:
    if col not in df.columns:
        df[col] = 0

X_rf = df[rf_features].values
scaler = StandardScaler()
X_rf_scaled = scaler.fit_transform(X_rf)

# ----------------------------
# 4. Predict
# ----------------------------
rf_labels = rf_model.predict(X_rf_scaled)
rf_probs = rf_model.predict_proba(X_rf_scaled)[:,1]

df["MGE_label"] = rf_labels
df["MGE_prob"] = rf_probs

# ----------------------------
# 5. Save output
# ----------------------------
os.makedirs(os.path.dirname(args.output), exist_ok=True)
df.to_csv(args.output, index=False)

print(f"Random Forest predictions completed. Results saved to: {args.output}")


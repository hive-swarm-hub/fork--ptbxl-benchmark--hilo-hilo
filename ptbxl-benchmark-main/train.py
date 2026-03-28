"""
PTB-XL 12-Lead ECG Classification — Baseline

Classifies 12-lead ECGs into 5 diagnostic superclasses:
  NORM (Normal), MI (Myocardial Infarction), STTC (ST/T Change),
  CD (Conduction Disturbance), HYP (Hypertrophy)

Agents: modify this file to improve the macro-averaged AUROC.

Baseline: per-lead statistical features (96 features) + XGBoost, trained
independently per class with class-weight balancing.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from xgboost import XGBClassifier


def extract_features(X):
    """Extract statistical features from raw ECG signals.

    Args:
        X: numpy array of shape (n_samples, 1000, 12) — 12-lead ECGs at 100Hz

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples, n_timesteps, n_leads = X.shape
    features = []

    for i in range(n_leads):
        lead = X[:, :, i]  # (n_samples, 1000)

        features.append(np.mean(lead, axis=1))
        features.append(np.std(lead, axis=1))
        features.append(np.min(lead, axis=1))
        features.append(np.max(lead, axis=1))
        features.append(sp_stats.skew(lead, axis=1))
        features.append(sp_stats.kurtosis(lead, axis=1))
        # Zero crossings
        zc = np.sum(np.diff(np.sign(lead), axis=1) != 0, axis=1)
        features.append(zc.astype(np.float64))
        # Peak-to-peak amplitude
        features.append(np.max(lead, axis=1) - np.min(lead, axis=1))

    return np.column_stack(features)


def main():
    os.makedirs("predictions", exist_ok=True)

    with open("data/registry.json") as f:
        registry = json.load(f)

    classes = registry["classes"]

    # Load precomputed numpy arrays
    print("Loading data...")
    X_train = np.load("data/X_train.npy")
    X_test = np.load("data/X_test.npy")
    y_train = pd.read_csv("data/y_train.csv")
    test_ids = pd.read_csv("data/y_test_ids.csv")["ecg_id"].values

    # Extract features
    print("Extracting features...")
    F_train = extract_features(X_train)
    F_test = extract_features(X_test)
    print(f"  Feature matrix: {F_train.shape[1]} features")

    # Replace NaN/Inf
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Train per-class XGBoost models
    predictions = {}
    for cls in classes:
        labels = y_train[cls].values
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        spw = n_neg / max(n_pos, 1)

        print(f"  Training {cls} (pos={int(n_pos)}, weight={spw:.1f})...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        model.fit(F_train, labels)
        predictions[cls] = model.predict_proba(F_test)[:, 1]

    # Save predictions with ecg_id for identity-based merging
    pred_df = pd.DataFrame(predictions)
    pred_df.insert(0, "ecg_id", test_ids)
    pred_df.to_csv("predictions/predictions.csv", index=False)
    print(f"Done. Predictions saved to predictions/predictions.csv ({len(pred_df)} rows)")


if __name__ == "__main__":
    main()

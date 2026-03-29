"""
PTB-XL 12-Lead ECG Classification

Classifies 12-lead ECGs into 5 diagnostic superclasses:
  NORM (Normal), MI (Myocardial Infarction), STTC (ST/T Change),
  CD (Conduction Disturbance), HYP (Hypertrophy)

Agents: modify this file to improve the macro-averaged AUROC.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import butter, filtfilt
import lightgbm as lgb


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=100.0, order=4):
    """Apply bandpass filter to ECG signal."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=1)


def extract_features(X):
    """Extract rich features from raw ECG signals.

    Args:
        X: numpy array of shape (n_samples, 1000, 12) — 12-lead ECGs at 100Hz

    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    n_samples, n_timesteps, n_leads = X.shape
    features = []

    # Apply bandpass filter (0.5-40 Hz) — standard ECG clinical range
    X_filt = np.zeros_like(X)
    for i in range(n_samples):
        try:
            X_filt[i] = bandpass_filter(X[i])
        except Exception:
            X_filt[i] = X[i]

    for i in range(n_leads):
        lead = X_filt[:, :, i]       # filtered (n_samples, 1000)

        # --- Statistical features ---
        features.append(np.mean(lead, axis=1))
        features.append(np.std(lead, axis=1))
        features.append(np.min(lead, axis=1))
        features.append(np.max(lead, axis=1))
        features.append(sp_stats.skew(lead, axis=1))
        features.append(sp_stats.kurtosis(lead, axis=1))
        zc = np.sum(np.diff(np.sign(lead), axis=1) != 0, axis=1)
        features.append(zc.astype(np.float64))
        features.append(np.max(lead, axis=1) - np.min(lead, axis=1))
        features.append(np.percentile(lead, 25, axis=1))
        features.append(np.percentile(lead, 75, axis=1))
        features.append(np.sqrt(np.mean(lead ** 2, axis=1)))
        features.append(np.mean(np.abs(lead - np.mean(lead, axis=1, keepdims=True)), axis=1))

        # --- FFT frequency features ---
        fft_vals = np.abs(np.fft.rfft(lead, axis=1))  # (n_samples, 501)
        freqs = np.fft.rfftfreq(n_timesteps, d=1.0 / 100.0)

        # Total power
        features.append(np.sum(fft_vals ** 2, axis=1))
        # Power in frequency bands
        bands = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 40)]
        for flo, fhi in bands:
            mask = (freqs >= flo) & (freqs < fhi)
            features.append(np.sum(fft_vals[:, mask] ** 2, axis=1))
        # Dominant frequency
        features.append(freqs[np.argmax(fft_vals[:, 1:], axis=1) + 1])
        # Spectral entropy
        psd = fft_vals ** 2
        psd_sum = psd.sum(axis=1, keepdims=True) + 1e-10
        psd_norm = psd / psd_sum
        spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=1)
        features.append(spec_entropy)
        # Spectral centroid
        spec_centroid = np.sum(freqs * fft_vals, axis=1) / (np.sum(fft_vals, axis=1) + 1e-10)
        features.append(spec_centroid)
        # Spectral rolloff (85%)
        cumsum = np.cumsum(fft_vals ** 2, axis=1)
        total_power = cumsum[:, -1:] + 1e-10
        rolloff_idx = np.argmax(cumsum >= 0.85 * total_power, axis=1)
        features.append(freqs[rolloff_idx])

    # --- Inter-lead correlation features ---
    lead_pairs = [(0, 1), (0, 6), (1, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
    for i, j in lead_pairs:
        lead_i = X_filt[:, :, i]
        lead_j = X_filt[:, :, j]
        mean_i = np.mean(lead_i, axis=1, keepdims=True)
        mean_j = np.mean(lead_j, axis=1, keepdims=True)
        num = np.sum((lead_i - mean_i) * (lead_j - mean_j), axis=1)
        den = np.sqrt(np.sum((lead_i - mean_i) ** 2, axis=1) * np.sum((lead_j - mean_j) ** 2, axis=1)) + 1e-10
        features.append(num / den)

    # --- Temporal segmentation features ---
    segment_size = n_timesteps // 4
    for seg in range(4):
        start = seg * segment_size
        end = start + segment_size
        for i in range(n_leads):
            seg_data = X_filt[:, start:end, i]
            features.append(np.mean(seg_data, axis=1))
            features.append(np.std(seg_data, axis=1))

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

    # Train per-class LightGBM models
    predictions = {}
    for cls in classes:
        labels = y_train[cls].values
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        spw = n_neg / max(n_pos, 1)

        print(f"  Training {cls} (pos={int(n_pos)}, weight={spw:.1f})...")
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
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

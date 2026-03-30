"""
PTB-XL 12-Lead ECG Classification

Compact 1D ResNet + LightGBM ensemble. Vectorized feature extraction for speed.
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as sp_stats
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import lightgbm as lgb
from catboost import CatBoostClassifier


def augment_batch(x, noise_std=0.05, amp_range=(0.8, 1.2), shift_range=50):
    """Apply random augmentations to a batch of ECG signals (N, 12, T)."""
    # Gaussian noise
    x = x + torch.randn_like(x) * noise_std
    # Random amplitude scaling per lead
    amp = torch.empty(x.shape[0], x.shape[1], 1).uniform_(amp_range[0], amp_range[1])
    x = x * amp
    # Random time shift
    shift = torch.randint(-shift_range, shift_range + 1, (1,)).item()
    if shift != 0:
        x = torch.roll(x, shift, dims=2)
    return x


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + (self.downsample(x) if self.downsample else x))


class ECGResNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, 15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock1D(32, 64, stride=2)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.layer3 = ResBlock1D(128, 128, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.drop   = nn.Dropout(0.3)
        self.fc     = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


def extract_features(X):
    """Vectorized feature extraction — all heavy ops done per-lead not per-sample."""
    n_samples, n_timesteps, n_leads = X.shape
    features = []

    # Bandpass filter: vectorized per-lead
    t0 = time.time()
    nyq = 50.0
    b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    X_filt = np.empty_like(X)
    for i in range(n_leads):
        X_filt[:, :, i] = filtfilt(b, a, X[:, :, i], axis=1)
    print(f"    bandpass: {time.time()-t0:.1f}s")

    t0 = time.time()
    for i in range(n_leads):
        lead = X_filt[:, :, i]
        features += [
            np.mean(lead, axis=1), np.std(lead, axis=1),
            np.min(lead, axis=1), np.max(lead, axis=1),
            sp_stats.skew(lead, axis=1), sp_stats.kurtosis(lead, axis=1),
            np.sum(np.diff(np.sign(lead), axis=1) != 0, axis=1).astype(np.float64),
            np.max(lead, axis=1) - np.min(lead, axis=1),
            np.percentile(lead, 25, axis=1), np.percentile(lead, 75, axis=1),
            np.sqrt(np.mean(lead ** 2, axis=1)),
            np.mean(np.abs(lead - np.mean(lead, axis=1, keepdims=True)), axis=1),
        ]
        fft_vals = np.abs(np.fft.rfft(lead, axis=1))
        freqs = np.fft.rfftfreq(n_timesteps, d=1.0 / 100.0)
        features.append(np.sum(fft_vals ** 2, axis=1))
        for flo, fhi in [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 40)]:
            mask = (freqs >= flo) & (freqs < fhi)
            features.append(np.sum(fft_vals[:, mask] ** 2, axis=1))
        features.append(freqs[np.argmax(fft_vals[:, 1:], axis=1) + 1])
        psd = fft_vals ** 2
        psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
        features.append(-np.sum(psd_norm * np.log(psd_norm + 1e-10), axis=1))
        features.append(np.sum(freqs * fft_vals, axis=1) / (np.sum(fft_vals, axis=1) + 1e-10))
        cumsum = np.cumsum(fft_vals ** 2, axis=1)
        features.append(freqs[np.argmax(cumsum >= 0.85 * (cumsum[:, -1:] + 1e-10), axis=1)])
    print(f"    stats+fft: {time.time()-t0:.1f}s")

    # Wavelet: vectorized per-lead using axis parameter
    t0 = time.time()
    for i in range(n_leads):
        lead = X_filt[:, :, i]
        coeffs = pywt.wavedec(lead, 'db4', level=4, axis=1)
        energies = np.stack([np.sum(c ** 2, axis=1) for c in coeffs], axis=1)  # (N, 5)
        total = energies.sum(axis=1, keepdims=True) + 1e-10
        for lvl in range(5):
            features.append(energies[:, lvl])
            features.append(energies[:, lvl] / total[:, 0])
    print(f"    wavelet: {time.time()-t0:.1f}s")

    # Inter-lead correlations
    for i, j in [(0, 1), (0, 6), (1, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]:
        li, lj = X_filt[:, :, i], X_filt[:, :, j]
        mi = np.mean(li, axis=1, keepdims=True); mj = np.mean(lj, axis=1, keepdims=True)
        num = np.sum((li - mi) * (lj - mj), axis=1)
        den = np.sqrt(np.sum((li - mi)**2, axis=1) * np.sum((lj - mj)**2, axis=1)) + 1e-10
        features.append(num / den)

    # Temporal segments
    seg = n_timesteps // 4
    for s in range(4):
        for i in range(n_leads):
            sd = X_filt[:, s*seg:(s+1)*seg, i]
            features += [np.mean(sd, axis=1), np.std(sd, axis=1)]

    # R-peak/HRV on lead II
    t0 = time.time()
    lead_ii = X_filt[:, :, 1]
    hr_feats = []
    for s in range(n_samples):
        sig = lead_ii[s]
        thresh = np.percentile(sig, 75)
        peaks, _ = find_peaks(sig, height=thresh, distance=40)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / 100.0
            hr_feats.append([len(peaks), 60.0/np.mean(rr), np.std(rr)*1000,
                             np.sqrt(np.mean(np.diff(rr)**2))*1000])
        else:
            hr_feats.append([0.0, 0.0, 0.0, 0.0])
    hr_arr = np.array(hr_feats)
    for c in range(4):
        features.append(hr_arr[:, c])
    print(f"    r-peak: {time.time()-t0:.1f}s")

    # ── Clinical features ────────────────────────────────────────────────────
    # Lead index reference: I=0, II=1, III=2, aVR=3, aVL=4, aVF=5,
    #                       V1=6, V2=7, V3=8, V4=9, V5=10, V6=11

    # Left-lead vs right-lead amplitude ratio (hypertrophy: high left voltage)
    left_leads  = [0, 4, 9, 10, 11]   # I, aVL, V4, V5, V6
    right_leads = [6, 7, 8]           # V1, V2, V3
    left_amp  = np.mean([np.max(np.abs(X_filt[:, :, i]), axis=1) for i in left_leads], axis=0)
    right_amp = np.mean([np.max(np.abs(X_filt[:, :, i]), axis=1) for i in right_leads], axis=0)
    features.append(left_amp)
    features.append(right_amp)
    features.append(left_amp / (right_amp + 1e-6))  # L/R ratio — high in LVH

    # Sokolow-Lyon index (S in V1 + R in V5): key for LV hypertrophy
    s_v1 = np.abs(np.min(X_filt[:, :, 6], axis=1))  # S in V1 (depth of S wave)
    r_v5 = np.max(X_filt[:, :, 10], axis=1)           # R in V5
    features.append(s_v1 + r_v5)  # Sokolow-Lyon, >35mm → LVH

    # T-wave polarity per lead (inverted T is diagnostic for MI, STTC)
    for i in [0, 1, 6, 9, 10, 11]:  # I, II, V1, V4, V5, V6
        lead = X_filt[:, :, i]
        # T wave ~ last 30% of signal (after ST segment)
        t_seg = lead[:, int(0.7 * n_timesteps):]
        features.append(np.mean(t_seg, axis=1))         # T-wave polarity (pos/neg)
        features.append(np.max(np.abs(t_seg), axis=1))  # T-wave amplitude

    # ST segment features (ST elevation/depression key for MI and STTC)
    # ST segment ~ 40-60% of signal (between QRS and T wave)
    st_start = int(0.35 * n_timesteps)
    st_end   = int(0.55 * n_timesteps)
    for i in [0, 1, 6, 9, 10, 11]:
        st = X_filt[:, st_start:st_end, i]
        features.append(np.mean(st, axis=1))   # ST level
        features.append(np.std(st, axis=1))    # ST variability

    # QRS width proxy: zero crossings in middle 30% of signal
    qrs_seg = X_filt[:, int(0.3*n_timesteps):int(0.6*n_timesteps), :]
    for i in [1, 6]:  # Lead II, V1
        qrs = qrs_seg[:, :, i]
        zc = np.sum(np.diff(np.sign(qrs), axis=1) != 0, axis=1)
        features.append(zc.astype(np.float64))

    return np.nan_to_num(np.column_stack(features), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    os.makedirs("predictions", exist_ok=True)

    with open("data/registry.json") as f:
        registry = json.load(f)
    classes = registry["classes"]

    print("Loading data...")
    X_train = np.load("data/X_train.npy")
    X_test  = np.load("data/X_test.npy")
    y_train = pd.read_csv("data/y_train.csv")
    test_ids = pd.read_csv("data/y_test_ids.csv")["ecg_id"].values
    y_train_np = y_train[classes].values.astype(np.float32)

    # ── LightGBM branch ─────────────────────────────────────────────────────
    t_lgb = time.time()
    print("Extracting features for LightGBM...")
    F_train = extract_features(X_train)
    print(f"  train features done ({F_train.shape[1]} feat)")
    F_test = extract_features(X_test)
    print(f"  test features done")

    lgb_preds = {}
    for cls in classes:
        labels = y_train[cls].values
        n_pos = labels.sum(); n_neg = len(labels) - n_pos; spw = n_neg / max(n_pos, 1)
        print(f"  GBM {cls} (w={spw:.1f})...")
        # LightGBM
        lgb_m = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgb_m.fit(F_train, labels)
        # CatBoost
        cb_m = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            scale_pos_weight=spw, random_seed=42, verbose=0, thread_count=-1,
        )
        cb_m.fit(F_train, labels)
        # Average LGB + CatBoost predictions
        lgb_preds[cls] = 0.5 * lgb_m.predict_proba(F_test)[:, 1] + 0.5 * cb_m.predict_proba(F_test)[:, 1]
    print(f"GBM branch total: {time.time()-t_lgb:.1f}s")

    # ── CNN branch ───────────────────────────────────────────────────────────
    t_cnn = time.time()
    X_tr = X_train.transpose(0, 2, 1).astype(np.float32)
    X_te = X_test.transpose(0, 2, 1).astype(np.float32)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    pos_weight = torch.tensor(
        [(y_train_np[:, i] == 0).sum() / max(y_train_np[:, i].sum(), 1)
         for i in range(y_train_np.shape[1])], dtype=torch.float32,
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_train_np)),
        batch_size=128, shuffle=True, num_workers=0,
    )

    torch.manual_seed(42)
    model  = ECGResNet(n_classes=len(classes))
    opt    = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=5e-3, epochs=20, steps_per_epoch=len(loader),
        pct_start=0.2, anneal_strategy='cos',
    )
    # Label smoothing: soft targets help with calibration and reduce overconfidence
    smooth_eps = 0.05
    smooth_targets = lambda y: y * (1 - smooth_eps) + 0.5 * smooth_eps
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("Training 1D ResNet (20 epochs, with augmentation)...")
    model.train()
    for epoch in range(20):
        total_loss = 0.0
        for xb, yb in loader:
            xb = augment_batch(xb)
            opt.zero_grad()
            loss = crit(model(xb), smooth_targets(yb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()
        print(f"  epoch {epoch+1}/20  loss={total_loss/len(loader):.4f}")
    print(f"CNN branch total: {time.time()-t_cnn:.1f}s")

    # Test-Time Augmentation: average over 1 clean + 7 augmented passes
    model.eval()
    X_te_t = torch.from_numpy(X_te)
    n_tta = 8
    preds_sum = np.zeros((len(X_te_t), len(classes)), dtype=np.float32)
    with torch.no_grad():
        # Clean pass
        for i in range(0, len(X_te_t), 256):
            preds_sum[i:i+256] += torch.sigmoid(model(X_te_t[i:i+256])).numpy()
        # Augmented passes (mild aug to stay near natural distribution)
        for _ in range(n_tta - 1):
            for i in range(0, len(X_te_t), 256):
                xb = augment_batch(X_te_t[i:i+256].clone(),
                                   noise_std=0.02, amp_range=(0.92, 1.08), shift_range=20)
                preds_sum[i:i+256] += torch.sigmoid(model(xb)).numpy()
    cnn_preds_arr = preds_sum / n_tta
    cnn_preds = {cls: cnn_preds_arr[:, i] for i, cls in enumerate(classes)}

    # ── Ensemble ─────────────────────────────────────────────────────────────
    predictions = {cls: 0.5 * lgb_preds[cls] + 0.5 * cnn_preds[cls] for cls in classes}

    pred_df = pd.DataFrame(predictions)
    pred_df.insert(0, "ecg_id", test_ids)
    pred_df.to_csv("predictions/predictions.csv", index=False)
    print(f"Done. {len(pred_df)} rows saved.")


if __name__ == "__main__":
    main()
